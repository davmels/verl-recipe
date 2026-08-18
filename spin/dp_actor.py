# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import math
from collections import defaultdict

import numpy as np
import torch
from recipe.spin.core_algos import compute_online_dpo_loss, get_batch_logps

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor import DataParallelPPOActor

__all__ = ["DataParallelPPOActor"]


class SPINDataParallelPPOActor(DataParallelPPOActor):
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        # Fix A (detection): ignore EMPTY multi_modal_inputs placeholders. This run is
        # text-only, but an image-capable dataset schema leaves empty {} placeholders in
        # non_tensor_batch that key-presence alone misreads as multimodal. A false-positive
        # True routes some ranks to the fixed-chunk multimodal path while text-only ranks
        # take the dynamic_bsz path -> divergent micro-batch counts desync the per-forward
        # FSDP all-gathers, AND `indices` is left unbound for the revert below
        # (UnboundLocalError in ref_compute_ref_log_prob). Detect on actual content so all
        # ranks take the same dynamic_bsz packing path.
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys() and any(
            bool(x) for x in data.non_tensor_batch["multi_modal_inputs"]
        )

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            # Fix A: pass dp_group so rearrange_micro_batches all-reduces the micro-batch
            # COUNT to the max across the DP group (same_micro_num_in_dp=True). Without it,
            # uneven per-rank counts desync the per-forward FSDP all-gathers -> NCCL watchdog
            # kills a rank (the ~673s ref_log_prob crash). SP=1 here so the DP group is WORLD;
            # get_reverse_idx below still restores the original sample order, values unchanged.
            dp_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None
            micro_batches, indices = rearrange_micro_batches(
                batch=batch, max_token_len=max_token_len, dp_group=dp_group
            )
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                output = self._forward_micro_batch(micro_batch, temperature=temperature)
                if isinstance(output, dict):
                    log_probs = output["log_probs"]
                else:
                    _, log_probs = output
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        # Fix A (revert guard): only revert when the dynamic_bsz path actually ran.
        # If has_multi_modal_inputs was True the fixed-chunk path was taken and `indices`
        # is unset, so gate the reorder on both flags.
        if use_dynamic_bsz and not has_multi_modal_inputs:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def update_policy_dpo_with_ref(self, data: DataProto):
        """
        Performs the DPO update step using pre-calculated reference log probs
        from an external, periodically updated reference model.
        """
        self.actor_module.train()  # Ensure training mode

        # Reclaim the caching-allocator blocks left fragmented by the preceding
        # generation + ref_log_prob phases before the memory-heavy DPO update. Without
        # this, ~7GB sits "reserved but unallocated" in non-contiguous blocks, so the
        # update's full-width logits transient ([mb, seq, 266752 vocab]) can't find a
        # contiguous slot -> CUDA OOM at scaled_loss.backward() even though the totals
        # fit (device peaks ~90/95). empty_cache returns the stale blocks so the update
        # re-reserves cleanly. NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        # would also defrag but is INCOMPATIBLE with SGLang's TorchMemorySaver (it
        # disables the hybrid sleep/resume and kills the rollout engine at init), so we
        # defragment explicitly here instead of via the allocator flag.
        get_torch_device().empty_cache()

        # --- Retrieve necessary data ---
        try:
            # Expects batch prepared by fit_dpo loop, including reference log probs
            batch_td = data.batch
            chosen_labels = batch_td["chosen_labels"]
            rejected_labels = batch_td["rejected_labels"]
            # ... other needed tensors like chosen/rejected input_ids, attention_mask, position_ids ...

            # === Get PRE-CALCULATED reference log probs from input data ===
            reference_chosen_logps = batch_td["reference_chosen_logps"]  # Should be sequence-level logps
            reference_rejected_logps = batch_td["reference_rejected_logps"]  # Should be sequence-level logps
            # ============================================================

            # Get DPO params from meta_info
            # beta = data.meta_info.get('dpo_beta', 0.1) # Default beta
            beta = self.config.get("dpo_beta", 0.1)  # Default beta
            loss_type = data.meta_info.get("dpo_loss_type", "sigmoid")
            label_smoothing = data.meta_info.get("dpo_label_smoothing", 0.0)
            # reference_free should now be False as we provide ref logps
            reference_free = data.meta_info.get("reference_free", False)  # Default False

        except KeyError as e:
            print(f"ERROR: Missing required key for DPO update (in update_policy_dpo): {e}")
            print(f"Available keys in data.batch: {list(batch_td.keys())}")  # Debug print
            return {}  # Return empty metrics on error
        except Exception as e_data:
            print(f"ERROR accessing data for DPO update (in update_policy_dpo): {e_data}")
            return {}

        # --- Micro-batching Setup ---
        micro_batch_size = self.config.get("ppo_micro_batch_size_per_gpu")
        if micro_batch_size is None:
            # Fallback or default if not set, or raise error
            micro_batch_size = 1  # Example fallback, adjust as needed
            print(f"Warning: 'ppo_micro_batch_size_per_gpu' not set, defaulting to {micro_batch_size}")
            # raise ValueError("Config 'ppo_micro_batch_size_per_gpu' must be set.")

        # Ensure chosen_input_ids exists before getting shape
        if "chosen_input_ids" not in batch_td:
            print("ERROR: 'chosen_input_ids' not found in batch_td for DPO update.")
            return {}
        bsz = batch_td["chosen_input_ids"].shape[0]

        if bsz == 0:
            print("Warning: DPO batch size is 0 in update_policy_dpo. Skipping update.")
            return {"actor/dpo_loss": 0.0, "actor/grad_norm": 0.0}  # Return zero metrics if batch is empty

        num_micro_batches = math.ceil(bsz / micro_batch_size)
        gradient_accumulation_steps = num_micro_batches

        # --- Metrics Accumulation ---
        total_loss = 0.0
        accumulated_metrics = defaultdict(list)
        metrics = {}  # Final metrics dict

        # Per-sequence KL chunks; reduced after the loop for true mean/max/min.
        # Lets us report KL without the trainer's separate policy forward pass.
        kl_chosen_raw_chunks = []
        kl_rejected_raw_chunks = []
        kl_chosen_tok_chunks = []
        kl_rejected_tok_chunks = []

        # --- Zero Gradients ---
        self.actor_optimizer.zero_grad(set_to_none=True)

        # Fix B (bounding-box trim): drop pad columns before each forward. Sequences are
        # left-padded prompt + right-padded response, so real tokens form one contiguous
        # block in the middle; forwarding the full max width materializes a
        # [mb, seq, 266752-vocab] logits transient (~2GB/forward, both chosen+rejected
        # graphs alive for the DPO backward) that is ~85% padding and OOMs the update.
        # Trimming to the union bbox of attention_mask==1 is LOSS-INVARIANT: get_batch_logps
        # sums only over shift_labels!=-100 positions, all of which (plus their preceding
        # logit column) lie inside the box, and position_ids are sliced with the same window
        # so RoPE keeps absolute positions. micro_batch_size is unchanged (loop count == bsz
        # on every rank), so this does NOT change the per-rank micro-batch COUNT and cannot
        # desync the FSDP all-gathers (the pitfall that crashed the earlier length-packed
        # Fix B attempt).
        def _bbox_trim(inputs, labels):
            attn = inputs["attention_mask"]
            cols = attn.any(dim=0)  # [seq]: real in at least one row of the micro-batch
            nz = torch.nonzero(cols, as_tuple=False)
            if nz.numel() == 0:
                return inputs, labels
            lo = int(nz[0].item())
            hi = int(nz[-1].item()) + 1
            return {k: v[:, lo:hi] for k, v in inputs.items()}, labels[:, lo:hi]

        # --- Micro-batch Loop ---
        for i in range(num_micro_batches):
            start_idx = i * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, bsz)
            if start_idx >= end_idx:
                continue

            # Slice the full DPO batch into micro-batches
            # Important: Slice ALL required tensors, including labels and inputs
            micro_batch_chosen_labels = chosen_labels[start_idx:end_idx]
            micro_batch_rejected_labels = rejected_labels[start_idx:end_idx]
            micro_batch_chosen_inputs = {
                "input_ids": batch_td["chosen_input_ids"][start_idx:end_idx],
                "attention_mask": batch_td["chosen_attention_mask"][start_idx:end_idx],
            }
            if "chosen_position_ids" in batch_td:
                micro_batch_chosen_inputs["position_ids"] = batch_td["chosen_position_ids"][start_idx:end_idx]

            micro_batch_rejected_inputs = {
                "input_ids": batch_td["rejected_input_ids"][start_idx:end_idx],
                "attention_mask": batch_td["rejected_attention_mask"][start_idx:end_idx],
            }
            if "rejected_position_ids" in batch_td:
                micro_batch_rejected_inputs["position_ids"] = batch_td["rejected_position_ids"][start_idx:end_idx]

            # Trim pad columns (loss-invariant; see _bbox_trim above). Chosen and rejected
            # are independent forwards so each trims to its own real-token span.
            micro_batch_chosen_inputs, micro_batch_chosen_labels = _bbox_trim(
                micro_batch_chosen_inputs, micro_batch_chosen_labels
            )
            micro_batch_rejected_inputs, micro_batch_rejected_labels = _bbox_trim(
                micro_batch_rejected_inputs, micro_batch_rejected_labels
            )

            length_normalize = data.meta_info.get("length_normalize", False)

            # Determine autocast dtype
            autocast_dtype = torch.bfloat16  # Or get dynamically from config/FSDP settings

            # --- Step 3: Retrieve PRE-CALCULATED reference log probs (NO grad needed) ---
            micro_ref_chosen_logps = reference_chosen_logps[start_idx:end_idx]
            micro_ref_rejected_logps = reference_rejected_logps[start_idx:end_idx]

            # --- Split-graph DPO update (memory fix) ---
            # The joint formulation keeps BOTH the chosen and rejected autograd graphs
            # (checkpointed activations + [seq, 266752-vocab] logits/CE chains) alive
            # until one backward -> ~80GB update peak on a full-length pair -> OOM at
            # 95GB even with an empty GPU at entry. Split into partial backwards so only
            # ONE graph is alive at a time. This is GRADIENT-EXACT for any differentiable
            # loss L(pi_c, pi_r): backward of L(pi_c, pi_r_const) contributes dL/dpi_c and
            # backward of L(pi_c_const, pi_r) contributes dL/dpi_r; FSDP accumulates the
            # two reduce-scatters like ordinary gradient accumulation. Cost: one extra
            # no-grad rejected forward (~+20% update time).

            # PASS 0 (no grad): rejected logps needed as the constant in the chosen pass.
            with torch.no_grad(), torch.autocast(device_type=get_device_name(), dtype=autocast_dtype):
                _ng_out = self.actor_module(**micro_batch_rejected_inputs, use_cache=False)
                policy_rejected_logps_const = get_batch_logps(
                    _ng_out.logits, micro_batch_rejected_labels, average_log_prob=length_normalize
                )
            del _ng_out

            # PASS 1 (grad): chosen forward; loss with rejected held constant; backward
            # immediately so the chosen graph is freed before the rejected grad forward.
            with torch.autocast(device_type=get_device_name(), dtype=autocast_dtype):
                policy_chosen_outputs = self.actor_module(**micro_batch_chosen_inputs, use_cache=False)
                policy_chosen_logps = get_batch_logps(
                    policy_chosen_outputs.logits, micro_batch_chosen_labels, average_log_prob=length_normalize
                )
                loss_chosen = compute_online_dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,  # Has grad
                    policy_rejected_logps=policy_rejected_logps_const,  # Constant (no-grad pass)
                    reference_chosen_logps=micro_ref_chosen_logps,
                    reference_rejected_logps=micro_ref_rejected_logps,
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    reference_free=reference_free,
                )
            del policy_chosen_outputs
            if loss_chosen.requires_grad:
                (loss_chosen / gradient_accumulation_steps).backward()
            policy_chosen_logps = policy_chosen_logps.detach()

            # PASS 2 (grad): rejected forward; loss with chosen detached; backward.
            with torch.autocast(device_type=get_device_name(), dtype=autocast_dtype):
                policy_rejected_outputs = self.actor_module(**micro_batch_rejected_inputs, use_cache=False)
                policy_rejected_logps = get_batch_logps(
                    policy_rejected_outputs.logits, micro_batch_rejected_labels, average_log_prob=length_normalize
                )
                loss = compute_online_dpo_loss(
                    policy_chosen_logps=policy_chosen_logps,  # Detached (grad came from pass 1)
                    policy_rejected_logps=policy_rejected_logps,  # Has grad
                    reference_chosen_logps=micro_ref_chosen_logps,
                    reference_rejected_logps=micro_ref_rejected_logps,
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    reference_free=reference_free,
                )
            del policy_rejected_outputs
            scaled_loss = loss / gradient_accumulation_steps
            if scaled_loss.requires_grad:
                scaled_loss.backward()
            policy_rejected_logps = policy_rejected_logps.detach()

            with torch.no_grad():
                # --- DPO Logits (for metrics; all inputs detached) ---
                pi_logratios = policy_chosen_logps - policy_rejected_logps
                ref_logratios = micro_ref_chosen_logps - micro_ref_rejected_logps
                logits = pi_logratios - ref_logratios  # DPO logits

                # --- Accumulate Metrics ---
                total_loss += loss.item()  # Unscaled loss
                accumulated_metrics["actor/dpo_loss_batch"].append(loss.item())
                accumulated_metrics["actor/dpo_logits_batch"].append(logits.mean().item())
                # Accumulate policy and reference log probs/ratios if needed for debugging
                accumulated_metrics["actor/policy_chosen_logps_batch"].append(policy_chosen_logps.mean().item())
                accumulated_metrics["actor/policy_rejected_logps_batch"].append(policy_rejected_logps.mean().item())
                accumulated_metrics["actor/reference_chosen_logps_batch"].append(micro_ref_chosen_logps.mean().item())
                accumulated_metrics["actor/reference_rejected_logps_batch"].append(
                    micro_ref_rejected_logps.mean().item()
                )
                accumulated_metrics["actor/pi_logratios_batch"].append(pi_logratios.mean().item())
                accumulated_metrics["actor/ref_logratios_batch"].append(ref_logratios.mean().item())
                accumulated_metrics["actor/beta_pi_logratios_batch"].append((beta * pi_logratios).mean().item())
                accumulated_metrics["actor/beta_ref_logratios_batch"].append((beta * ref_logratios).mean().item())
                _pi_mag = pi_logratios.abs().mean().item()
                _ref_mag = ref_logratios.abs().mean().item()
                _denom = _pi_mag + _ref_mag
                accumulated_metrics["actor/ref_fraction_batch"].append(_ref_mag / _denom if _denom > 0 else 0.0)
                accumulated_metrics["actor/rewards_accuracies_batch"].append(
                    (logits > 0).float().mean().item()
                )

            # --- KL diagnostics: reuse the policy/ref logps already computed for
            # the DPO loss, so the trainer needs no separate forward pass. ---
            with torch.no_grad():
                _chosen_tokens = (micro_batch_chosen_labels[..., 1:] != -100).sum(-1).clamp(min=1)
                _rejected_tokens = (micro_batch_rejected_labels[..., 1:] != -100).sum(-1).clamp(min=1)
                _kl_chosen = policy_chosen_logps.detach() - micro_ref_chosen_logps
                _kl_rejected = policy_rejected_logps.detach() - micro_ref_rejected_logps
                if length_normalize:
                    # get_batch_logps returned per-token means -> already per-token KL
                    kl_chosen_tok_chunks.append(_kl_chosen)
                    kl_rejected_tok_chunks.append(_kl_rejected)
                    kl_chosen_raw_chunks.append(_kl_chosen * _chosen_tokens)
                    kl_rejected_raw_chunks.append(_kl_rejected * _rejected_tokens)
                else:
                    # get_batch_logps returned sequence sums -> raw (summed) KL
                    kl_chosen_raw_chunks.append(_kl_chosen)
                    kl_rejected_raw_chunks.append(_kl_rejected)
                    kl_chosen_tok_chunks.append(_kl_chosen / _chosen_tokens)
                    kl_rejected_tok_chunks.append(_kl_rejected / _rejected_tokens)

            # (Backward passes already executed above, one per graph — see split-graph
            # DPO update. Nothing further to backprop here.)

        # --- End Micro-batch Loop ---

        # --- Optimizer Step (after accumulating gradients for all micro-batches) ---
        grad_norm = self._optimizer_step()

        # --- Populate Final Metrics ---
        if num_micro_batches > 0 and bsz > 0:  # Check if any processing happened
            metrics["actor/dpo_loss"] = total_loss / num_micro_batches
            metrics["actor/grad_norm"] = (
                grad_norm.item() if torch.is_tensor(grad_norm) and torch.isfinite(grad_norm) else float("inf")
            )
            # Average other accumulated metrics
            for key, val_list in accumulated_metrics.items():
                if val_list:
                    metrics[key.replace("_batch", "")] = np.mean(val_list)

            # KL metrics, emitted under the same keys the trainer previously
            # computed from a separate forward pass over the subset.
            if kl_chosen_raw_chunks and kl_rejected_raw_chunks:
                _kl_c_raw = torch.cat(kl_chosen_raw_chunks)
                _kl_r_raw = torch.cat(kl_rejected_raw_chunks)
                _kl_c_tok = torch.cat(kl_chosen_tok_chunks)
                _kl_r_tok = torch.cat(kl_rejected_tok_chunks)
                _kl_raw_all = torch.cat([_kl_c_raw, _kl_r_raw])
                _kl_tok_all = torch.cat([_kl_c_tok, _kl_r_tok])
                metrics["kl/policy_vs_ref_raw/mean"] = _kl_raw_all.mean().item()
                metrics["kl/policy_vs_ref_raw/chosen"] = _kl_c_raw.mean().item()
                metrics["kl/policy_vs_ref_raw/rejected"] = _kl_r_raw.mean().item()
                metrics["kl/policy_vs_ref/mean"] = _kl_tok_all.mean().item()
                metrics["kl/policy_vs_ref/max"] = _kl_tok_all.max().item()
                metrics["kl/policy_vs_ref/min"] = _kl_tok_all.min().item()
                metrics["kl/chosen/mean"] = _kl_c_tok.mean().item()
                metrics["kl/rejected/mean"] = _kl_r_tok.mean().item()

            # Calculate accuracy / rewards / margins based on averaged logprobs if desired
            if (
                "actor/policy_chosen_logps" in metrics
                and "actor/policy_rejected_logps" in metrics
                and "actor/reference_chosen_logps" in metrics
                and "actor/reference_rejected_logps" in metrics
            ):
                policy_ratio_mean = metrics["actor/policy_chosen_logps"] - metrics["actor/policy_rejected_logps"]
                ref_ratio_mean = metrics["actor/reference_chosen_logps"] - metrics["actor/reference_rejected_logps"]
                logits_mean = policy_ratio_mean - ref_ratio_mean
                metrics["actor/rewards_chosen"] = beta * (
                    metrics["actor/policy_chosen_logps"] - metrics["actor/reference_chosen_logps"]
                )
                metrics["actor/rewards_rejected"] = beta * (
                    metrics["actor/policy_rejected_logps"] - metrics["actor/reference_rejected_logps"]
                )
                metrics["actor/rewards_accuracies"] = np.mean(
                    accumulated_metrics.get("actor/rewards_accuracies_batch", [0.0])
                )
                metrics["actor/rewards_margins"] = metrics["actor/rewards_chosen"] - metrics["actor/rewards_rejected"]

        else:  # Handle case where no micro-batches were run (e.g., bsz=0)
            metrics["actor/dpo_loss"] = 0.0
            metrics["actor/grad_norm"] = 0.0
            # Initialize other metrics to 0 or NaN as appropriate
            for key in accumulated_metrics.keys():
                metrics[key.replace("_batch", "")] = 0.0
            metrics["actor/rewards_chosen"] = 0.0
            metrics["actor/rewards_rejected"] = 0.0
            metrics["actor/rewards_accuracies"] = 0.0
            metrics["actor/rewards_margins"] = 0.0
            metrics["actor/beta_pi_logratios"] = 0.0
            metrics["actor/beta_ref_logratios"] = 0.0
            metrics["actor/ref_fraction"] = 0.0

        return metrics  # Return aggregated metrics
