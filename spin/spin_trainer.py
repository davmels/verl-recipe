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

import os
import traceback
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

_PAIRS_LOG_DIR = "/iopsstor/scratch/cscs/dmelikidze/verl-training/logs"
os.makedirs(_PAIRS_LOG_DIR, exist_ok=True)
_PAIRS_JOB_ID = os.environ.get("SLURM_JOB_ID", "nojob")
_PAIRS_LOG_FILE = open(
    os.path.join(_PAIRS_LOG_DIR, f"pairs_job{_PAIRS_JOB_ID}_pid{os.getpid()}.log"), "a", buffering=1
)

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from recipe.spin import core_algos
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.metric_utils import compute_throughout_metrics, compute_timing_metrics, process_validation_metrics
from verl.trainer.ppo.utils import Role, WorkerType, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, postprocess_data
from verl.utils.tracking import ValidationGenerationsLogger


def make_spin_collate_fn(tokenizer, max_prompt_length: int, truncation: str = "error"):
    """
    Latest verl's RLHFDataset.__getitem__ no longer tokenizes — it returns only
    `raw_prompt` (chat messages) + `dummy_tensor`. The spin recipe still expects
    `input_ids`/`attention_mask`/`position_ids` in the batch, so we tokenize here.
    """

    def _collate(data_list: list[dict]) -> dict:
        tensors = defaultdict(list)
        non_tensors = defaultdict(list)

        batch_input_ids = []
        batch_attention_mask = []
        raw_prompt_ids_list = []

        for data in data_list:
            messages = data["raw_prompt"]
            prompt_text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            tokenized = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            raw_prompt_ids_list.append(input_ids[0].tolist())

            input_ids, attention_mask = postprocess_data(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_prompt_length,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation=truncation,
            )
            batch_input_ids.append(input_ids[0])
            batch_attention_mask.append(attention_mask[0])

            for key, val in data.items():
                if key in ("raw_prompt", "dummy_tensor"):
                    continue
                if isinstance(val, torch.Tensor):
                    tensors[key].append(val)
                else:
                    non_tensors[key].append(val)

        input_ids = torch.stack(batch_input_ids, dim=0)
        attention_mask = torch.stack(batch_attention_mask, dim=0)
        position_ids = compute_position_id_with_mask(attention_mask)

        for key, val in tensors.items():
            tensors[key] = torch.stack(val, dim=0)

        tensors["input_ids"] = input_ids
        tensors["attention_mask"] = attention_mask
        tensors["position_ids"] = position_ids

        non_tensors["raw_prompt_ids"] = raw_prompt_ids_list
        non_tensors["raw_prompt"] = [d["raw_prompt"] for d in data_list]

        for key, val in non_tensors.items():
            non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

        return {**tensors, **non_tensors}

    return _collate


def tokenize_offpolicy_pairs(batch: "DataProto", tokenizer, max_prompt_length: int, max_response_length: int):
    """Build rollout-compatible DataProtos for off-policy chosen and rejected pairs.

    Takes a batch produced by the standard collate_fn (prompts already tokenized)
    and the chosen_response / rejected_response dicts carried in non_tensor_batch.
    Returns a single DataProto with 2*N rows (chosen first, rejected second) that
    has the same tensor layout as on-policy rollout output after union:
        input_ids      [2N, prompt_len + resp_len]
        attention_mask  [2N, prompt_len + resp_len]
        position_ids   [2N, prompt_len + resp_len]
        responses      [2N, resp_len]
        response_mask  [2N, resp_len]
        prompts        [2N, prompt_len]
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_input_ids = batch.batch["input_ids"]          # [N, prompt_len]  (left-padded)
    prompt_attention_mask = batch.batch["attention_mask"] # [N, prompt_len]
    n = prompt_input_ids.shape[0]
    prompt_len = prompt_input_ids.shape[1]

    chosen_responses = batch.non_tensor_batch["chosen_response"]   # array of dicts
    rejected_responses = batch.non_tensor_batch["rejected_response"]

    all_input_ids = []
    all_attention_mask = []
    all_responses = []
    all_response_mask = []
    all_prompts = []

    for side_responses in [chosen_responses, rejected_responses]:
        for i in range(n):
            p_ids = prompt_input_ids[i]       # [prompt_len]
            p_mask = prompt_attention_mask[i]  # [prompt_len]

            resp_msg = side_responses[i]
            raw_prompt_msgs = batch.non_tensor_batch["raw_prompt"][i]

            full_msgs = list(raw_prompt_msgs) + [resp_msg]
            full_text = tokenizer.apply_chat_template(full_msgs, add_generation_prompt=False, tokenize=False)
            prompt_text = tokenizer.apply_chat_template(raw_prompt_msgs, add_generation_prompt=True, tokenize=False)
            full_tok = tokenizer(full_text, add_special_tokens=False)["input_ids"]
            prompt_tok = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            resp_ids = full_tok[len(prompt_tok):]

            if len(resp_ids) > max_response_length:
                resp_ids = resp_ids[:max_response_length]
            resp_len_actual = len(resp_ids)

            resp_tensor = torch.full((max_response_length,), pad_id, dtype=torch.long)
            resp_tensor[:resp_len_actual] = torch.tensor(resp_ids, dtype=torch.long)
            resp_mask = torch.zeros(max_response_length, dtype=torch.long)
            resp_mask[:resp_len_actual] = 1

            seq_ids = torch.cat([p_ids, resp_tensor], dim=0)
            seq_mask = torch.cat([p_mask, resp_mask], dim=0)

            all_input_ids.append(seq_ids)
            all_attention_mask.append(seq_mask)
            all_responses.append(resp_tensor)
            all_response_mask.append(resp_mask)
            all_prompts.append(p_ids)

    input_ids = torch.stack(all_input_ids, dim=0)
    attention_mask = torch.stack(all_attention_mask, dim=0)
    position_ids = compute_position_id_with_mask(attention_mask)
    responses = torch.stack(all_responses, dim=0)
    response_mask = torch.stack(all_response_mask, dim=0)
    prompts = torch.stack(all_prompts, dim=0)

    from tensordict import TensorDict
    td = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_mask": response_mask,
        "prompts": prompts,
    }, batch_size=2 * n)

    return DataProto(batch=td)


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different
            # WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this "
                    f"ray cluster"
                )


def _compute_response_info(batch: DataProto) -> dict[str, Any]:
    """Computes actual (unpadded) prompt and response lengths from masks."""
    batch_size = batch.batch.batch_size[0]
    device = batch.batch.device

    response_lengths_tensor = batch.batch["response_mask"].sum(dim=1).float()
    total_lengths = batch.batch["attention_mask"].sum(dim=1).float()
    prompt_lengths_tensor = total_lengths - response_lengths_tensor

    max_response_length = batch.batch["responses"].shape[1]
    max_prompt_length = batch.batch["prompts"].shape[1]

    return {
        "prompt_length": prompt_lengths_tensor,
        "response_length": response_lengths_tensor,
        "max_response_length": max_response_length,
        "max_prompt_length": max_prompt_length,
    }


# --- Modified Metric Function ---
def compute_dpo_data_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Computes metrics from the generation batch: reward scores and sequence lengths.
    DPO-specific metrics (loss, logprobs, accuracies) come from actor_output and
    are merged separately via reduce_metrics at the call site.
    """
    metrics = {}
    try:
        # --- Rewards (from reward_fn, stored as token_level_rewards) ---
        if "token_level_rewards" in batch.batch and batch.batch["token_level_rewards"] is not None:
            response_mask = batch.batch.get("response_mask")
            token_rewards = batch.batch["token_level_rewards"]
            if response_mask is not None:
                sequence_reward = (token_rewards * response_mask).sum(-1)
            else:
                sequence_reward = token_rewards.sum(-1)
            metrics.update(
                {
                    "reward/rewards/mean": torch.mean(sequence_reward).item(),
                    "reward/rewards/max": torch.max(sequence_reward).item(),
                    "reward/rewards/min": torch.min(sequence_reward).item(),
                }
            )

        # --- Preference pair count (inferred from batch size / n_rollouts) ---

        # --- Ref log prob stats (token-level, before splitting into chosen/rejected) ---
        if "ref_log_prob" in batch.batch and batch.batch["ref_log_prob"] is not None:
            response_mask = batch.batch.get("response_mask")
            ref_lp = batch.batch["ref_log_prob"]
            if response_mask is not None:
                ref_seq_logps = (ref_lp * response_mask).sum(-1)
            else:
                ref_seq_logps = ref_lp.sum(-1)
            metrics["ref/seq_logprob/mean"] = torch.mean(ref_seq_logps).item()

        # --- Policy log prob stats (old_log_probs from the current policy snapshot) ---
        if "old_log_probs" in batch.batch and batch.batch["old_log_probs"] is not None:
            response_mask = batch.batch.get("response_mask")
            old_lp = batch.batch["old_log_probs"]
            if response_mask is not None:
                policy_seq_logps = (old_lp * response_mask).sum(-1)
            else:
                policy_seq_logps = old_lp.sum(-1)
            metrics["policy/seq_logprob/mean"] = torch.mean(policy_seq_logps).item()

        # --- KL divergence between policy and ref (if both available) ---
        if (
            "old_log_probs" in batch.batch
            and "ref_log_prob" in batch.batch
            and batch.batch["old_log_probs"] is not None
            and batch.batch["ref_log_prob"] is not None
        ):
            response_mask = batch.batch.get("response_mask")
            kl = batch.batch["old_log_probs"] - batch.batch["ref_log_prob"]
            if response_mask is not None:
                kl_masked = kl * response_mask
                kl_per_seq = kl_masked.sum(-1) / response_mask.sum(-1).clamp(min=1)
            else:
                kl_per_seq = kl.mean(-1)
            metrics["kl/policy_vs_ref/mean"] = torch.mean(kl_per_seq).item()
            metrics["kl/policy_vs_ref/max"] = torch.max(kl_per_seq).item()

        # --- Length Metrics ---
        response_info = _compute_response_info(batch)
        prompt_length = response_info["prompt_length"]
        response_length = response_info["response_length"]
        max_response_length = response_info["max_response_length"]
        max_prompt_length = response_info["max_prompt_length"]

        metrics.update(
            {
                "response_length/mean": torch.mean(response_length).item(),
                "response_length/max": torch.max(response_length).item(),
                "response_length/min": torch.min(response_length).item(),
                "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).item(),
                "prompt_length/mean": torch.mean(prompt_length).item(),
                "prompt_length/max": torch.max(prompt_length).item(),
                "prompt_length/min": torch.min(prompt_length).item(),
                "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).item(),
            }
        )

    except KeyError as e:
        print(f"ERROR in compute_dpo_data_metrics: Missing key {e}")
    except Exception as e:
        print(f"ERROR in compute_dpo_data_metrics: {e}")
        traceback.print_exc()

    return metrics


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_onlineDPO_pref(data: DataProto, n_rollouts: int = 2):
    """
    Compute DPO chosen/rejected indices from token-level rewards.
    Returns (chosen_idx, rejected_idx) tensors of shape [num_prompts].
    """
    rewards_tensor = data.batch.get("token_level_rewards")
    mask_tensor = data.batch.get("response_mask")

    if rewards_tensor is None or mask_tensor is None:
        print("  ERROR: Missing 'token_level_rewards' or 'response_mask' in input data!")
        return None, None

    try:
        return core_algos.compute_onlinedpo_pref(
            token_level_rewards=rewards_tensor, response_mask=mask_tensor, n_rollouts=n_rollouts,
        )
    except Exception as e_pref:
        print(f"ERROR during core_algos.compute_onlinedpo_pref: {e_pref}")
        traceback.print_exc()
        return None, None


@contextmanager
def _timer(name: str, timing_raw: dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RaySPINTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        # assert get_torch_device().is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = True #need_reference_policy(role_worker_mapping)
        self.use_rm = False #need_reward_model(role_worker_mapping)
        self.use_critic = False
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.async_rollout_mode = False
        self.device_name = device_name if device_name else self.config.trainer.device

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders, plus an optional off-policy dataloader.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            collate_fn = make_spin_collate_fn(
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation=self.config.data.get("truncation", "error"),
            )

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

        # --- Off-policy dataloader (optional) ---
        offpolicy_files = self.config.data.get("offpolicy_files", None)
        self.offpolicy_dataloader = None
        if offpolicy_files:
            offpolicy_data_config = OmegaConf.to_container(self.config.data, resolve=True)
            offpolicy_data_config["train_files"] = offpolicy_files if isinstance(offpolicy_files, list) else [offpolicy_files]
            offpolicy_data_config["seed"] = (offpolicy_data_config.get("seed") or 42) + 7919
            offpolicy_data_config = OmegaConf.create(offpolicy_data_config)

            offpolicy_dataset = create_rl_dataset(
                offpolicy_data_config.train_files,
                offpolicy_data_config,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("offpolicy_max_samples", -1),
            )
            offpolicy_sampler = create_rl_sampler(offpolicy_data_config, offpolicy_dataset)
            offpolicy_collate_fn = make_spin_collate_fn(
                tokenizer=self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length,
                truncation=self.config.data.get("truncation", "error"),
            )
            offpolicy_batch_size = self.config.data.get("offpolicy_batch_size", self.config.data.train_batch_size)
            self.offpolicy_dataloader = StatefulDataLoader(
                dataset=offpolicy_dataset,
                batch_size=offpolicy_batch_size,
                num_workers=self.config.data.get("dataloader_num_workers", 8),
                drop_last=True,
                collate_fn=offpolicy_collate_fn,
                sampler=offpolicy_sampler,
            )
            print(
                f"Off-policy dataloader created: {len(self.offpolicy_dataloader)} batches, "
                f"batch_size={offpolicy_batch_size}, dataset_size={len(offpolicy_dataset)}"
            )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_inputs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs"])
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            sample_gts = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        print(f"DEBUG: Data sources shape: {data_sources.shape}")  # Added Print
        print(f"DEBUG: reward_extra_infos_dict keys before processing: {reward_extra_infos_dict.keys()}")  # Added Print

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        print(
            f"DEBUG: Output of process_validation_metrics (data_src2var2metric2val): {data_src2var2metric2val}"
        )  # Added Print
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different
        # parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to
        # different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # Latest verl deprecated sync rollout. actor_rollout_wg.generate_sequences()
        # calls rollout.resume() which looks up a `sglang_server_*` ray actor that is
        # only created by AgentLoopManager. So always create it and route gen through it.
        from verl.experimental.agent_loop import AgentLoopManager

        actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
        )
        self.async_rollout_mode = True

        from verl.checkpoint_engine import CheckpointEngineManager
        from verl.utils.config import omega_conf_to_dataclass

        checkpoint_engine_config = omega_conf_to_dataclass(
            self.config.actor_rollout_ref.rollout.checkpoint_engine
        )
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and "
                "max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if self.offpolicy_dataloader is not None:
            offpolicy_local_path = os.path.join(local_global_step_folder, "offpolicy_data.pt")
            torch.save(self.offpolicy_dataloader.state_dict(), offpolicy_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    @staticmethod
    def _is_valid_checkpoint(ckpt_path):
        """Check that a checkpoint dir actually contains saved model files."""
        actor_path = os.path.join(ckpt_path, "actor")
        if not os.path.isdir(actor_path):
            return False
        actor_files = os.listdir(actor_path)
        return len(actor_files) > 0

    def _find_valid_checkpoint(self, checkpoint_folder):
        """Find the latest valid checkpoint, skipping corrupted/empty ones."""
        if not os.path.isdir(checkpoint_folder):
            return None
        step_dirs = sorted(
            [d for d in os.listdir(checkpoint_folder) if d.startswith("global_step_")],
            key=lambda d: int(d.split("global_step_")[-1]),
            reverse=True,
        )
        for step_dir in step_dirs:
            candidate = os.path.join(checkpoint_folder, step_dir)
            if self._is_valid_checkpoint(candidate):
                return candidate
            else:
                print(f"Skipping invalid/incomplete checkpoint: {candidate}")
        return None

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = self._find_valid_checkpoint(checkpoint_folder)

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
                if not self._is_valid_checkpoint(global_step_folder):
                    raise ValueError(f"Checkpoint at {global_step_folder} is invalid/incomplete (empty actor dir)")
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        if self.offpolicy_dataloader is not None:
            offpolicy_local_path = os.path.join(global_step_folder, "offpolicy_data.pt")
            if os.path.exists(offpolicy_local_path):
                offpolicy_state_dict = torch.load(offpolicy_local_path, weights_only=False)
                self.offpolicy_dataloader.load_state_dict(offpolicy_state_dict)
            else:
                print(f"Warning: No off-policy dataloader state found at {offpolicy_local_path}, will start from scratch")

        return self.global_steps

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit_dpo(self):  # Renamed for clarity as standard PPO loop
        """
        The training loop of Online DPO using a periodically updated reference model.
        The driver process calls worker groups for computation.
        Advantage computation is replaced by DPO logic.
        """
        import traceback  # Ensure traceback is imported

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        # Initialize logger
        logger = None
        try:
            logger = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True, throw_on_missing=False),
            )
        except Exception as e:
            print(f"Warning: Failed to initialize logger: {e}")

        self.global_steps = 0
        # Load checkpoint before doing anything
        loaded_step = self._load_checkpoint()
        self.global_steps = loaded_step + 1 if loaded_step is not None and loaded_step > 0 else 1
        print(
            f"Starting Online DPO training from global step {self.global_steps}. "
            f"Total steps: {self.total_training_steps}"
        )
        print(f"Reference model update frequency: {self.config.trainer.get('ref_update_freq', 'Not Set')}")

        # Wake up rollout replicas and sync initial weights
        self.checkpoint_manager.update_weights(self.global_steps)
        print("Initial weight sync to rollout replicas complete.")

        # Check if reference policy is configured correctly for this mode
        if not self.use_reference_policy:
            print(
                "WARNING: 'use_reference_policy' is False. Periodic reference model update requires a "
                "reference policy worker. DPO updates might fail or use incorrect logic."
            )
            # Consider raising an error if strict adherence is required:
            # raise ValueError("Periodic reference model update requires 'use_reference_policy' to be True "
            #                  "and a configured reference worker.")

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            print("Running validation before Online DPO training...")
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            if logger and val_metrics:
                logger.log(data=val_metrics, step=max(0, self.global_steps - 1))
            if self.config.trainer.get("val_only", False):
                print("Validation only mode enabled. Exiting training.")
                if logger and hasattr(logger, "finish"):
                    logger.finish()
                return

        # Add tqdm progress bar
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Online DPO Training Progress",
            position=0,
            leave=True,
        )

        last_val_metrics = None
        should_stop = False

        offpolicy_iterator = None

        for epoch in range(self.config.trainer.total_epochs):
            if should_stop:
                break
            print(f"--- Starting Online DPO Epoch {epoch} ---")
            try:
                train_iterator = iter(self.train_dataloader)
            except TypeError:
                print("Warning: Dataloader is not iterable.")
                train_iterator = self.train_dataloader  # Fallback attempt

            if self.offpolicy_dataloader is not None:
                offpolicy_iterator = iter(self.offpolicy_dataloader)

            for batch_idx, batch_dict in enumerate(train_iterator):
                if self.global_steps > self.total_training_steps:
                    should_stop = True
                    break

                metrics = {}
                timing_raw = {}
                step_timer = Timer(logger=None)
                ref_log_prob_computed = False  # Flag to track if ref log probs were computed

                try:  # Outer try-except for the whole step
                    step_timer.start()
                    with _timer("step", timing_raw):
                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                        current_batch_size = batch.batch.batch_size[0]
                        print(
                            f"\n[Step {self.global_steps}, Batch {batch_idx}] Processing batch size: "
                            f"{current_batch_size}"
                        )

                        # --- Reference Model Update ---
                        ref_update_freq = self.config.trainer.get("ref_update_freq", -1)
                        if (
                            self.use_reference_policy
                            and ref_update_freq > 0
                            and self.global_steps % ref_update_freq == 0
                        ):
                            print(f"\n[Step {self.global_steps}] Updating Reference Model Weights from Actor...")
                            try:
                                _scratch = os.environ.get("SCRATCH", "/tmp")
                                _job_id = os.environ.get("SLURM_JOB_ID", "nojob")
                                actor_state_path = os.path.join(_scratch, "online_dpo_ckpts", f"actor_state_for_ref_{_job_id}")
                                self.actor_rollout_wg.save_checkpoint(actor_state_path)
                                self.ref_policy_wg.load_ref_model_weights(actor_state_path)
                                print(f"[Step {self.global_steps}] Reference Model Weights Updated.")
                            except Exception as sync_e:
                                print(f"ERROR during reference model sync at step {self.global_steps}: {sync_e}")
                                traceback.print_exc()

                        # Pop keys for generation
                        pop_batch_keys = ["input_ids", "attention_mask"]
                        if "position_ids" in batch.batch:
                            pop_batch_keys.append("position_ids")
                        pop_non_tensor_keys = ["raw_prompt_ids"] if "raw_prompt_ids" in batch.non_tensor_batch else []
                        if "raw_prompt" in batch.non_tensor_batch:
                            pop_non_tensor_keys.append("raw_prompt")
                        if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                            pop_non_tensor_keys.extend(["multi_modal_data", "multi_modal_inputs"])
                        original_non_tensor_data = batch.non_tensor_batch
                        gen_batch = batch.pop(
                            batch_keys=pop_batch_keys,
                            non_tensor_batch_keys=pop_non_tensor_keys,
                        )
                        gen_batch = gen_batch.repeat(
                            repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                        )
                        print(f"  [DBG step={self.global_steps}] gen_batch after repeat: "
                              f"input_ids={gen_batch.batch['input_ids'].shape}", flush=True)

                        # Generate sequences (chosen/rejected pairs)
                        print(f"  [DBG step={self.global_steps}] >>> ENTERING generation", flush=True)
                        with _timer("gen", timing_raw):
                            try:
                                if self.async_rollout_mode:
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                else:
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                # (Add Debug prints for gen_batch_output if needed)
                            except Exception as gen_e:
                                print(f"\n!!!!!!!! ERROR DURING GENERATION (Step {self.global_steps}) !!!!!!!!")
                                print(gen_e)
                                traceback.print_exc()
                                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                step_timer.stop()
                                continue

                        print(f"  [DBG step={self.global_steps}] <<< GENERATION done "
                              f"({timing_raw.get('gen', '?'):.1f}s)", flush=True)

                        self.checkpoint_manager.sleep_replicas()

                        # Combine original prompts with generated sequences
                        batch.non_tensor_batch = original_non_tensor_data  # Restore non-tensor data
                        batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(current_batch_size)], dtype=object
                        )
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                        # Compute response mask (needed for reward calc and DPO prep)
                        batch.batch["response_mask"] = compute_response_mask(batch)

                        # --- Filter truncated responses before annotation ---
                        n_rollouts = self.config.actor_rollout_ref.rollout.n
                        max_resp_len = batch.batch["responses"].shape[1]
                        resp_lengths = batch.batch["response_mask"].sum(dim=1)
                        is_truncated = (resp_lengths >= max_resp_len).cpu().numpy()
                        n_total = len(is_truncated)
                        n_prompts_in_batch = n_total // n_rollouts
                        skip_annotation = np.zeros(n_total, dtype=bool)
                        truncated_prompt_count = 0
                        for _pi in range(n_prompts_in_batch):
                            _start = _pi * n_rollouts
                            _end = _start + n_rollouts
                            _group_trunc = is_truncated[_start:_end]
                            n_non_trunc = int((~_group_trunc).sum())
                            if n_non_trunc >= 2:
                                skip_annotation[_start:_end] = _group_trunc
                            else:
                                truncated_prompt_count += 1
                        if truncated_prompt_count > 0:
                            _PAIRS_LOG_FILE.write(
                                f"[Step {self.global_steps}] WARNING: {truncated_prompt_count}/{n_prompts_in_batch} "
                                f"prompts have <2 non-truncated responses, annotating all for those.\n"
                            )
                            _PAIRS_LOG_FILE.flush()
                        n_skipped = int(skip_annotation.sum())
                        print(f"  [Step {self.global_steps}] Truncation filter: {n_skipped}/{n_total} responses skipped, "
                              f"{truncated_prompt_count} prompts with all truncated", flush=True)
                        batch.non_tensor_batch["skip_reward_annotation"] = skip_annotation

                        # --- Compute Rewards/Scores (needed to pick best/worst) ---
                        print(f"  [DBG step={self.global_steps}] >>> ENTERING reward_calc", flush=True)
                        with _timer("reward_calc", timing_raw):
                            if self.use_rm:
                                reward_tensor_rm = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor_rm)

                            reward_extra_infos_dict = {}
                            try:
                                if self.reward_fn is None:
                                    reward_tensor = batch.batch.get(
                                        "rm_scores", torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                                    )
                                else:
                                    reward_result = self.reward_fn(batch, return_dict=True)
                                    reward_tensor = reward_result["reward_tensor"]
                                    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})

                            except Exception:
                                traceback.print_exc()
                                reward_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32)
                                reward_extra_infos_dict = {}

                            batch.batch["token_level_rewards"] = reward_tensor
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )

                        print(f"  [DBG step={self.global_steps}] <<< reward_calc done "
                              f"({timing_raw.get('reward_calc', '?'):.1f}s)", flush=True)

                        # --- Select chosen/rejected indices (excluding truncated) ---
                        n_rollouts = self.config.actor_rollout_ref.rollout.n
                        print(f"  [DBG step={self.global_steps}] >>> ENTERING preference selection "
                              f"(n_rollouts={n_rollouts})", flush=True)

                        _resp_mask_sel = batch.batch["response_mask"]
                        _tok_rew_sel = batch.batch["token_level_rewards"]
                        _scores_sel = (_tok_rew_sel * _resp_mask_sel).sum(dim=-1)
                        _score_groups = _scores_sel.view(-1, n_rollouts)
                        _trunc_groups = torch.tensor(
                            is_truncated.reshape(-1, n_rollouts), device=_scores_sel.device, dtype=torch.bool
                        )
                        _n_prompts_sel = _score_groups.shape[0]

                        chosen_idx_list = []
                        rejected_idx_list = []
                        fallback_prompts = 0
                        for _pi in range(_n_prompts_sel):
                            _valid_mask = ~_trunc_groups[_pi]
                            _valid_indices = _valid_mask.nonzero(as_tuple=True)[0]
                            if len(_valid_indices) >= 2:
                                _valid_scores = _score_groups[_pi][_valid_indices]
                                _best_local = _valid_indices[_valid_scores.argmax()]
                                _worst_local = _valid_indices[_valid_scores.argmin()]
                            else:
                                fallback_prompts += 1
                                _all_indices = torch.arange(n_rollouts, device=_score_groups.device)
                                _all_scores = _score_groups[_pi]
                                _best_local = _all_indices[_all_scores.argmax()]
                                _worst_local = _all_indices[_all_scores.argmin()]
                            _offset = _pi * n_rollouts
                            chosen_idx_list.append(_offset + _best_local.item())
                            rejected_idx_list.append(_offset + _worst_local.item())

                        if fallback_prompts > 0:
                            _PAIRS_LOG_FILE.write(
                                f"[Step {self.global_steps}] WARNING: {fallback_prompts}/{_n_prompts_sel} prompts "
                                f"had <2 non-truncated responses, used all responses for those.\n"
                            )
                            _PAIRS_LOG_FILE.flush()
                            print(f"  [Step {self.global_steps}] {fallback_prompts}/{_n_prompts_sel} prompts "
                                  f"fell back to all responses (< 2 non-truncated)", flush=True)

                        chosen_idx = torch.tensor(chosen_idx_list, device=_scores_sel.device)
                        rejected_idx = torch.tensor(rejected_idx_list, device=_scores_sel.device)
                        print(f"  [DBG step={self.global_steps}] <<< preference selection done "
                              f"(chosen={chosen_idx.shape})", flush=True)

                        # Log chosen/rejected reward stats
                        _resp_mask_all = batch.batch["response_mask"]
                        _tok_rewards_all = batch.batch["token_level_rewards"]
                        _seq_rewards_all = (_tok_rewards_all * _resp_mask_all).sum(-1)
                        chosen_rewards = _seq_rewards_all[chosen_idx]
                        rejected_rewards = _seq_rewards_all[rejected_idx]
                        reward_margin = chosen_rewards - rejected_rewards
                        metrics["reward/chosen/mean"] = chosen_rewards.mean().item()
                        metrics["reward/rejected/mean"] = rejected_rewards.mean().item()
                        metrics["reward/margin/mean"] = reward_margin.mean().item()
                        metrics["reward/margin/min"] = reward_margin.min().item()
                        metrics["reward/margin/max"] = reward_margin.max().item()
                        print(f"  [Step {self.global_steps}] reward chosen={chosen_rewards.mean().item():.3f} "
                              f"rejected={rejected_rewards.mean().item():.3f} "
                              f"margin={reward_margin.mean().item():.3f}")

                        # Log chosen/rejected response pairs to file
                        _n_log = min(5, len(chosen_idx))
                        for _i in range(_n_log):
                            _c_ids = batch.batch["responses"][chosen_idx[_i]]
                            _r_ids = batch.batch["responses"][rejected_idx[_i]]
                            _p_ids = batch.batch["prompts"][chosen_idx[_i]]
                            _p_mask = batch.batch["attention_mask"][chosen_idx[_i]]
                            _p_len = _p_mask[:_p_ids.shape[0]].sum().item()
                            _prompt_str = self.tokenizer.decode(_p_ids[-int(_p_len):], skip_special_tokens=False)
                            _c_str = self.tokenizer.decode(_c_ids, skip_special_tokens=False)
                            _r_str = self.tokenizer.decode(_r_ids, skip_special_tokens=False)
                            _PAIRS_LOG_FILE.write(
                                f"\n{'='*80}\n"
                                f"[Step {self.global_steps} | Pair {_i}]\n"
                                f"Chosen reward: {chosen_rewards[_i].item():.3f} | "
                                f"Rejected reward: {rejected_rewards[_i].item():.3f} | "
                                f"Margin: {reward_margin[_i].item():.3f}\n"
                                f"\n--- PROMPT ---\n{_prompt_str}\n"
                                f"\n--- CHOSEN ---\n{_c_str}\n"
                                f"\n--- REJECTED ---\n{_r_str}\n"
                                f"{'='*80}\n\n"
                            )
                            _PAIRS_LOG_FILE.flush()

                        # --- Build subset batch with only chosen+rejected (2 per prompt) ---
                        subset_idx = torch.cat([chosen_idx, rejected_idx], dim=0)
                        subset_batch = batch[subset_idx]
                        num_prompts = chosen_idx.shape[0]
                        print(f"  [DBG step={self.global_steps}] Subset batch: {subset_batch.batch['input_ids'].shape[0]} "
                              f"rows (from {batch.batch['input_ids'].shape[0]} total rollouts)", flush=True)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(subset_batch, metrics=metrics)

                        subset_batch.meta_info["global_token_num"] = torch.sum(
                            subset_batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # --- Compute policy log probs on subset only ---
                        print(f"  [DBG step={self.global_steps}] >>> ENTERING policy_log_prob "
                              f"(subset size={subset_batch.batch['input_ids'].shape[0]})", flush=True)
                        with _timer("policy_log_prob", timing_raw):
                            policy_log_prob_output = self.actor_rollout_wg.compute_log_prob(subset_batch)
                            subset_batch = subset_batch.union(policy_log_prob_output)

                        print(f"  [DBG step={self.global_steps}] <<< policy_log_prob done "
                              f"({timing_raw.get('policy_log_prob', '?'):.1f}s)", flush=True)

                        # --- Compute ref log probs on subset only ---
                        if self.use_reference_policy:
                            print(f"  [DBG step={self.global_steps}] >>> ENTERING ref_log_prob "
                                  f"(subset size={subset_batch.batch['input_ids'].shape[0]})", flush=True)
                            with _timer("ref_log_prob_dpo", timing_raw):
                                try:
                                    ref_log_prob_output = self.ref_policy_wg.compute_ref_log_prob(subset_batch)
                                    subset_batch = subset_batch.union(ref_log_prob_output)
                                    ref_log_prob_computed = True
                                except Exception as ref_e:
                                    print(f"ERROR computing reference log probs at step {self.global_steps}: {ref_e}")
                                    traceback.print_exc()
                                    subset_batch.batch["ref_log_prob"] = None
                                    ref_log_prob_computed = False
                            print(f"  [DBG step={self.global_steps}] <<< ref_log_prob done "
                                  f"({timing_raw.get('ref_log_prob_dpo', '?'):.1f}s)", flush=True)
                        else:
                            print(
                                "Warning: Skipping external reference log prob calculation as use_reference_policy "
                                "is False."
                            )

                        # --- Compute KL divergence between policy and ref on subset ---
                        if (
                            ref_log_prob_computed
                            and "old_log_probs" in subset_batch.batch
                            and "ref_log_prob" in subset_batch.batch
                            and subset_batch.batch["old_log_probs"] is not None
                            and subset_batch.batch["ref_log_prob"] is not None
                        ):
                            _resp_mask = subset_batch.batch["response_mask"]
                            _kl = subset_batch.batch["old_log_probs"] - subset_batch.batch["ref_log_prob"]
                            _kl_masked = _kl * _resp_mask
                            _kl_raw_per_seq = _kl_masked.sum(-1)
                            _kl_per_seq = _kl_raw_per_seq / _resp_mask.sum(-1).clamp(min=1)
                            # raw (sequence-level, not normalized by length)
                            metrics["kl/policy_vs_ref_raw/mean"] = torch.mean(_kl_raw_per_seq).item()
                            metrics["kl/policy_vs_ref_raw/chosen"] = torch.mean(_kl_raw_per_seq[:num_prompts]).item()
                            metrics["kl/policy_vs_ref_raw/rejected"] = torch.mean(_kl_raw_per_seq[num_prompts:]).item()
                            # per-token (normalized by response length)
                            metrics["kl/policy_vs_ref/mean"] = torch.mean(_kl_per_seq).item()
                            metrics["kl/policy_vs_ref/max"] = torch.max(_kl_per_seq).item()
                            metrics["kl/policy_vs_ref/min"] = torch.min(_kl_per_seq).item()
                            metrics["kl/chosen/mean"] = torch.mean(_kl_per_seq[:num_prompts]).item()
                            metrics["kl/rejected/mean"] = torch.mean(_kl_per_seq[num_prompts:]).item()

                        # --- Prepare DPO Batch ---
                        # subset_batch is ordered as [chosen_0..chosen_N, rejected_0..rejected_N]
                        dpo_update_batch_proto = None
                        with _timer("prepare_dpo_batch", timing_raw):
                            try:
                                if self.use_reference_policy and not ref_log_prob_computed:
                                    raise ValueError("Reference log probs required but failed to compute.")

                                required_keys = ["input_ids", "attention_mask", "response_mask"]
                                for rk in required_keys:
                                    if rk not in subset_batch.batch or subset_batch.batch[rk] is None:
                                        raise KeyError(f"Required key '{rk}' missing from subset_batch for DPO prep.")

                                chosen_input_ids = subset_batch.batch["input_ids"][:num_prompts]
                                chosen_attention_mask = subset_batch.batch["attention_mask"][:num_prompts]
                                rejected_input_ids = subset_batch.batch["input_ids"][num_prompts:]
                                rejected_attention_mask = subset_batch.batch["attention_mask"][num_prompts:]
                                chosen_position_ids = (
                                    subset_batch.batch["position_ids"][:num_prompts]
                                    if "position_ids" in subset_batch.batch
                                    else None
                                )
                                rejected_position_ids = (
                                    subset_batch.batch["position_ids"][num_prompts:]
                                    if "position_ids" in subset_batch.batch
                                    else None
                                )

                                chosen_response_mask = subset_batch.batch["response_mask"][:num_prompts]
                                rejected_response_mask = subset_batch.batch["response_mask"][num_prompts:]
                                seq_len = chosen_input_ids.shape[1]
                                resp_len = chosen_response_mask.shape[1]
                                prompt_len = seq_len - resp_len
                                prompt_pad = torch.zeros(
                                    chosen_response_mask.shape[0], prompt_len,
                                    dtype=chosen_response_mask.dtype, device=chosen_response_mask.device
                                )
                                chosen_full_mask = torch.cat([prompt_pad, chosen_response_mask], dim=1)
                                rejected_full_mask = torch.cat([
                                    torch.zeros(rejected_response_mask.shape[0], prompt_len,
                                                dtype=rejected_response_mask.dtype, device=rejected_response_mask.device),
                                    rejected_response_mask
                                ], dim=1)
                                chosen_labels = chosen_input_ids.clone()
                                chosen_labels[chosen_full_mask == 0] = -100
                                rejected_labels = rejected_input_ids.clone()
                                rejected_labels[rejected_full_mask == 0] = -100

                                if self.use_reference_policy:
                                    ref_log_prob_tensor = subset_batch.batch["ref_log_prob"]
                                    response_mask_sub = subset_batch.batch["response_mask"]
                                    length_normalize = self.config.algorithm.get("length_normalize", False)
                                    ref_sequence_logps = (ref_log_prob_tensor * response_mask_sub).sum(dim=-1)
                                    if length_normalize:
                                        response_lengths = response_mask_sub.sum(dim=-1).clamp(min=1)
                                        ref_sequence_logps = ref_sequence_logps / response_lengths
                                    reference_chosen_logps = ref_sequence_logps[:num_prompts]
                                    reference_rejected_logps = ref_sequence_logps[num_prompts:]
                                else:
                                    reference_chosen_logps = None
                                    reference_rejected_logps = None

                                dpo_tensors = {
                                    "chosen_input_ids": chosen_input_ids,
                                    "chosen_attention_mask": chosen_attention_mask,
                                    "chosen_labels": chosen_labels,
                                    "rejected_input_ids": rejected_input_ids,
                                    "rejected_attention_mask": rejected_attention_mask,
                                    "rejected_labels": rejected_labels,
                                }
                                if reference_chosen_logps is not None:
                                    dpo_tensors["reference_chosen_logps"] = reference_chosen_logps
                                if reference_rejected_logps is not None:
                                    dpo_tensors["reference_rejected_logps"] = reference_rejected_logps
                                if chosen_position_ids is not None:
                                    dpo_tensors["chosen_position_ids"] = chosen_position_ids
                                if rejected_position_ids is not None:
                                    dpo_tensors["rejected_position_ids"] = rejected_position_ids

                                dpo_global_token_num = (
                                    torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
                                    .sum(dim=-1)
                                    .tolist()
                                )
                                dpo_meta = {
                                    "dpo_beta": OmegaConf.select(self.config.algorithm, "dpo_beta", default=0.1),
                                    "dpo_loss_type": OmegaConf.select(
                                        self.config.algorithm, "dpo_loss_type", default="sigmoid"
                                    ),
                                    "dpo_label_smoothing": OmegaConf.select(
                                        self.config.algorithm, "dpo_label_smoothing", default=0.0
                                    ),
                                    "length_normalize": OmegaConf.select(
                                        self.config.algorithm, "length_normalize", default=False
                                    ),
                                    "use_reference_policy": self.use_reference_policy,
                                    "reference_free": not self.use_reference_policy,
                                    "global_step": self.global_steps,
                                    "global_token_num": dpo_global_token_num,
                                }

                                dpo_update_batch_proto = DataProto.from_dict(tensors=dpo_tensors, meta_info=dpo_meta)
                                print(f"  [DBG step={self.global_steps}] <<< prepare_dpo_batch done "
                                      f"(chosen={dpo_update_batch_proto.batch['chosen_input_ids'].shape})",
                                      flush=True)

                            except Exception as e_prep:
                                print(f"ERROR preparing DPO batch at step {self.global_steps}: {e_prep}")
                                traceback.print_exc()
                                dpo_update_batch_proto = None

                        # --- Off-policy DPO batch ---
                        offpolicy_dpo_proto = None
                        if offpolicy_iterator is not None:
                            with _timer("offpolicy", timing_raw):
                                try:
                                    try:
                                        offpolicy_batch_dict = next(offpolicy_iterator)
                                    except StopIteration:
                                        offpolicy_iterator = iter(self.offpolicy_dataloader)
                                        offpolicy_batch_dict = next(offpolicy_iterator)

                                    offpolicy_batch: DataProto = DataProto.from_single_dict(offpolicy_batch_dict)
                                    offpolicy_n = offpolicy_batch.batch.batch_size[0]
                                    print(f"  [Step {self.global_steps}] Off-policy batch: {offpolicy_n} prompts",
                                          flush=True)

                                    max_resp_len = self.config.data.max_response_length
                                    max_prmpt_len = self.config.data.max_prompt_length
                                    offpolicy_pairs = tokenize_offpolicy_pairs(
                                        offpolicy_batch, self.tokenizer, max_prmpt_len, max_resp_len,
                                    )

                                    offpolicy_pairs.meta_info["micro_batch_size"] = (
                                        self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", 1)
                                    )
                                    offpolicy_pairs.meta_info["temperature"] = (
                                        self.config.actor_rollout_ref.rollout.get("temperature", 1.0)
                                    )
                                    offpolicy_pairs.meta_info["use_dynamic_bsz"] = (
                                        self.config.actor_rollout_ref.actor.get("use_dynamic_bsz", False)
                                    )
                                    if offpolicy_pairs.meta_info["use_dynamic_bsz"]:
                                        offpolicy_pairs.meta_info["max_token_len"] = (
                                            self.config.actor_rollout_ref.actor.get("max_token_len", 0)
                                        )

                                    print(f"  [Step {self.global_steps}] Off-policy: computing policy log-probs "
                                          f"({offpolicy_pairs.batch['input_ids'].shape})", flush=True)
                                    offpolicy_logprob_out = self.actor_rollout_wg.compute_log_prob(offpolicy_pairs)
                                    offpolicy_pairs = offpolicy_pairs.union(offpolicy_logprob_out)

                                    if self.use_reference_policy:
                                        print(f"  [Step {self.global_steps}] Off-policy: computing ref log-probs",
                                              flush=True)
                                        offpolicy_ref_out = self.ref_policy_wg.compute_ref_log_prob(offpolicy_pairs)
                                        offpolicy_pairs = offpolicy_pairs.union(offpolicy_ref_out)

                                    op_resp_mask = offpolicy_pairs.batch["response_mask"]
                                    op_input_ids = offpolicy_pairs.batch["input_ids"]
                                    op_attn_mask = offpolicy_pairs.batch["attention_mask"]
                                    op_pos_ids = offpolicy_pairs.batch.get("position_ids", None)

                                    op_chosen_ids = op_input_ids[:offpolicy_n]
                                    op_chosen_attn = op_attn_mask[:offpolicy_n]
                                    op_rejected_ids = op_input_ids[offpolicy_n:]
                                    op_rejected_attn = op_attn_mask[offpolicy_n:]

                                    op_chosen_resp_mask = op_resp_mask[:offpolicy_n]
                                    op_rejected_resp_mask = op_resp_mask[offpolicy_n:]
                                    op_seq_len = op_input_ids.shape[1]
                                    op_resp_len_dim = op_resp_mask.shape[1]
                                    op_prompt_len = op_seq_len - op_resp_len_dim
                                    op_prompt_pad_c = torch.zeros(
                                        offpolicy_n, op_prompt_len,
                                        dtype=op_chosen_resp_mask.dtype, device=op_chosen_resp_mask.device
                                    )
                                    op_chosen_full_mask = torch.cat([op_prompt_pad_c, op_chosen_resp_mask], dim=1)
                                    op_prompt_pad_r = torch.zeros(
                                        offpolicy_n, op_prompt_len,
                                        dtype=op_rejected_resp_mask.dtype, device=op_rejected_resp_mask.device
                                    )
                                    op_rejected_full_mask = torch.cat([op_prompt_pad_r, op_rejected_resp_mask], dim=1)

                                    op_chosen_labels = op_chosen_ids.clone()
                                    op_chosen_labels[op_chosen_full_mask == 0] = -100
                                    op_rejected_labels = op_rejected_ids.clone()
                                    op_rejected_labels[op_rejected_full_mask == 0] = -100

                                    length_normalize = self.config.algorithm.get("length_normalize", False)
                                    if self.use_reference_policy and "ref_log_prob" in offpolicy_pairs.batch:
                                        op_ref_lp = offpolicy_pairs.batch["ref_log_prob"]
                                        op_ref_seq_logps = (op_ref_lp * op_resp_mask).sum(dim=-1)
                                        if length_normalize:
                                            op_ref_seq_logps = op_ref_seq_logps / op_resp_mask.sum(dim=-1).clamp(min=1)
                                        op_ref_chosen_logps = op_ref_seq_logps[:offpolicy_n]
                                        op_ref_rejected_logps = op_ref_seq_logps[offpolicy_n:]
                                    else:
                                        op_ref_chosen_logps = None
                                        op_ref_rejected_logps = None

                                    op_dpo_tensors = {
                                        "chosen_input_ids": op_chosen_ids,
                                        "chosen_attention_mask": op_chosen_attn,
                                        "chosen_labels": op_chosen_labels,
                                        "rejected_input_ids": op_rejected_ids,
                                        "rejected_attention_mask": op_rejected_attn,
                                        "rejected_labels": op_rejected_labels,
                                    }
                                    if op_ref_chosen_logps is not None:
                                        op_dpo_tensors["reference_chosen_logps"] = op_ref_chosen_logps
                                    if op_ref_rejected_logps is not None:
                                        op_dpo_tensors["reference_rejected_logps"] = op_ref_rejected_logps
                                    if op_pos_ids is not None:
                                        op_dpo_tensors["chosen_position_ids"] = op_pos_ids[:offpolicy_n]
                                        op_dpo_tensors["rejected_position_ids"] = op_pos_ids[offpolicy_n:]

                                    offpolicy_dpo_proto = DataProto.from_dict(tensors=op_dpo_tensors)
                                    print(f"  [Step {self.global_steps}] Off-policy DPO batch ready: "
                                          f"{offpolicy_n} pairs", flush=True)

                                except Exception as op_e:
                                    print(f"ERROR in off-policy processing at step {self.global_steps}: {op_e}")
                                    traceback.print_exc()
                                    offpolicy_dpo_proto = None

                        # --- Merge on-policy and off-policy DPO batches ---
                        if dpo_update_batch_proto is not None and offpolicy_dpo_proto is not None:
                            merged_tensors = {}
                            for key in dpo_update_batch_proto.batch.keys():
                                if key in offpolicy_dpo_proto.batch:
                                    on_t = dpo_update_batch_proto.batch[key]
                                    off_t = offpolicy_dpo_proto.batch[key]
                                    if on_t.shape[1:] == off_t.shape[1:]:
                                        merged_tensors[key] = torch.cat([on_t, off_t], dim=0)
                                    else:
                                        max_len = max(on_t.shape[1], off_t.shape[1])
                                        pad_val = -100 if "labels" in key else 0
                                        on_padded = torch.nn.functional.pad(
                                            on_t, (0, max_len - on_t.shape[1]), value=pad_val
                                        )
                                        off_padded = torch.nn.functional.pad(
                                            off_t, (0, max_len - off_t.shape[1]), value=pad_val
                                        )
                                        merged_tensors[key] = torch.cat([on_padded, off_padded], dim=0)
                                else:
                                    merged_tensors[key] = dpo_update_batch_proto.batch[key]

                            dpo_meta = dpo_update_batch_proto.meta_info.copy()
                            if "global_token_num" in dpo_meta and "global_token_num" not in offpolicy_dpo_proto.meta_info:
                                off_tok_num = merged_tensors.get("chosen_attention_mask", merged_tensors.get("rejected_attention_mask"))
                                if off_tok_num is not None:
                                    pass
                            dpo_update_batch_proto = DataProto.from_dict(tensors=merged_tensors, meta_info=dpo_meta)

                            on_n = num_prompts
                            off_n = offpolicy_dpo_proto.batch["chosen_input_ids"].shape[0]
                            print(f"  [Step {self.global_steps}] Merged DPO batch: {on_n} on-policy + "
                                  f"{off_n} off-policy = {on_n + off_n} total pairs", flush=True)
                        elif dpo_update_batch_proto is None and offpolicy_dpo_proto is not None:
                            dpo_update_batch_proto = offpolicy_dpo_proto
                            dpo_meta = {
                                "dpo_beta": OmegaConf.select(self.config.algorithm, "dpo_beta", default=0.1),
                                "dpo_loss_type": OmegaConf.select(
                                    self.config.algorithm, "dpo_loss_type", default="sigmoid"
                                ),
                                "dpo_label_smoothing": OmegaConf.select(
                                    self.config.algorithm, "dpo_label_smoothing", default=0.0
                                ),
                                "length_normalize": OmegaConf.select(
                                    self.config.algorithm, "length_normalize", default=False
                                ),
                                "use_reference_policy": self.use_reference_policy,
                                "reference_free": not self.use_reference_policy,
                                "global_step": self.global_steps,
                            }
                            dpo_update_batch_proto.meta_info.update(dpo_meta)
                            print(f"  [Step {self.global_steps}] Using off-policy DPO batch only "
                                  f"(on-policy failed)", flush=True)

                        # --- Actor Update Step ---
                        actor_output = None
                        if self.config.trainer.critic_warmup <= self.global_steps and dpo_update_batch_proto:
                            print(f"  [DBG step={self.global_steps}] >>> ENTERING update_actor_dpo "
                                  f"(batch size={dpo_update_batch_proto.batch['chosen_input_ids'].shape})", flush=True)
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor_dpo(dpo_update_batch_proto)
                            print(f"  [DBG step={self.global_steps}] <<< update_actor_dpo done "
                                  f"({timing_raw.get('update_actor', '?'):.1f}s)", flush=True)
                            if actor_output is not None and "metrics" in actor_output.meta_info:
                                metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                            with _timer("update_weights", timing_raw):
                                self.checkpoint_manager.update_weights(self.global_steps)
                            print(f"  [DBG step={self.global_steps}] <<< update_weights done "
                                  f"({timing_raw.get('update_weights', '?'):.1f}s)", flush=True)
                        elif dpo_update_batch_proto is None:
                            print(
                                f"Skipping actor update at step {self.global_steps} due to DPO batch preparation error."
                            )

                        # --- Validation and Saving ---
                        test_freq = OmegaConf.select(self.config.trainer, "test_freq", default=-1)
                        is_last_step = self.global_steps >= self.total_training_steps
                        if (
                            self.val_reward_fn is not None
                            and test_freq > 0
                            and (is_last_step or self.global_steps % test_freq == 0)
                        ):
                            print(f"\nRunning DPO validation at step {self.global_steps}...")
                            val_timing_raw = {}
                            with _timer("testing", val_timing_raw):
                                val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                            if val_metrics:
                                metrics["time/validation_run"] = val_timing_raw.get("testing", 0)
                                metrics.update(val_metrics)
                            else:
                                print("Validation skipped or returned no metrics.")

                        save_freq = OmegaConf.select(self.config.trainer, "save_freq", default=-1)
                        if save_freq > 0 and (is_last_step or self.global_steps % save_freq == 0):
                            print(f"\nSaving DPO checkpoint at step {self.global_steps}...")
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()  # Saves actor (and potentially critic if used elsewhere)
                            metrics["time/save_checkpoint"] = timing_raw.get("save_checkpoint", 0)

                    # --- End main step timer context ---

                    # --- Metrics calculation AFTER the 'step' timer block ---
                    metrics.update(compute_dpo_data_metrics(batch=batch))  # Use DPO-specific metrics
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    if "step" in timing_raw:
                        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    else:
                        print(
                            f"Warning: 'step' key missing from timing_raw at step {self.global_steps}. "
                            f"Skipping throughput."
                        )

                    step_timer.stop()
                    metrics["time/step"] = step_timer.last

                    # Log metrics
                    log_freq = OmegaConf.select(self.config.trainer, "log_freq", default=1)
                    if logger and self.global_steps % log_freq == 0:
                        log_payload = metrics.copy()
                        # Add learning rate to log payload
                        if actor_output and "actor/lr" in metrics:
                            log_payload["actor/lr"] = metrics["actor/lr"]

                        print(f"[Step {self.global_steps} DPO] Logging Step Payload Keys: {list(log_payload.keys())}")
                        try:
                            logger.log(data=log_payload, step=self.global_steps)
                        except Exception as e:
                            print(f"Logging failed at step {self.global_steps}: {e}")

                    # Update progress bar
                    postfix_metrics = {
                        k: f"{v:.3f}" if isinstance(v, float) else v
                        for k, v in metrics.items()
                        if isinstance(v, int | float)
                    }
                    progress_bar.set_postfix(postfix_metrics)

                except Exception as step_e:
                    print(f"\n!!!!!!!! ERROR DURING DPO Step {self.global_steps} !!!!!!!!")
                    print(f"Caught Exception: {step_e}")
                    traceback.print_exc()
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    step_timer.stop()
                    should_stop = True
                    break

                if is_last_step or should_stop:
                    print(f"Stopping DPO training at step {self.global_steps}.")
                    break

                self.global_steps += 1
                progress_bar.update(1)

            # End of epoch handling
            if hasattr(self.train_dataloader, "reset"):
                try:
                    self.train_dataloader.reset()
                except Exception as e:
                    print(f"Warning: Failed to reset train dataloader state: {e}")
            if self.offpolicy_dataloader is not None and hasattr(self.offpolicy_dataloader, "reset"):
                try:
                    self.offpolicy_dataloader.reset()
                except Exception as e:
                    print(f"Warning: Failed to reset off-policy dataloader state: {e}")
            if should_stop:
                break

        # --- Final cleanup and logging ---
        progress_bar.close()
        final_step = max(0, self.global_steps - 1)
        print(f"Online DPO Training finished at step {final_step}.")
        # Save final checkpoint
        save_freq = OmegaConf.select(self.config.trainer, "save_freq", default=-1)
        if not self.config.trainer.get("val_only", False) and (save_freq <= 0 or final_step % save_freq != 0):
            print(f"Saving final DPO checkpoint at step {final_step}...")
            self._save_checkpoint()

        # Final validation run
        if self.val_reward_fn and last_val_metrics is None and not self.config.trainer.get("val_only", False):
            print("Running final validation...")
            last_val_metrics = self._validate()
            if last_val_metrics and logger:
                last_val_metrics["final_validation"] = True
                try:
                    logger.log(data=last_val_metrics, step=final_step)
                except Exception as e:
                    print(f"[Final Val Metrics Log Error]: {e}")

        pprint(f"Final validation metrics: {last_val_metrics}")
        if logger and hasattr(logger, "finish"):
            logger.finish()
        print("Online DPO Training Run Complete.")
