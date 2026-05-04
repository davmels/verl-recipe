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


import numpy as np
import torch


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_onlinedpo_pref(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    n_rollouts: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each prompt group of *n_rollouts* interleaved responses, select
    chosen = argmax(score) and rejected = argmin(score among the rest).

    Assumes inputs are interleaved:
        [R0_P0, R1_P0, ..., R(n-1)_P0, R0_P1, R1_P1, ..., R(n-1)_P1, ...]

    Args:
        token_level_rewards: [num_prompts * n_rollouts, seq_len]
        response_mask:       [num_prompts * n_rollouts, seq_len]
        n_rollouts: number of rollouts per prompt (N)

    Returns:
        chosen_indices:   [num_prompts] — global indices of chosen responses
        rejected_indices: [num_prompts] — global indices of rejected responses
    """
    total = token_level_rewards.shape[0]
    if total % n_rollouts != 0:
        raise ValueError(
            f"Batch size {total} is not divisible by n_rollouts={n_rollouts}"
        )
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError(
            f"Shape mismatch: rewards {token_level_rewards.shape} vs mask {response_mask.shape}"
        )

    scores = (token_level_rewards * response_mask).sum(dim=-1)  # [total]
    score_groups = scores.view(-1, n_rollouts)  # [num_prompts, n_rollouts]
    num_prompts = score_groups.shape[0]

    chosen_local = torch.argmax(score_groups, dim=1)  # [num_prompts]

    # mask out the chosen position before taking argmin for rejected
    masked_scores = score_groups.clone()
    masked_scores[torch.arange(num_prompts, device=scores.device), chosen_local] = float("inf")
    rejected_local = torch.argmin(masked_scores, dim=1)  # [num_prompts]

    offsets = torch.arange(num_prompts, device=scores.device) * n_rollouts
    chosen_indices = offsets + chosen_local
    rejected_indices = offsets + rejected_local

    return chosen_indices, rejected_indices


def compute_online_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    reference_free: bool = False,
) -> torch.Tensor:
    import torch.nn.functional as F

    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = torch.zeros_like(pi_logratios)

    logits = pi_logratios - ref_logratios

    if loss_type == "sigmoid":
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'sigmoid', 'ipo', or 'hinge'.")

    return losses.mean()


def get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    """
    Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (e.g., huggingface CausalLMOutputs `logits`).
                Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for computing the sequence log probabilities. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per sequence. Otherwise, return the sum.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given sequences.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have the same shape[:-1]")

    # Ensure labels are contiguous and on the same device as logits
    labels = labels.contiguous().to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Calculate per token log probability
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    per_token_logps = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_logps = per_token_logps.view(
        shift_logits.size(0), shift_logits.size(1)
    )  # Reshape back to (batch_size, seq_len-1)

    # Create a mask for the labels that are not -100
    loss_mask = shift_labels != -100

    # Apply the mask to the per token log probabilities
    masked_logps = per_token_logps * loss_mask

    # Calculate the sum or average log probability per sequence
    sequence_logps = masked_logps.sum(dim=-1)

    if average_log_prob:
        # Avoid division by zero for sequences with no valid tokens
        num_valid_tokens = loss_mask.sum(dim=-1)
        return sequence_logps / torch.clamp(num_valid_tokens, min=1)
    else:
        return sequence_logps
