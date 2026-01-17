# Copyright 2024 ZeroModel Authors
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
"""
Loss functions for LRU training.

This module provides loss functions that encourage:
1. Stability: Representations should converge over iterations
2. Sparsity: Activations should be sparse (Hoyer-Square)
3. Ponder cost: Minimize unnecessary computation (ACT-style)

Combined loss:
    L_total = L_pred + w_stability * L_stability + w_sparsity * L_sparsity + w_ponder * L_ponder
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from verl.models.mla.lru import LRUOutput


@dataclass
class LRULossOutput:
    """Container for LRU loss computation results."""
    total_loss: torch.Tensor
    stability_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    ponder_loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]


class LearnableLossWeights(nn.Module):
    """Learnable loss weights with log-softplus parameterization.

    Addresses academic review feedback about "magic number" weights (0.1, 0.01, 0.001)
    by making weights learnable while constraining them to reasonable ranges.

    The parameterization uses log-softplus to ensure:
    1. Weights are always positive
    2. Gradients flow smoothly
    3. Weights remain in sensible ranges [min_weight, max_weight]

    Usage:
        learnable_weights = LearnableLossWeights()
        weights = learnable_weights()  # Returns dict of current weights

        # In optimizer:
        # Use separate param group with lower LR (0.1x base_lr recommended)
        optimizer = Adam([
            {'params': model.parameters()},
            {'params': learnable_weights.parameters(), 'lr': base_lr * 0.1},
        ])
    """

    def __init__(
        self,
        init_stability: float = 0.1,
        init_sparsity: float = 0.01,
        init_ponder: float = 0.001,
        min_weight: float = 1e-6,
        max_weight: float = 1.0,
    ):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Initialize parameters in log-space for stability
        # softplus(x) = log(1 + exp(x)), inverse is: x = log(exp(y) - 1)
        def inverse_softplus(y):
            """Inverse of softplus for initialization."""
            return torch.log(torch.exp(torch.tensor(y)) - 1 + 1e-8)

        self.log_stability = nn.Parameter(inverse_softplus(init_stability))
        self.log_sparsity = nn.Parameter(inverse_softplus(init_sparsity))
        self.log_ponder = nn.Parameter(inverse_softplus(init_ponder))

    def _get_weight(self, log_weight: torch.Tensor) -> torch.Tensor:
        """Convert log parameter to clamped weight."""
        weight = F.softplus(log_weight)
        return weight.clamp(self.min_weight, self.max_weight)

    @property
    def stability_weight(self) -> torch.Tensor:
        return self._get_weight(self.log_stability)

    @property
    def sparsity_weight(self) -> torch.Tensor:
        return self._get_weight(self.log_sparsity)

    @property
    def ponder_weight(self) -> torch.Tensor:
        return self._get_weight(self.log_ponder)

    def forward(self) -> Dict[str, torch.Tensor]:
        """Return current weight values as a dictionary."""
        return {
            'stability': self.stability_weight,
            'sparsity': self.sparsity_weight,
            'ponder': self.ponder_weight,
        }

    def get_weight_dict(self) -> Dict[str, float]:
        """Return detached weight values for logging."""
        return {
            'stability': self.stability_weight.item(),
            'sparsity': self.sparsity_weight.item(),
            'ponder': self.ponder_weight.item(),
        }


class StabilityLoss(nn.Module):
    """Stability loss for encouraging representation convergence.

    Penalizes large changes between consecutive iterations, encouraging
    the LRU to converge to a stable representation.

    L_stability = sum_{t=1}^{N-1} w_t * ||h_t - h_{t+1}||^2

    where w_t = decay^(N-1-t) / sum(decay^s) gives more weight to
    later iterations (convergence should happen toward the end).
    """

    def __init__(
        self,
        decay: float = 0.9,
        normalize: bool = True,
    ):
        super().__init__()
        self.decay = decay
        self.normalize = normalize

    def forward(
        self,
        intermediate_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute stability loss.

        Args:
            intermediate_states: States from each iteration [T, B, S, D]
            mask: Optional attention mask [B, S]

        Returns:
            Scalar stability loss
        """
        if intermediate_states is None or intermediate_states.shape[0] < 2:
            return torch.tensor(0.0, device=intermediate_states.device if intermediate_states is not None else 'cpu')

        num_iters = intermediate_states.shape[0]

        # Compute differences between consecutive states
        diffs = intermediate_states[1:] - intermediate_states[:-1]  # [T-1, B, S, D]

        # Squared L2 norm of differences
        diff_norms = torch.sum(diffs ** 2, dim=-1)  # [T-1, B, S]

        # Compute weights: more weight on later iterations
        weights = torch.tensor(
            [self.decay ** (num_iters - 2 - t) for t in range(num_iters - 1)],
            device=intermediate_states.device,
            dtype=intermediate_states.dtype,
        )
        weights = weights / weights.sum()  # Normalize

        # Weighted sum across iterations
        weighted_norms = torch.einsum('t,tbs->bs', weights, diff_norms)  # [B, S]

        # Apply mask if provided
        if mask is not None:
            weighted_norms = weighted_norms * mask

        # Normalize by dimension if requested
        if self.normalize:
            weighted_norms = weighted_norms / intermediate_states.shape[-1]

        return weighted_norms.mean()


class SparsityLoss(nn.Module):
    """Hoyer-Square sparsity loss for encouraging sparse activations.

    L_sparsity = (||a||_1)^2 / (||a||_2^2 + eps)

    This ratio is minimized when activations are sparse (few large values)
    and maximized when activations are dense (many small values).

    The loss is normalized to [0, 1] where 0 is maximally sparse.
    """

    def __init__(
        self,
        eps: float = 1e-8,
        target_sparsity: float = 0.8,
    ):
        super().__init__()
        self.eps = eps
        self.target_sparsity = target_sparsity

    def forward(
        self,
        activations: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Hoyer-Square sparsity loss.

        Args:
            activations: Activation tensor [B, S, D] or [B, S, H, D]
            mask: Optional mask [B, S]

        Returns:
            Scalar sparsity loss
        """
        # Flatten to [B*S, D] or [B*S, H*D]
        flat = activations.view(activations.shape[0] * activations.shape[1], -1)

        # L1 and L2 norms
        l1_norm = torch.abs(flat).sum(dim=-1)  # [B*S]
        l2_norm_sq = (flat ** 2).sum(dim=-1)   # [B*S]

        # Hoyer-Square ratio
        dim = flat.shape[-1]
        hoyer = (l1_norm ** 2) / (l2_norm_sq + self.eps)

        # Normalize to [0, 1]: sqrt(dim) is the max value (all equal)
        # 1.0 is the min value (single non-zero element)
        normalized = (hoyer - 1.0) / (dim ** 0.5 - 1.0 + self.eps)

        # Loss: penalize deviation from target sparsity
        # When normalized is low, activations are sparse (good)
        loss = F.relu(normalized - (1.0 - self.target_sparsity))

        # Apply mask if provided
        if mask is not None:
            mask_flat = mask.view(-1)
            loss = loss * mask_flat

        return loss.mean()


class PonderLoss(nn.Module):
    """ACT-style pondering cost loss.

    Penalizes excessive computation by adding the expected number of
    iterations to the loss. This encourages the model to halt early
    when confident.

    L_ponder = mean(N + R)

    where N is the number of iterations and R is the remainder
    (fractional part of the final halt probability).
    """

    def __init__(
        self,
        normalize_by_max: bool = True,
        max_iterations: int = 8,
    ):
        super().__init__()
        self.normalize_by_max = normalize_by_max
        self.max_iterations = max_iterations

    def forward(
        self,
        lru_output: LRUOutput,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute pondering cost loss.

        Args:
            lru_output: Output from LRU containing iteration info
            mask: Optional mask [B, S]

        Returns:
            Scalar ponder loss
        """
        # N + R gives the expected number of iterations
        ponder_cost = lru_output.num_iterations + lru_output.remainders  # [B, S]

        # Normalize by max iterations if requested
        if self.normalize_by_max:
            ponder_cost = ponder_cost / self.max_iterations

        # Apply mask if provided
        if mask is not None:
            ponder_cost = ponder_cost * mask

        return ponder_cost.mean()


class LRULossModule(nn.Module):
    """Combined loss module for LRU training.

    Computes and combines all LRU-specific losses with configurable
    weights. Supports weight scheduling for curriculum learning.

    Supports two weight modes:
    1. Fixed weights (default): Use provided weight values directly
    2. Learnable weights: Weights are learned during training (recommended)

    When using learnable weights, the weight parameters should be added to
    a separate optimizer param group with lower learning rate (0.1x recommended).
    """

    def __init__(
        self,
        stability_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        ponder_weight: float = 0.001,
        stability_decay: float = 0.9,
        sparsity_target: float = 0.8,
        max_iterations: int = 8,
        warmup_steps: int = 0,
        weight_schedule: str = 'constant',  # 'constant', 'linear', 'cosine'
        use_learnable_weights: bool = False,
    ):
        super().__init__()

        self.use_learnable_weights = use_learnable_weights

        # Loss weights - either fixed or learnable
        if use_learnable_weights:
            self.learnable_weights = LearnableLossWeights(
                init_stability=stability_weight,
                init_sparsity=sparsity_weight,
                init_ponder=ponder_weight,
            )
            # These are just for reference/logging
            self._init_stability_weight = stability_weight
            self._init_sparsity_weight = sparsity_weight
            self._init_ponder_weight = ponder_weight
        else:
            self.learnable_weights = None
            self.stability_weight = stability_weight
            self.sparsity_weight = sparsity_weight
            self.ponder_weight = ponder_weight

        # Loss modules
        self.stability_loss = StabilityLoss(decay=stability_decay)
        self.sparsity_loss = SparsityLoss(target_sparsity=sparsity_target)
        self.ponder_loss = PonderLoss(max_iterations=max_iterations)

        # Scheduling (only applies to fixed weights)
        self.warmup_steps = warmup_steps
        self.weight_schedule = weight_schedule
        self.current_step = 0

    def get_weight_params(self):
        """Get parameters for learnable weights (for separate optimizer group).

        Usage:
            optimizer = Adam([
                {'params': model.parameters(), 'lr': base_lr},
                {'params': loss_module.get_weight_params(), 'lr': base_lr * 0.1},
            ])
        """
        if self.learnable_weights is not None:
            return self.learnable_weights.parameters()
        return []

    def _get_scheduled_weights(self) -> Dict[str, Union[float, torch.Tensor]]:
        """Get loss weights based on current training step."""
        # If using learnable weights, return them directly (no scheduling)
        if self.use_learnable_weights and self.learnable_weights is not None:
            return self.learnable_weights()

        # Fixed weights with scheduling
        if self.current_step < self.warmup_steps:
            # During warmup, reduce auxiliary loss weights
            warmup_factor = self.current_step / max(self.warmup_steps, 1)
        else:
            warmup_factor = 1.0

        if self.weight_schedule == 'constant':
            schedule_factor = 1.0
        elif self.weight_schedule == 'linear':
            # Linearly increase weights
            schedule_factor = warmup_factor
        elif self.weight_schedule == 'cosine':
            # Cosine schedule
            import math
            schedule_factor = 0.5 * (1 + math.cos(math.pi * (1 - warmup_factor)))
        else:
            schedule_factor = 1.0

        return {
            'stability': self.stability_weight * schedule_factor,
            'sparsity': self.sparsity_weight * schedule_factor,
            'ponder': self.ponder_weight * schedule_factor,
        }

    def forward(
        self,
        lru_output: LRUOutput,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> LRULossOutput:
        """
        Compute combined LRU losses.

        Args:
            lru_output: Output from LRU module
            attention_mask: Optional attention mask [B, S]

        Returns:
            LRULossOutput with individual and combined losses
        """
        weights = self._get_scheduled_weights()

        # Compute individual losses
        if lru_output.intermediate_states is not None:
            stability = self.stability_loss(
                lru_output.intermediate_states,
                mask=attention_mask,
            )
        else:
            stability = torch.tensor(0.0, device=lru_output.output.device)

        sparsity = self.sparsity_loss(
            lru_output.output,
            mask=attention_mask,
        )

        ponder = self.ponder_loss(
            lru_output,
            mask=attention_mask,
        )

        # Combine losses
        total_loss = (
            weights['stability'] * stability +
            weights['sparsity'] * sparsity +
            weights['ponder'] * ponder
        )

        # Compute metrics
        with torch.no_grad():
            metrics = {
                'lru/avg_iterations': lru_output.num_iterations.mean(),
                'lru/max_iterations': lru_output.num_iterations.max(),
                'lru/min_iterations': lru_output.num_iterations.min(),
                'lru/avg_remainder': lru_output.remainders.mean(),
                'lru/stability_loss': stability,
                'lru/sparsity_loss': sparsity,
                'lru/ponder_loss': ponder,
                'lru/total_loss': total_loss,
            }

            # Add weight values to metrics if learnable
            if self.use_learnable_weights and self.learnable_weights is not None:
                weight_dict = self.learnable_weights.get_weight_dict()
                metrics['lru/weight_stability'] = weight_dict['stability']
                metrics['lru/weight_sparsity'] = weight_dict['sparsity']
                metrics['lru/weight_ponder'] = weight_dict['ponder']

        return LRULossOutput(
            total_loss=total_loss,
            stability_loss=stability,
            sparsity_loss=sparsity,
            ponder_loss=ponder,
            metrics=metrics,
        )

    def step(self):
        """Advance training step for scheduling."""
        self.current_step += 1


class ConvergenceRatioLoss(nn.Module):
    """Loss that directly penalizes poor convergence.

    Measures the ratio of final to initial state change:
    L_conv = ||h_N - h_0|| / (||h_1 - h_0|| + eps)

    Lower ratio indicates better convergence.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        intermediate_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute convergence ratio loss.

        Args:
            intermediate_states: States [T, B, S, D]
            mask: Optional mask [B, S]

        Returns:
            Scalar convergence ratio loss
        """
        if intermediate_states is None or intermediate_states.shape[0] < 2:
            return torch.tensor(0.0, device=intermediate_states.device if intermediate_states is not None else 'cpu')

        # Initial and final differences from input (first state)
        initial_diff = torch.norm(intermediate_states[1] - intermediate_states[0], dim=-1)  # [B, S]
        final_diff = torch.norm(intermediate_states[-1] - intermediate_states[0], dim=-1)   # [B, S]

        # Ratio (lower is better)
        ratio = final_diff / (initial_diff + self.eps)

        if mask is not None:
            ratio = ratio * mask

        return ratio.mean()
