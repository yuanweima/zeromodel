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
Halting mechanisms for Adaptive Computation Time (ACT).

Provides different strategies for determining when to stop iterating
in the Latent Reasoning Unit.

References:
- Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks
- Dehghani, M., et al. (2018). Universal Transformers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HaltingState:
    """State container for halting computation."""
    accumulated_halt: torch.Tensor  # Cumulative halting probability [B, S]
    accumulated_output: torch.Tensor  # Weighted sum of states [B, S, D]
    num_iterations: torch.Tensor  # Count of iterations [B, S]
    remainders: torch.Tensor  # Final remainder probabilities [B, S]
    active: torch.Tensor  # Which positions are still active [B, S]


class HaltingUnit(nn.Module, ABC):
    """Base class for halting mechanisms."""

    @abstractmethod
    def compute_halt_probability(
        self,
        state: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Compute halting probability for current state.

        Args:
            state: Current hidden state [B, S, D]
            iteration: Current iteration number

        Returns:
            Halting probability [B, S]
        """
        pass

    @abstractmethod
    def initialize_state(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> HaltingState:
        """Initialize halting state for a new sequence."""
        pass

    @abstractmethod
    def update_state(
        self,
        halting_state: HaltingState,
        new_hidden: torch.Tensor,
        halt_prob: torch.Tensor,
        iteration: int,
        is_last: bool,
    ) -> HaltingState:
        """Update halting state after one iteration."""
        pass


class ACTHaltingUnit(HaltingUnit):
    """Adaptive Computation Time (ACT) halting mechanism.

    Implements the halting mechanism from Graves (2016):
    - Each iteration produces a halt probability h_t
    - Accumulated halt probability R_t = sum_{s<=t} h_s
    - Stop when R_t >= 1 - epsilon
    - Final output is weighted sum: sum_t h_t * state_t
    """

    def __init__(
        self,
        hidden_dim: int,
        threshold: float = 0.99,
        init_bias: float = -2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        # Halting probability projection
        self.halt_proj = nn.Linear(hidden_dim, 1)
        nn.init.constant_(self.halt_proj.bias, init_bias)

    def compute_halt_probability(
        self,
        state: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Compute halting probability using learned projection."""
        return torch.sigmoid(self.halt_proj(state)).squeeze(-1)

    def initialize_state(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> HaltingState:
        """Initialize halting state."""
        return HaltingState(
            accumulated_halt=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            accumulated_output=torch.zeros(batch_size, seq_len, hidden_dim, device=device, dtype=dtype),
            num_iterations=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            remainders=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            active=torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
        )

    def update_state(
        self,
        halting_state: HaltingState,
        new_hidden: torch.Tensor,
        halt_prob: torch.Tensor,
        iteration: int,
        is_last: bool,
    ) -> HaltingState:
        """Update halting state after one iteration."""
        active = halting_state.active
        accumulated_halt = halting_state.accumulated_halt

        if is_last:
            # Force halt on last iteration
            weight = 1.0 - accumulated_halt
            new_remainders = weight.clone()
            new_active = torch.zeros_like(active)
        else:
            # Check if position should halt
            new_accumulated = accumulated_halt + halt_prob * active.float()
            halting_now = (new_accumulated >= self.threshold) & active

            # Compute weight
            weight = torch.where(
                halting_now,
                1.0 - accumulated_halt,
                halt_prob * active.float()
            )

            # Update remainders for halting positions
            new_remainders = torch.where(
                halting_now,
                weight,
                halting_state.remainders
            )

            # Update active mask
            new_active = active & ~halting_now
            accumulated_halt = new_accumulated.clamp(max=1.0)

        # Update accumulated output
        new_accumulated_output = (
            halting_state.accumulated_output + weight.unsqueeze(-1) * new_hidden
        )

        # Update iteration count
        new_num_iterations = halting_state.num_iterations + active.float()

        return HaltingState(
            accumulated_halt=accumulated_halt,
            accumulated_output=new_accumulated_output,
            num_iterations=new_num_iterations,
            remainders=new_remainders,
            active=new_active,
        )


class ConfidenceHaltingUnit(HaltingUnit):
    """Confidence-based halting mechanism.

    Halts when the model's confidence (based on state change) exceeds
    a threshold. This measures whether the state has converged.
    """

    def __init__(
        self,
        hidden_dim: int,
        threshold: float = 0.99,
        convergence_threshold: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.threshold = threshold
        self.convergence_threshold = convergence_threshold

        self.halt_proj = nn.Linear(hidden_dim, 1)
        self._prev_state: Optional[torch.Tensor] = None

    def compute_halt_probability(
        self,
        state: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Compute halting probability based on state convergence."""
        # Base halting probability from learned projection
        base_halt = torch.sigmoid(self.halt_proj(state)).squeeze(-1)

        # Convergence bonus: if state has stabilized, increase halt probability
        if self._prev_state is not None:
            state_change = torch.norm(state - self._prev_state, dim=-1)  # [B, S]
            convergence_factor = torch.exp(-state_change / self.convergence_threshold)
            halt_prob = base_halt + (1 - base_halt) * convergence_factor * 0.5
        else:
            halt_prob = base_halt

        self._prev_state = state.detach()
        return halt_prob

    def initialize_state(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> HaltingState:
        """Initialize halting state and reset prev_state."""
        self._prev_state = None
        return HaltingState(
            accumulated_halt=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            accumulated_output=torch.zeros(batch_size, seq_len, hidden_dim, device=device, dtype=dtype),
            num_iterations=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            remainders=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            active=torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
        )

    def update_state(
        self,
        halting_state: HaltingState,
        new_hidden: torch.Tensor,
        halt_prob: torch.Tensor,
        iteration: int,
        is_last: bool,
    ) -> HaltingState:
        """Update state (same logic as ACT)."""
        # Reuse ACT update logic
        active = halting_state.active
        accumulated_halt = halting_state.accumulated_halt

        if is_last:
            weight = 1.0 - accumulated_halt
            new_remainders = weight.clone()
            new_active = torch.zeros_like(active)
            accumulated_halt = torch.ones_like(accumulated_halt)
        else:
            new_accumulated = accumulated_halt + halt_prob * active.float()
            halting_now = (new_accumulated >= self.threshold) & active

            weight = torch.where(
                halting_now,
                1.0 - accumulated_halt,
                halt_prob * active.float()
            )

            new_remainders = torch.where(
                halting_now,
                weight,
                halting_state.remainders
            )

            new_active = active & ~halting_now
            accumulated_halt = new_accumulated.clamp(max=1.0)

        new_accumulated_output = (
            halting_state.accumulated_output + weight.unsqueeze(-1) * new_hidden
        )
        new_num_iterations = halting_state.num_iterations + active.float()

        return HaltingState(
            accumulated_halt=accumulated_halt,
            accumulated_output=new_accumulated_output,
            num_iterations=new_num_iterations,
            remainders=new_remainders,
            active=new_active,
        )


class DynamicHaltingUnit(HaltingUnit):
    """Dynamic halting with learnable iteration-dependent thresholds.

    Learns different halting behaviors for different iteration depths,
    allowing the model to adapt its computation budget based on the
    iteration number.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_iterations: int = 8,
        threshold: float = 0.99,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_iterations = max_iterations
        self.threshold = threshold

        # Iteration-specific halting projections
        self.halt_projs = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(max_iterations)
        ])

        # Initialize with decreasing bias (later iterations more likely to halt)
        for i, proj in enumerate(self.halt_projs):
            bias = -3.0 + (i / max_iterations) * 4.0  # -3 to +1
            nn.init.constant_(proj.bias, bias)

    def compute_halt_probability(
        self,
        state: torch.Tensor,
        iteration: int,
    ) -> torch.Tensor:
        """Compute iteration-specific halting probability."""
        idx = min(iteration, self.max_iterations - 1)
        return torch.sigmoid(self.halt_projs[idx](state)).squeeze(-1)

    def initialize_state(
        self,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> HaltingState:
        """Initialize halting state."""
        return HaltingState(
            accumulated_halt=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            accumulated_output=torch.zeros(batch_size, seq_len, hidden_dim, device=device, dtype=dtype),
            num_iterations=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            remainders=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            active=torch.ones(batch_size, seq_len, device=device, dtype=torch.bool),
        )

    def update_state(
        self,
        halting_state: HaltingState,
        new_hidden: torch.Tensor,
        halt_prob: torch.Tensor,
        iteration: int,
        is_last: bool,
    ) -> HaltingState:
        """Update state with standard ACT logic."""
        active = halting_state.active
        accumulated_halt = halting_state.accumulated_halt

        if is_last:
            weight = 1.0 - accumulated_halt
            new_remainders = weight.clone()
            new_active = torch.zeros_like(active)
            accumulated_halt = torch.ones_like(accumulated_halt)
        else:
            new_accumulated = accumulated_halt + halt_prob * active.float()
            halting_now = (new_accumulated >= self.threshold) & active

            weight = torch.where(
                halting_now,
                1.0 - accumulated_halt,
                halt_prob * active.float()
            )

            new_remainders = torch.where(
                halting_now,
                weight,
                halting_state.remainders
            )

            new_active = active & ~halting_now
            accumulated_halt = new_accumulated.clamp(max=1.0)

        new_accumulated_output = (
            halting_state.accumulated_output + weight.unsqueeze(-1) * new_hidden
        )
        new_num_iterations = halting_state.num_iterations + active.float()

        return HaltingState(
            accumulated_halt=accumulated_halt,
            accumulated_output=new_accumulated_output,
            num_iterations=new_num_iterations,
            remainders=new_remainders,
            active=new_active,
        )


def create_halting_unit(
    halting_type: str,
    hidden_dim: int,
    max_iterations: int = 8,
    threshold: float = 0.99,
    **kwargs,
) -> HaltingUnit:
    """Factory function to create halting units.

    Args:
        halting_type: One of 'act', 'confidence', 'dynamic'
        hidden_dim: Hidden dimension size
        max_iterations: Maximum number of iterations
        threshold: Halting probability threshold
        **kwargs: Additional arguments for specific halting types

    Returns:
        HaltingUnit instance
    """
    if halting_type == 'act':
        return ACTHaltingUnit(
            hidden_dim=hidden_dim,
            threshold=threshold,
            init_bias=kwargs.get('init_bias', -2.0),
        )
    elif halting_type == 'confidence':
        return ConfidenceHaltingUnit(
            hidden_dim=hidden_dim,
            threshold=threshold,
            convergence_threshold=kwargs.get('convergence_threshold', 0.01),
        )
    elif halting_type == 'dynamic':
        return DynamicHaltingUnit(
            hidden_dim=hidden_dim,
            max_iterations=max_iterations,
            threshold=threshold,
        )
    else:
        raise ValueError(f"Unknown halting type: {halting_type}")
