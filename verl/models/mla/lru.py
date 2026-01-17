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
Latent Reasoning Unit (LRU) - Iterative refinement in latent space.

The LRU performs "latent oscillation" - iterative processing of the
compressed KV representation using GRU-style gating. This allows the
model to perform multi-step reasoning within a single layer.

Key components:
1. GRU-style gates (reset, update) for stable recurrence
2. Adaptive Computation Time (ACT) for dynamic halting
3. Accumulated output for gradient-friendly pondering

The forward pass:
    c_kv [B, S, d_c]  (initial latent from MLA)
         │
    ┌────┴────┐
    │ iterate │ ← up to max_iterations
    │ GRU step│
    │ + halt  │
    └────┬────┘
         │
    c_kv' [B, S, d_c]  (refined latent)
    + halting_info (for loss computation)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import LRUConfig


@dataclass
class LRUOutput:
    """Output container for LRU forward pass."""
    output: torch.Tensor  # Final refined latent [B, S, d_c]
    halting_probabilities: torch.Tensor  # Per-position halt probs [B, S]
    num_iterations: torch.Tensor  # Per-position iteration counts [B, S]
    remainders: torch.Tensor  # Remainder probabilities for ACT [B, S]
    intermediate_states: Optional[torch.Tensor] = None  # For stability loss


class LatentReasoningUnit(nn.Module):
    """Latent Reasoning Unit with GRU gating and ACT halting.

    Performs iterative refinement of the latent KV representation,
    allowing multi-step reasoning within a single attention layer.

    Args:
        config: LRUConfig with hyperparameters
    """

    def __init__(self, config: LRUConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.max_iterations = config.max_iterations
        self.halt_threshold = config.halt_threshold
        self.use_layer_norm = config.use_layer_norm
        self.gradient_checkpointing = config.gradient_checkpointing

        # GRU-style gates
        # Input: concatenation of [current_state, original_input]
        self.reset_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.update_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.candidate = nn.Linear(2 * self.latent_dim, self.latent_dim)

        # Halting unit
        self.halt_proj = nn.Linear(self.latent_dim, 1)

        # Initialize halt bias to encourage more iterations initially
        nn.init.constant_(self.halt_proj.bias, config.init_halt_bias)

        # Optional layer normalization
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.latent_dim)

        # Activation functions
        if config.gate_activation == 'sigmoid':
            self.gate_act = torch.sigmoid
        else:  # hard_sigmoid
            self.gate_act = lambda x: F.hardtanh(x * 0.2 + 0.5, 0, 1)

        if config.candidate_activation == 'tanh':
            self.candidate_act = torch.tanh
        else:  # gelu
            self.candidate_act = F.gelu

    def _gru_step(
        self,
        state: torch.Tensor,
        input_: torch.Tensor,
    ) -> torch.Tensor:
        """Single GRU-style update step.

        Args:
            state: Current hidden state [B, S, d_c]
            input_: Original input (constant across iterations) [B, S, d_c]

        Returns:
            New state [B, S, d_c]
        """
        # Concatenate state and input
        combined = torch.cat([state, input_], dim=-1)  # [B, S, 2*d_c]

        # Reset gate: controls how much of the old state to forget
        r = self.gate_act(self.reset_gate(combined))  # [B, S, d_c]

        # Update gate: controls how much of new state to use
        z = self.gate_act(self.update_gate(combined))  # [B, S, d_c]

        # Candidate state: new proposed state
        combined_reset = torch.cat([r * state, input_], dim=-1)
        h_tilde = self.candidate_act(self.candidate(combined_reset))  # [B, S, d_c]

        # New state: interpolation between old and candidate
        new_state = (1 - z) * state + z * h_tilde

        if self.use_layer_norm:
            new_state = self.layer_norm(new_state)

        return new_state

    def _compute_halt_probability(self, state: torch.Tensor) -> torch.Tensor:
        """Compute halting probability for ACT.

        Args:
            state: Current state [B, S, d_c]

        Returns:
            Halting probability [B, S] in range [0, 1]
        """
        return torch.sigmoid(self.halt_proj(state)).squeeze(-1)

    def forward(
        self,
        c_kv: torch.Tensor,
        return_intermediates: bool = True,
    ) -> Tuple[torch.Tensor, LRUOutput]:
        """
        Forward pass with iterative refinement and ACT halting.

        Args:
            c_kv: Compressed KV latent [batch_size, seq_len, latent_dim]
            return_intermediates: Whether to return intermediate states
                                 (needed for stability loss)

        Returns:
            Tuple of (refined_latent, LRUOutput)
        """
        batch_size, seq_len, _ = c_kv.shape
        device = c_kv.device
        dtype = c_kv.dtype

        # Initialize state and ACT tracking
        state = c_kv  # [B, S, d_c]
        accumulated_output = torch.zeros_like(c_kv)  # Weighted sum of states
        accumulated_halt = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        num_iterations = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        remainders = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        # For stability loss
        intermediate_states = [] if return_intermediates else None

        # Active mask: positions that haven't halted yet
        active = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

        for t in range(self.max_iterations):
            if not active.any():
                break

            # GRU step (with optional gradient checkpointing)
            if self.gradient_checkpointing and self.training:
                new_state = checkpoint(self._gru_step, state, c_kv, use_reentrant=False)
            else:
                new_state = self._gru_step(state, c_kv)

            # Store intermediate state for stability loss
            if return_intermediates:
                intermediate_states.append(new_state.clone())

            # Compute halting probability
            halt_prob = self._compute_halt_probability(new_state)  # [B, S]

            # Determine which positions halt this iteration
            # A position halts if accumulated_halt + halt_prob >= threshold
            new_accumulated = accumulated_halt + halt_prob * active.float()

            # Handle halting for this iteration
            if t == self.max_iterations - 1:
                # Last iteration: force halt, use remainder
                weight = 1.0 - accumulated_halt
                remainders = weight.clone()
            else:
                # Check if position should halt
                halting_now = (new_accumulated >= self.halt_threshold) & active

                # For positions halting now: use exact amount to reach threshold
                # For others: use full halt_prob
                weight = torch.where(
                    halting_now,
                    1.0 - accumulated_halt,  # Remainder to reach threshold
                    halt_prob * active.float()
                )

                # Update remainder for halting positions
                remainders = torch.where(
                    halting_now,
                    weight,
                    remainders
                )

                # Update active mask
                active = active & ~halting_now

            # Accumulate weighted output
            accumulated_output = accumulated_output + weight.unsqueeze(-1) * new_state
            accumulated_halt = new_accumulated.clamp(max=1.0)

            # Track iterations (increment for active positions)
            num_iterations = num_iterations + active.float()

            # Update state for next iteration
            state = new_state

        # Stack intermediate states if collected
        if return_intermediates and intermediate_states:
            intermediate_states = torch.stack(intermediate_states, dim=0)  # [T, B, S, d_c]
        else:
            intermediate_states = None

        # Create output container
        lru_output = LRUOutput(
            output=accumulated_output,
            halting_probabilities=accumulated_halt,
            num_iterations=num_iterations + 1,  # Add 1 since we start counting from 0
            remainders=remainders,
            intermediate_states=intermediate_states,
        )

        return accumulated_output, lru_output


class SimpleLRU(nn.Module):
    """Simplified LRU without ACT (fixed iterations).

    Useful for ablation studies comparing adaptive vs fixed computation.
    """

    def __init__(self, config: LRUConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_iterations = config.max_iterations
        self.use_layer_norm = config.use_layer_norm

        # GRU-style gates
        self.reset_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.update_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.candidate = nn.Linear(2 * self.latent_dim, self.latent_dim)

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.latent_dim)

    def forward(
        self,
        c_kv: torch.Tensor,
        return_intermediates: bool = True,
    ) -> Tuple[torch.Tensor, LRUOutput]:
        """Fixed iteration forward pass."""
        batch_size, seq_len, _ = c_kv.shape
        device = c_kv.device
        dtype = c_kv.dtype

        state = c_kv
        intermediate_states = [] if return_intermediates else None

        for t in range(self.num_iterations):
            combined = torch.cat([state, c_kv], dim=-1)
            r = torch.sigmoid(self.reset_gate(combined))
            z = torch.sigmoid(self.update_gate(combined))
            combined_reset = torch.cat([r * state, c_kv], dim=-1)
            h_tilde = torch.tanh(self.candidate(combined_reset))
            state = (1 - z) * state + z * h_tilde

            if self.use_layer_norm:
                state = self.layer_norm(state)

            if return_intermediates:
                intermediate_states.append(state.clone())

        if return_intermediates and intermediate_states:
            intermediate_states = torch.stack(intermediate_states, dim=0)

        # For fixed iterations, all positions have same count and no remainders
        lru_output = LRUOutput(
            output=state,
            halting_probabilities=torch.ones(batch_size, seq_len, device=device, dtype=dtype),
            num_iterations=torch.full((batch_size, seq_len), self.num_iterations, device=device, dtype=dtype),
            remainders=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            intermediate_states=intermediate_states,
        )

        return state, lru_output
