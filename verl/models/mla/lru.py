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

IMPROVEMENTS (based on academic review):
1. Added positional mixing for cross-position interaction
2. Added global halting mechanism alongside per-position halting
3. Fixed SimpleLRU to use weighted accumulation (consistent with LatentReasoningUnit)

Key components:
1. GRU-style gates (reset, update) for stable recurrence
2. Adaptive Computation Time (ACT) for dynamic halting
3. Accumulated output for gradient-friendly pondering
4. Positional mixing for cross-token reasoning

The forward pass:
    c_kv [B, S, d_c]  (initial latent from MLA)
         │
    ┌────┴────┐
    │ iterate │ ← up to max_iterations
    │ GRU step│
    │ pos_mix │ ← NEW: cross-position interaction
    │ + halt  │
    └────┬────┘
         │
    c_kv' [B, S, d_c]  (refined latent)
    + halting_info (for loss computation)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Literal

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
    ponder_cost: Optional[torch.Tensor] = None  # N + R for ponder loss


class PositionalMixing(nn.Module):
    """Cross-position interaction module.

    Allows information flow between positions during LRU iteration.
    Uses causal convolution to maintain autoregressive property.
    """

    def __init__(
        self,
        latent_dim: int,
        kernel_size: int = 3,
        causal: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.causal = causal

        # Depthwise separable convolution for efficiency
        self.depthwise = nn.Conv1d(
            latent_dim, latent_dim,
            kernel_size=kernel_size,
            padding=kernel_size - 1 if causal else kernel_size // 2,
            groups=latent_dim,  # Depthwise
        )
        self.pointwise = nn.Linear(latent_dim, latent_dim)
        self.gate = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, S, D]

        Returns:
            Mixed tensor [B, S, D]
        """
        # x: [B, S, D] -> [B, D, S] for conv1d
        x_conv = x.transpose(1, 2)

        # Depthwise conv
        mixed = self.depthwise(x_conv)

        # Causal: remove future positions
        if self.causal:
            mixed = mixed[:, :, :x.shape[1]]

        # [B, D, S] -> [B, S, D]
        mixed = mixed.transpose(1, 2)

        # Pointwise projection
        mixed = self.pointwise(mixed)

        # Gated residual connection
        gate_input = torch.cat([x, mixed], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))

        return x + gate * (mixed - x)


class GlobalHaltingUnit(nn.Module):
    """Global halting mechanism for sequence-level stopping.

    Computes a global halt signal that combines with per-position halting.
    This allows the model to express "the entire sequence is done reasoning".
    """

    def __init__(
        self,
        latent_dim: int,
        global_weight: float = 0.3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.global_weight = global_weight

        # Global halt projection (from pooled sequence)
        self.global_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        local_halt: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state: Current state [B, S, D]
            local_halt: Per-position halt probabilities [B, S]
            attention_mask: Mask for valid positions [B, S]

        Returns:
            Combined halt probabilities [B, S]
        """
        # Mean pooling (masked if provided)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = state.mean(dim=1)  # [B, D]

        # Global halt probability
        global_halt = torch.sigmoid(self.global_proj(pooled))  # [B, 1]

        # Combine local and global
        combined = (1 - self.global_weight) * local_halt + self.global_weight * global_halt

        return combined


class LatentReasoningUnit(nn.Module):
    """Latent Reasoning Unit with GRU gating and ACT halting.

    Performs iterative refinement of the latent KV representation,
    allowing multi-step reasoning within a single attention layer.

    IMPROVEMENTS:
    - Added optional positional mixing for cross-position interaction
    - Added optional global halting for sequence-level stopping
    - Improved intermediate state tracking

    Args:
        config: LRUConfig with hyperparameters
        use_positional_mixing: Whether to enable cross-position interaction
        use_global_halting: Whether to enable global halting
    """

    def __init__(
        self,
        config: LRUConfig,
        use_positional_mixing: bool = True,
        use_global_halting: bool = True,
    ):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.max_iterations = config.max_iterations
        self.halt_threshold = config.halt_threshold
        self.use_layer_norm = config.use_layer_norm
        self.gradient_checkpointing = config.gradient_checkpointing
        self.use_positional_mixing = use_positional_mixing
        self.use_global_halting = use_global_halting

        # GRU-style gates
        # Input: concatenation of [current_state, original_input]
        self.reset_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.update_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.candidate = nn.Linear(2 * self.latent_dim, self.latent_dim)

        # Local halting unit
        self.halt_proj = nn.Linear(self.latent_dim, 1)

        # Initialize halt bias to encourage more iterations initially
        nn.init.constant_(self.halt_proj.bias, config.init_halt_bias)

        # Optional positional mixing (NEW)
        if use_positional_mixing:
            self.pos_mixing = PositionalMixing(
                latent_dim=self.latent_dim,
                kernel_size=3,
                causal=True,
            )
        else:
            self.pos_mixing = None

        # Optional global halting (NEW)
        if use_global_halting:
            self.global_halt = GlobalHaltingUnit(
                latent_dim=self.latent_dim,
                global_weight=0.3,
            )
        else:
            self.global_halt = None

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

        # Apply positional mixing (NEW)
        if self.pos_mixing is not None:
            new_state = self.pos_mixing(new_state)

        if self.use_layer_norm:
            new_state = self.layer_norm(new_state)

        return new_state

    def _compute_halt_probability(
        self,
        state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute halting probability for ACT.

        Args:
            state: Current state [B, S, d_c]
            attention_mask: Optional mask [B, S]

        Returns:
            Halting probability [B, S] in range [0, 1]
        """
        local_halt = torch.sigmoid(self.halt_proj(state)).squeeze(-1)

        # Combine with global halting if enabled (NEW)
        if self.global_halt is not None:
            return self.global_halt(state, local_halt, attention_mask)

        return local_halt

    def forward(
        self,
        c_kv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = True,
    ) -> Tuple[torch.Tensor, LRUOutput]:
        """
        Forward pass with iterative refinement and ACT halting.

        Args:
            c_kv: Compressed KV latent [batch_size, seq_len, latent_dim]
            attention_mask: Optional attention mask [batch_size, seq_len]
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
            halt_prob = self._compute_halt_probability(new_state, attention_mask)  # [B, S]

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

        # Compute ponder cost (N + R)
        ponder_cost = (num_iterations + 1) + remainders

        # Create output container
        lru_output = LRUOutput(
            output=accumulated_output,
            halting_probabilities=accumulated_halt,
            num_iterations=num_iterations + 1,  # Add 1 since we start counting from 0
            remainders=remainders,
            intermediate_states=intermediate_states,
            ponder_cost=ponder_cost,
        )

        return accumulated_output, lru_output


class SimpleLRU(nn.Module):
    """Simplified LRU without ACT (fixed iterations).

    Useful for ablation studies comparing adaptive vs fixed computation.

    FIXED: Now uses weighted accumulation like LatentReasoningUnit for fair comparison.
    Each iteration contributes equally (1/N weight) to the output.
    """

    def __init__(self, config: LRUConfig, use_positional_mixing: bool = True):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.num_iterations = config.max_iterations
        self.use_layer_norm = config.use_layer_norm
        self.use_positional_mixing = use_positional_mixing

        # GRU-style gates
        self.reset_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.update_gate = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.candidate = nn.Linear(2 * self.latent_dim, self.latent_dim)

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.latent_dim)

        # Optional positional mixing
        if use_positional_mixing:
            self.pos_mixing = PositionalMixing(
                latent_dim=self.latent_dim,
                kernel_size=3,
                causal=True,
            )
        else:
            self.pos_mixing = None

    def forward(
        self,
        c_kv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_intermediates: bool = True,
    ) -> Tuple[torch.Tensor, LRUOutput]:
        """Fixed iteration forward pass with weighted accumulation."""
        batch_size, seq_len, _ = c_kv.shape
        device = c_kv.device
        dtype = c_kv.dtype

        state = c_kv
        intermediate_states = [] if return_intermediates else None

        # FIXED: Use weighted accumulation like LatentReasoningUnit
        accumulated_output = torch.zeros_like(c_kv)
        weight_per_iter = 1.0 / self.num_iterations

        for t in range(self.num_iterations):
            combined = torch.cat([state, c_kv], dim=-1)
            r = torch.sigmoid(self.reset_gate(combined))
            z = torch.sigmoid(self.update_gate(combined))
            combined_reset = torch.cat([r * state, c_kv], dim=-1)
            h_tilde = torch.tanh(self.candidate(combined_reset))
            state = (1 - z) * state + z * h_tilde

            # Apply positional mixing
            if self.pos_mixing is not None:
                state = self.pos_mixing(state)

            if self.use_layer_norm:
                state = self.layer_norm(state)

            # FIXED: Accumulate weighted output
            accumulated_output = accumulated_output + weight_per_iter * state

            if return_intermediates:
                intermediate_states.append(state.clone())

        if return_intermediates and intermediate_states:
            intermediate_states = torch.stack(intermediate_states, dim=0)

        # For fixed iterations, all positions have same count
        # Ponder cost is fixed at num_iterations
        ponder_cost = torch.full(
            (batch_size, seq_len), self.num_iterations,
            device=device, dtype=dtype
        )

        lru_output = LRUOutput(
            output=accumulated_output,  # FIXED: Use accumulated instead of final state
            halting_probabilities=torch.ones(batch_size, seq_len, device=device, dtype=dtype),
            num_iterations=torch.full((batch_size, seq_len), self.num_iterations, device=device, dtype=dtype),
            remainders=torch.zeros(batch_size, seq_len, device=device, dtype=dtype),
            intermediate_states=intermediate_states,
            ponder_cost=ponder_cost,
        )

        return accumulated_output, lru_output


class UniversalTransformerLRU(nn.Module):
    """Universal Transformer style LRU.

    Instead of iterating within a single layer, this module wraps
    an entire decoder layer and iterates at the layer level.

    This addresses the review concern that per-position iteration
    lacks cross-token interaction - here the full attention is
    applied at each iteration.

    Usage:
        decoder_layer = DeepSeekMLADecoderLayer(config, layer_idx)
        ut_layer = UniversalTransformerLRU(decoder_layer, config)
        output = ut_layer(hidden_states, ...)
    """

    def __init__(
        self,
        decoder_layer: nn.Module,
        max_iterations: int = 4,
        halt_threshold: float = 0.99,
        share_weights: bool = True,
    ):
        super().__init__()
        self.decoder_layer = decoder_layer
        self.max_iterations = max_iterations
        self.halt_threshold = halt_threshold
        self.share_weights = share_weights

        # Get hidden size from decoder layer
        hidden_size = decoder_layer.hidden_size

        # Halting mechanism (sequence-level)
        self.halt_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )
        nn.init.constant_(self.halt_proj[-1].bias, -2.0)

        # Iteration embedding (like position embedding but for iterations)
        self.iteration_embed = nn.Embedding(max_iterations, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with layer-level iteration.

        Returns same format as decoder_layer for drop-in compatibility.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Initialize
        state = hidden_states
        accumulated_output = torch.zeros_like(hidden_states)
        accumulated_halt = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        active = torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)
        num_iterations = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)
        remainders = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

        all_attentions = []

        for t in range(self.max_iterations):
            if not active.any():
                break

            # Add iteration embedding
            # iteration_embed([t]) gives [1, D], then unsqueeze(1) gives [1, 1, D]
            iter_embed = self.iteration_embed(
                torch.tensor([t], device=device)
            ).unsqueeze(1)  # [1, 1, D]
            state_with_iter = state + iter_embed

            # Apply decoder layer
            layer_outputs = self.decoder_layer(
                state_with_iter,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,  # Don't use cache in iteration
                output_attentions=output_attentions,
                use_cache=False,
                **kwargs,
            )

            new_state = layer_outputs[0]

            if output_attentions and len(layer_outputs) > 1:
                all_attentions.append(layer_outputs[1])

            # Compute halt probability (pooled over sequence)
            pooled = new_state.mean(dim=1)  # [B, D]
            halt_prob = torch.sigmoid(self.halt_proj(pooled)).squeeze(-1)  # [B]
            halt_prob = halt_prob.unsqueeze(-1).expand(-1, seq_len)  # [B, S]

            # ACT halting logic
            new_accumulated = accumulated_halt + halt_prob * active.float()

            if t == self.max_iterations - 1:
                # Last iteration: force halt, use remainder
                weight = 1.0 - accumulated_halt
                remainders = weight.clone()
            else:
                halting_now = (new_accumulated >= self.halt_threshold) & active
                weight = torch.where(
                    halting_now,
                    1.0 - accumulated_halt,
                    halt_prob * active.float()
                )
                # Update remainder for halting positions
                remainders = torch.where(
                    halting_now,
                    weight,
                    remainders
                )
                active = active & ~halting_now

            accumulated_output = accumulated_output + weight.unsqueeze(-1) * new_state
            accumulated_halt = new_accumulated.clamp(max=1.0)
            # Track iterations (increment for active positions)
            num_iterations = num_iterations + active.float()
            state = new_state

        # Create LRU output for statistics
        ponder_cost = (num_iterations + 1) + remainders
        lru_output = LRUOutput(
            output=accumulated_output,
            halting_probabilities=accumulated_halt,
            num_iterations=num_iterations + 1,  # Add 1 since we start from 0
            remainders=remainders,
            intermediate_states=None,
            ponder_cost=ponder_cost,
        )

        # Build output tuple matching decoder_layer format
        outputs = (accumulated_output,)

        if output_attentions:
            # Average attention weights across iterations
            if all_attentions:
                avg_attention = torch.stack(all_attentions, dim=0).mean(dim=0)
                outputs += (avg_attention,)

        if use_cache:
            # Return final state as cache
            outputs += (past_key_value,)

        return outputs, lru_output
