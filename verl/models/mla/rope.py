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
Decoupled Rotary Position Embedding (RoPE) for MLA.

In MLA, RoPE is only applied to a portion of each head's dimension (rope_head_dim),
leaving the rest (nope_head_dim) without positional encoding. This allows the
compressed latent space to remain position-agnostic while still preserving
relative position information through the RoPE portion.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class DecoupledRotaryEmbedding(nn.Module):
    """Decoupled Rotary Position Embedding.

    Only applies RoPE to the first `rope_dim` dimensions of each head,
    leaving the remaining dimensions unchanged.

    Args:
        rope_dim: Number of dimensions to apply RoPE to
        max_position_embeddings: Maximum sequence length
        base: Base for frequency computation (default: 10000)
        scaling_factor: Optional scaling factor for extended context
        scaling_type: Type of scaling ('linear', 'dynamic', 'yarn', None)
    """

    def __init__(
        self,
        rope_dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_type: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.rope_dim = rope_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.scaling_type = scaling_type

        # Compute inverse frequencies
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache for cos/sin values
        self._cos_cached = None
        self._sin_cached = None
        self._cached_seq_len = 0

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute inverse frequencies for RoPE."""
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device) / self.rope_dim)
        )

        if self.scaling_type == "linear":
            inv_freq = inv_freq / self.scaling_factor
        elif self.scaling_type == "dynamic":
            # Dynamic NTK scaling - computed during forward pass
            pass
        elif self.scaling_type == "yarn":
            # YaRN scaling - more complex, applied during forward
            pass

        return inv_freq

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update the cos/sin cache if needed."""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len

            # Apply dynamic scaling if needed
            inv_freq = self.inv_freq
            if self.scaling_type == "dynamic" and seq_len > self.max_position_embeddings:
                # Dynamic NTK-aware scaling
                base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
                ) ** (self.rope_dim / (self.rope_dim - 2))
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device) / self.rope_dim)
                )

            # Compute position embeddings
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)  # [seq_len, rope_dim // 2]

            # Different from paper: [seq_len, rope_dim] with interleaved cos/sin
            emb = torch.cat((freqs, freqs), dim=-1)

            self._cos_cached = emb.cos().to(dtype)
            self._sin_cached = emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for RoPE.

        Args:
            x: Input tensor [batch_size, seq_len, ...]
            position_ids: Position indices [batch_size, seq_len]
            seq_len: Sequence length (inferred from x if not provided)

        Returns:
            Tuple of (cos, sin) tensors for RoPE application
        """
        if seq_len is None:
            seq_len = x.shape[1]

        self._update_cache(seq_len, x.device, x.dtype)

        if position_ids is not None:
            # Gather cos/sin for specific positions
            cos = self._cos_cached[position_ids]  # [batch, seq_len, rope_dim]
            sin = self._sin_cached[position_ids]
        else:
            cos = self._cos_cached[:seq_len]  # [seq_len, rope_dim]
            sin = self._sin_cached[:seq_len]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.LongTensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine values from RoPE
        sin: Sine values from RoPE
        position_ids: Position indices (for gathering if needed)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of (q_embed, k_embed) with RoPE applied
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


def apply_rotary_pos_emb_decoupled(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    position_ids: Optional[torch.LongTensor] = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply decoupled rotary position embedding.

    Only applies RoPE to the first `rope_dim` dimensions of each head,
    leaving the remaining dimensions unchanged.

    Args:
        q: Query tensor [..., head_dim]
        k: Key tensor [..., head_dim]
        cos: Cosine values [seq_len, rope_dim] or [batch, seq_len, rope_dim]
        sin: Sine values [seq_len, rope_dim] or [batch, seq_len, rope_dim]
        rope_dim: Number of dimensions to apply RoPE to
        position_ids: Position indices (for gathering if needed)
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting

    Returns:
        Tuple of (q_embed, k_embed) with partial RoPE applied
    """
    # Split into RoPE and non-RoPE dimensions
    q_rope = q[..., :rope_dim]
    q_nope = q[..., rope_dim:]
    k_rope = k[..., :rope_dim]
    k_nope = k[..., rope_dim:]

    # Apply RoPE only to the first rope_dim dimensions
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_rope_embed = (q_rope * cos) + (rotate_half(q_rope) * sin)
    k_rope_embed = (k_rope * cos) + (rotate_half(k_rope) * sin)

    # Concatenate RoPE and non-RoPE parts
    q_embed = torch.cat([q_rope_embed, q_nope], dim=-1)
    k_embed = torch.cat([k_rope_embed, k_nope], dim=-1)

    return q_embed, k_embed


class YaRNScaledRotaryEmbedding(DecoupledRotaryEmbedding):
    """YaRN (Yet another RoPE extensioN) scaled rotary embedding.

    Implements the YaRN method for extending context length with
    attention scaling and frequency interpolation.
    """

    def __init__(
        self,
        rope_dim: int,
        max_position_embeddings: int = 4096,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        super().__init__(
            rope_dim=rope_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            scaling_factor=scaling_factor,
            scaling_type="yarn",
            device=device,
        )

    def _compute_inv_freq(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Compute YaRN-scaled inverse frequencies."""
        pos_freqs = self.base ** (
            torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device) / self.rope_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scaling_factor * pos_freqs)

        low, high = self._yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rope_dim,
            self.base,
            self.original_max_position_embeddings,
        )

        # Linear ramp for frequency mixing
        inv_freq_mask = 1 - self._yarn_linear_ramp_mask(low, high, self.rope_dim // 2, device)
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        return inv_freq

    def _yarn_find_correction_range(
        self,
        beta_fast: float,
        beta_slow: float,
        dim: int,
        base: float,
        original_max_seq_len: int,
    ) -> Tuple[float, float]:
        """Find the correction range for YaRN."""
        low = math.floor(
            self._yarn_find_correction_dim(beta_fast, dim, base, original_max_seq_len)
        )
        high = math.ceil(
            self._yarn_find_correction_dim(beta_slow, dim, base, original_max_seq_len)
        )
        return max(low, 0), min(high, dim - 1)

    def _yarn_find_correction_dim(
        self,
        num_rotations: float,
        dim: int,
        base: float,
        max_seq_len: int,
    ) -> float:
        """Find correction dimension for YaRN."""
        return (dim * math.log(max_seq_len / (num_rotations * 2 * math.pi))) / (
            2 * math.log(base)
        )

    def _yarn_linear_ramp_mask(
        self,
        low: float,
        high: float,
        dim: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Create linear ramp mask for frequency mixing."""
        if high == low:
            high += 0.001  # Prevent division by zero

        linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - low) / (high - low)
        return torch.clamp(linear_func, 0.0, 1.0)

    def _get_mscale(self, scale: float = 1.0) -> float:
        """Get magnitude scaling factor."""
        if self.mscale_all_dim > 0:
            return (0.1 * math.log(scale) + 1.0) ** self.mscale_all_dim
        return (0.1 * math.log(scale) + 1.0) ** self.mscale
