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
Multi-head Latent Attention (MLA) implementation.

Based on DeepSeek-V3's MLA architecture:
- Compresses KV into a low-rank latent space
- Uses decoupled RoPE (only applied to part of head dimensions)
- Supports integration with LRU for latent space reasoning

Data flow:
    Input x [B, S, H]
           │
           ▼
    ┌─────────────────┐
    │ W_DKV (down)    │ → c_kv [B, S, d_c]  (latent compression)
    └─────────────────┘
           │
           ▼
    ┌─────────────────┐
    │ LRU (optional)  │ → c_kv' [B, S, d_c] (iterative reasoning)
    └─────────────────┘
           │
           ├──→ W_UK (up) → K [B, S, n_kv * head_dim]
           │
           └──→ W_UV (up) → V [B, S, n_kv * head_dim]

    ┌─────────────────┐
    │ W_DQ → W_UQ     │ → Q [B, S, n_heads * head_dim]
    └─────────────────┘
           │
           ▼
    Flash Attention → Output
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MLAConfig
from .rope import DecoupledRotaryEmbedding, apply_rotary_pos_emb_decoupled


class MLAAttention(nn.Module):
    """Multi-head Latent Attention with decoupled RoPE.

    This module implements the MLA mechanism from DeepSeek-V3:
    1. Projects input to a compressed latent space for KV
    2. Optionally applies LRU for iterative refinement
    3. Up-projects from latent to full K, V tensors
    4. Uses decoupled RoPE (only on part of head dimensions)
    5. Computes attention with Flash Attention 2
    """

    def __init__(
        self,
        config: MLAConfig,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.kv_latent_dim = config.kv_latent_dim
        self.q_latent_dim = config.q_latent_dim
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = config.nope_head_dim
        self.attention_dropout = config.attention_dropout

        # Compute group size for GQA
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # KV compression projections
        # Down projection: H -> d_c (latent)
        self.w_dkv = nn.Linear(
            self.hidden_size,
            self.kv_latent_dim,
            bias=config.attention_bias,
        )

        # Up projections from latent to K and V
        # K: d_c -> n_kv * (nope_head_dim) for content
        # K_rope: separately computed for position
        self.w_uk = nn.Linear(
            self.kv_latent_dim,
            self.num_kv_heads * self.nope_head_dim,
            bias=config.attention_bias,
        )
        self.w_uv = nn.Linear(
            self.kv_latent_dim,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )

        # Separate projection for K's RoPE dimensions
        # FIXED: Project from latent space (c_kv) so LRU refinement affects RoPE part too
        # This ensures the iterative reasoning in LRU influences both content and position
        self.w_k_rope = nn.Linear(
            self.kv_latent_dim,  # Changed from hidden_size to kv_latent_dim
            self.num_kv_heads * self.rope_head_dim,
            bias=config.attention_bias,
        )

        # Q projections (also compressed for efficiency in MLA)
        self.w_dq = nn.Linear(
            self.hidden_size,
            self.q_latent_dim,
            bias=config.attention_bias,
        )
        self.w_uq = nn.Linear(
            self.q_latent_dim,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )

        # Output projection
        self.w_o = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Rotary embedding for decoupled RoPE
        self.rotary_emb = DecoupledRotaryEmbedding(
            rope_dim=self.rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling.get("factor", 1.0) if config.rope_scaling else 1.0,
            scaling_type=config.rope_scaling.get("type") if config.rope_scaling else None,
        )

        # LRU module placeholder (set externally if used)
        self.lru = None

    def set_lru(self, lru_module: nn.Module):
        """Attach an LRU module for latent space reasoning."""
        self.lru = lru_module

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass for MLA attention.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_len, kv_seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_value: Cached (c_kv, k_rope) for incremental decoding
            output_attentions: Whether to output attention weights
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # === Q projection ===
        # Compress then expand Q
        q_latent = self.w_dq(hidden_states)  # [B, S, q_latent_dim]
        q = self.w_uq(q_latent)  # [B, S, n_heads * head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # === KV projection ===
        # Compress to latent space
        c_kv = self.w_dkv(hidden_states)  # [B, S, kv_latent_dim]

        # Apply LRU for iterative refinement if available
        lru_info = None
        if self.lru is not None:
            c_kv, lru_info = self.lru(c_kv)

        # Up-project to K (content part, without RoPE dims) and V
        k_nope = self.w_uk(c_kv)  # [B, S, n_kv * nope_head_dim]
        v = self.w_uv(c_kv)  # [B, S, n_kv * head_dim]

        # FIXED: K_rope now projects from refined c_kv, not hidden_states
        # This ensures LRU refinement affects both content and positional parts of K
        k_rope = self.w_k_rope(c_kv)  # [B, S, n_kv * rope_head_dim]

        # Reshape
        k_nope = k_nope.view(batch_size, seq_len, self.num_kv_heads, self.nope_head_dim)
        k_rope = k_rope.view(batch_size, seq_len, self.num_kv_heads, self.rope_head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # === Handle KV Cache ===
        if past_key_value is not None:
            past_c_kv, _ = past_key_value  # We no longer cache k_rope separately
            # Concatenate along sequence dimension
            c_kv = torch.cat([past_c_kv, c_kv], dim=1)

            # Recompute k_nope, k_rope, and v from full c_kv
            # FIXED: Now k_rope is also recomputed from c_kv for consistency
            kv_seq_len = c_kv.shape[1]
            k_nope = self.w_uk(c_kv).view(batch_size, kv_seq_len, self.num_kv_heads, self.nope_head_dim)
            k_rope = self.w_k_rope(c_kv).view(batch_size, kv_seq_len, self.num_kv_heads, self.rope_head_dim)
            v = self.w_uv(c_kv).view(batch_size, kv_seq_len, self.num_kv_heads, self.head_dim)

        kv_seq_len = k_nope.shape[1]

        # === Apply Decoupled RoPE ===
        # Get cos, sin for positions
        if position_ids is None:
            position_ids = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)

        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids[:, -seq_len:], seq_len=kv_seq_len)

        # Split Q into RoPE and non-RoPE parts
        q_rope = q[..., :self.rope_head_dim]
        q_nope = q[..., self.rope_head_dim:]

        # Apply RoPE only to rope dimensions
        # For Q: apply to q_rope
        q_rope_embed = q_rope * cos.unsqueeze(2) + self._rotate_half(q_rope) * sin.unsqueeze(2)
        q = torch.cat([q_rope_embed, q_nope], dim=-1)

        # For K: k_rope already separate, apply RoPE
        cos_full = self.rotary_emb._cos_cached[:kv_seq_len].unsqueeze(0).unsqueeze(2)
        sin_full = self.rotary_emb._sin_cached[:kv_seq_len].unsqueeze(0).unsqueeze(2)
        k_rope_embed = k_rope * cos_full + self._rotate_half(k_rope) * sin_full

        # Combine K parts
        k = torch.cat([k_rope_embed, k_nope], dim=-1)  # [B, kv_seq_len, n_kv, head_dim]

        # === Transpose for attention: [B, n_heads, S, head_dim] ===
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # === Expand KV for GQA if needed ===
        if self.num_key_value_groups > 1:
            k = self._repeat_kv(k, self.num_key_value_groups)
            v = self._repeat_kv(v, self.num_key_value_groups)

        # === Attention computation ===
        attn_output, attn_weights = self._attention(
            q, k, v,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # === Output projection ===
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        attn_output = self.w_o(attn_output)

        # === Prepare cache ===
        if use_cache:
            # Cache only the compressed c_kv (k_rope is recomputed from c_kv)
            # FIXED: Simplified cache since k_rope now derives from c_kv
            # The tuple format (c_kv, None) maintains backward compatibility
            past_key_value = (c_kv, None)
        else:
            past_key_value = None

        # Attach LRU info to output for loss computation
        if lru_info is not None:
            attn_output = (attn_output, lru_info)

        return attn_output, attn_weights, past_key_value

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for grouped-query attention."""
        if n_rep == 1:
            return x
        batch_size, num_kv_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch_size, num_kv_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch_size, num_kv_heads * n_rep, seq_len, head_dim)

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute scaled dot-product attention.

        Attempts to use Flash Attention 2 if available, falls back to
        standard attention otherwise.
        """
        try:
            # Try Flash Attention 2
            from flash_attn import flash_attn_func

            # Flash attention expects [B, S, H, D] format
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=True,
            )

            # Back to [B, H, S, D]
            attn_output = attn_output.transpose(1, 2)

            return attn_output, None

        except ImportError:
            # Fallback to standard attention
            return self._standard_attention(q, k, v, attention_mask, output_attentions)

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Standard scaled dot-product attention (fallback)."""
        # [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Causal mask
        seq_len = q.shape[2]
        kv_seq_len = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, kv_seq_len, dtype=torch.bool, device=q.device),
            diagonal=kv_seq_len - seq_len + 1,
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights


class MLAFlashAttention2(MLAAttention):
    """MLA with Flash Attention 2 as the primary attention mechanism.

    This is a variant that always uses Flash Attention 2 and provides
    additional optimizations for long sequences.
    """

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Always use Flash Attention 2."""
        from flash_attn import flash_attn_func

        # Flash attention expects [B, S, H, D] format
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True,
        )

        # Back to [B, H, S, D]
        attn_output = attn_output.transpose(1, 2)

        if output_attentions:
            # Flash attention doesn't return weights, return None
            return attn_output, None

        return attn_output, None
