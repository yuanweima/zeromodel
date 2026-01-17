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
Parameter counting utilities for fair comparison in ablation studies.

This module provides tools to:
1. Count parameters in different model configurations
2. Design parameter-matched baselines
3. Report parameter breakdowns by component
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ParameterBreakdown:
    """Detailed parameter count breakdown."""
    embedding: int
    attention: int
    mlp: int
    layer_norm: int
    lm_head: int
    lru: int  # LRU-specific parameters
    total: int

    def __str__(self) -> str:
        return f"""Parameter Breakdown:
  Embedding:    {self.embedding:>12,} ({100*self.embedding/self.total:.1f}%)
  Attention:    {self.attention:>12,} ({100*self.attention/self.total:.1f}%)
  MLP:          {self.mlp:>12,} ({100*self.mlp/self.total:.1f}%)
  Layer Norm:   {self.layer_norm:>12,} ({100*self.layer_norm/self.total:.1f}%)
  LM Head:      {self.lm_head:>12,} ({100*self.lm_head/self.total:.1f}%)
  LRU:          {self.lru:>12,} ({100*self.lru/self.total:.1f}%)
  ─────────────────────────────
  Total:        {self.total:>12,}"""


def count_standard_transformer_params(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_kv_heads: int,
    tie_embeddings: bool = False,
) -> ParameterBreakdown:
    """Count parameters in a standard transformer (like Qwen2).

    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        tie_embeddings: Whether embeddings are tied with LM head

    Returns:
        ParameterBreakdown with detailed counts
    """
    head_dim = hidden_size // num_attention_heads

    # Embedding
    embedding = vocab_size * hidden_size

    # Attention per layer (with GQA)
    # Q: hidden_size -> num_heads * head_dim
    # K: hidden_size -> num_kv_heads * head_dim
    # V: hidden_size -> num_kv_heads * head_dim
    # O: num_heads * head_dim -> hidden_size
    attn_per_layer = (
        hidden_size * num_attention_heads * head_dim +  # Q
        hidden_size * num_kv_heads * head_dim +         # K
        hidden_size * num_kv_heads * head_dim +         # V
        num_attention_heads * head_dim * hidden_size    # O
    )
    attention = attn_per_layer * num_layers

    # MLP per layer (SwiGLU style)
    # gate_proj: hidden_size -> intermediate_size
    # up_proj: hidden_size -> intermediate_size
    # down_proj: intermediate_size -> hidden_size
    mlp_per_layer = (
        hidden_size * intermediate_size +    # gate
        hidden_size * intermediate_size +    # up
        intermediate_size * hidden_size      # down
    )
    mlp = mlp_per_layer * num_layers

    # Layer norms (2 per layer + 1 final)
    ln_per_layer = 2 * hidden_size  # RMSNorm has no bias
    layer_norm = ln_per_layer * num_layers + hidden_size

    # LM head
    lm_head = 0 if tie_embeddings else vocab_size * hidden_size

    total = embedding + attention + mlp + layer_norm + lm_head

    return ParameterBreakdown(
        embedding=embedding,
        attention=attention,
        mlp=mlp,
        layer_norm=layer_norm,
        lm_head=lm_head,
        lru=0,
        total=total,
    )


def count_mla_params(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_kv_heads: int,
    kv_latent_dim: int,
    q_latent_dim: int,
    rope_head_dim: int,
    tie_embeddings: bool = False,
) -> ParameterBreakdown:
    """Count parameters in MLA model (without LRU).

    Args:
        kv_latent_dim: KV compression latent dimension
        q_latent_dim: Q compression latent dimension
        rope_head_dim: Dimension for RoPE application

    Returns:
        ParameterBreakdown with detailed counts
    """
    head_dim = hidden_size // num_attention_heads
    nope_head_dim = head_dim - rope_head_dim

    # Embedding
    embedding = vocab_size * hidden_size

    # MLA Attention per layer
    # W_DKV: hidden_size -> kv_latent_dim (down projection)
    # W_UK: kv_latent_dim -> num_kv_heads * nope_head_dim (K content up)
    # W_K_ROPE: kv_latent_dim -> num_kv_heads * rope_head_dim (K position up)
    # W_UV: kv_latent_dim -> num_kv_heads * head_dim (V up)
    # W_DQ: hidden_size -> q_latent_dim (Q down)
    # W_UQ: q_latent_dim -> num_heads * head_dim (Q up)
    # W_O: num_heads * head_dim -> hidden_size (output)
    attn_per_layer = (
        hidden_size * kv_latent_dim +                      # W_DKV
        kv_latent_dim * num_kv_heads * nope_head_dim +     # W_UK
        kv_latent_dim * num_kv_heads * rope_head_dim +     # W_K_ROPE
        kv_latent_dim * num_kv_heads * head_dim +          # W_UV
        hidden_size * q_latent_dim +                       # W_DQ
        q_latent_dim * num_attention_heads * head_dim +    # W_UQ
        num_attention_heads * head_dim * hidden_size       # W_O
    )
    attention = attn_per_layer * num_layers

    # MLP (same as standard)
    mlp_per_layer = 3 * hidden_size * intermediate_size
    mlp = mlp_per_layer * num_layers

    # Layer norms
    ln_per_layer = 2 * hidden_size
    layer_norm = ln_per_layer * num_layers + hidden_size

    # LM head
    lm_head = 0 if tie_embeddings else vocab_size * hidden_size

    total = embedding + attention + mlp + layer_norm + lm_head

    return ParameterBreakdown(
        embedding=embedding,
        attention=attention,
        mlp=mlp,
        layer_norm=layer_norm,
        lm_head=lm_head,
        lru=0,
        total=total,
    )


def count_mla_lru_params(
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_kv_heads: int,
    kv_latent_dim: int,
    q_latent_dim: int,
    rope_head_dim: int,
    lru_layers: Optional[int] = None,  # Number of layers with LRU
    use_positional_mixing: bool = True,
    use_global_halting: bool = True,
    tie_embeddings: bool = False,
) -> ParameterBreakdown:
    """Count parameters in MLA + LRU model.

    Args:
        lru_layers: Number of layers with LRU (None = all layers)
        use_positional_mixing: Whether LRU uses positional mixing
        use_global_halting: Whether LRU uses global halting

    Returns:
        ParameterBreakdown with detailed counts
    """
    # Start with MLA params
    mla_breakdown = count_mla_params(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        kv_latent_dim=kv_latent_dim,
        q_latent_dim=q_latent_dim,
        rope_head_dim=rope_head_dim,
        tie_embeddings=tie_embeddings,
    )

    # LRU parameters per layer
    # GRU gates: reset, update, candidate (each: 2*latent_dim -> latent_dim)
    lru_gru = 3 * (2 * kv_latent_dim * kv_latent_dim)

    # Halt projection: latent_dim -> 1
    lru_halt = kv_latent_dim + 1  # weight + bias

    # Layer norm: latent_dim
    lru_ln = kv_latent_dim

    lru_per_layer = lru_gru + lru_halt + lru_ln

    # Positional mixing (if enabled)
    if use_positional_mixing:
        # Depthwise conv: latent_dim weights * kernel_size(3)
        # Pointwise: latent_dim -> latent_dim
        # Gate: 2*latent_dim -> latent_dim
        pos_mix = (
            kv_latent_dim * 3 +                    # depthwise conv
            kv_latent_dim * kv_latent_dim +        # pointwise
            2 * kv_latent_dim * kv_latent_dim      # gate
        )
        lru_per_layer += pos_mix

    # Global halting (if enabled)
    if use_global_halting:
        # MLP: latent_dim -> latent_dim//4 -> 1
        global_halt = (
            kv_latent_dim * (kv_latent_dim // 4) +
            (kv_latent_dim // 4) + 1
        )
        lru_per_layer += global_halt

    # Total LRU params
    num_lru_layers = lru_layers if lru_layers is not None else num_layers
    lru_total = lru_per_layer * num_lru_layers

    return ParameterBreakdown(
        embedding=mla_breakdown.embedding,
        attention=mla_breakdown.attention,
        mlp=mla_breakdown.mlp,
        layer_norm=mla_breakdown.layer_norm,
        lm_head=mla_breakdown.lm_head,
        lru=lru_total,
        total=mla_breakdown.total + lru_total,
    )


def design_matched_baseline(
    target_params: int,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    num_kv_heads: int,
    tie_embeddings: bool = False,
) -> Tuple[int, ParameterBreakdown]:
    """Design a baseline model with matched parameter count.

    Adjusts the intermediate_size to match target parameter count.

    Args:
        target_params: Target total parameter count to match

    Returns:
        Tuple of (intermediate_size, ParameterBreakdown)
    """
    head_dim = hidden_size // num_attention_heads

    # Fixed params (embedding, attention, layer norm, lm_head)
    embedding = vocab_size * hidden_size
    attn_per_layer = (
        hidden_size * num_attention_heads * head_dim +
        hidden_size * num_kv_heads * head_dim +
        hidden_size * num_kv_heads * head_dim +
        num_attention_heads * head_dim * hidden_size
    )
    attention = attn_per_layer * num_layers
    layer_norm = (2 * hidden_size) * num_layers + hidden_size
    lm_head = 0 if tie_embeddings else vocab_size * hidden_size

    fixed = embedding + attention + layer_norm + lm_head

    # Solve for intermediate_size
    # MLP = 3 * hidden_size * intermediate_size * num_layers
    # target = fixed + MLP
    remaining = target_params - fixed
    intermediate_size = remaining // (3 * hidden_size * num_layers)

    # Round to nearest multiple of 64 for efficiency
    intermediate_size = (intermediate_size // 64) * 64

    # Verify
    breakdown = count_standard_transformer_params(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        tie_embeddings=tie_embeddings,
    )

    return intermediate_size, breakdown


def compare_configurations(
    configs: Dict[str, ParameterBreakdown],
    reference: str = None,
) -> str:
    """Generate comparison table for multiple configurations.

    Args:
        configs: Dict mapping config names to ParameterBreakdown
        reference: Name of reference config for relative comparison

    Returns:
        Formatted comparison string
    """
    lines = ["=" * 80]
    lines.append("Parameter Count Comparison")
    lines.append("=" * 80)

    # Header
    header = f"{'Config':<25} {'Total':>15} {'Attention':>12} {'MLP':>12} {'LRU':>10}"
    if reference:
        header += f" {'Δ vs {}'.format(reference[:8]):>10}"
    lines.append(header)
    lines.append("-" * 80)

    ref_total = configs[reference].total if reference else None

    for name, breakdown in configs.items():
        row = f"{name:<25} {breakdown.total:>15,} {breakdown.attention:>12,} {breakdown.mlp:>12,} {breakdown.lru:>10,}"
        if ref_total:
            delta = breakdown.total - ref_total
            delta_pct = 100 * delta / ref_total
            row += f" {delta_pct:>+9.1f}%"
        lines.append(row)

    lines.append("=" * 80)
    return "\n".join(lines)


# Preset configurations for Qwen2.5-0.5B based experiments
QWEN_0_5B_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_layers": 24,
    "num_attention_heads": 14,
    "num_kv_heads": 2,
}

MLA_LRU_CONFIG = {
    **QWEN_0_5B_CONFIG,
    "kv_latent_dim": 256,
    "q_latent_dim": 512,
    "rope_head_dim": 64,
}


if __name__ == "__main__":
    # Example: Compare configurations
    print("Computing parameter counts for different configurations...\n")

    # Standard Qwen-0.5B
    baseline = count_standard_transformer_params(**QWEN_0_5B_CONFIG)
    print("Baseline (Qwen2.5-0.5B):")
    print(baseline)
    print()

    # MLA only
    mla_only = count_mla_params(**MLA_LRU_CONFIG)
    print("MLA Only:")
    print(mla_only)
    print()

    # MLA + LRU
    mla_lru = count_mla_lru_params(**MLA_LRU_CONFIG)
    print("MLA + LRU (full):")
    print(mla_lru)
    print()

    # Design matched baseline
    target = mla_lru.total
    matched_intermediate, matched_baseline = design_matched_baseline(
        target_params=target,
        **{k: v for k, v in QWEN_0_5B_CONFIG.items() if k != "intermediate_size"}
    )
    print(f"Parameter-Matched Baseline (intermediate_size={matched_intermediate}):")
    print(matched_baseline)
    print()

    # Comparison table
    configs = {
        "baseline_original": baseline,
        "baseline_matched": matched_baseline,
        "mla_only": mla_only,
        "mla_lru_full": mla_lru,
    }
    print(compare_configurations(configs, reference="mla_lru_full"))
