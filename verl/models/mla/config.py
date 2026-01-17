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
Configuration classes for Multi-head Latent Attention (MLA) and
Latent Reasoning Unit (LRU) modules.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class MLAConfig:
    """Configuration for Multi-head Latent Attention.

    Based on DeepSeek-V3's MLA architecture with decoupled RoPE.

    Attributes:
        hidden_size: Model hidden dimension (H)
        num_attention_heads: Number of attention heads (n_heads)
        num_kv_heads: Number of key-value heads for GQA (n_kv)
        kv_latent_dim: Latent dimension for KV compression (d_c)
        q_latent_dim: Latent dimension for Q compression (d_c')
        rope_head_dim: Dimension for RoPE encoding (d_rope)
        head_dim: Per-head dimension (head_dim = H / n_heads typically)
        max_position_embeddings: Maximum sequence length
        rope_theta: Base for RoPE frequency computation
        rope_scaling: RoPE scaling configuration
        attention_dropout: Dropout rate for attention weights
        attention_bias: Whether to use bias in attention projections
    """
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_kv_heads: int = 8
    kv_latent_dim: int = 512  # d_c: KV compression latent dimension
    q_latent_dim: int = 1536  # d_c': Q compression latent dimension
    rope_head_dim: int = 64   # d_rope: dimension for RoPE (decoupled)
    head_dim: int = 128       # per-head dimension
    max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: Optional[dict] = None
    attention_dropout: float = 0.0
    attention_bias: bool = False

    @property
    def kv_projection_size(self) -> int:
        """Size of the up-projected KV."""
        return self.num_kv_heads * self.head_dim

    @property
    def q_projection_size(self) -> int:
        """Size of the up-projected Q."""
        return self.num_attention_heads * self.head_dim

    @property
    def nope_head_dim(self) -> int:
        """Dimension for non-RoPE part of each head."""
        return self.head_dim - self.rope_head_dim


@dataclass
class LRUConfig:
    """Configuration for Latent Reasoning Unit.

    The LRU performs iterative refinement in the latent space using
    GRU-style gating with Adaptive Computation Time (ACT) halting.

    Attributes:
        latent_dim: Dimension of the latent space (matches kv_latent_dim)
        max_iterations: Maximum number of refinement iterations
        halt_threshold: Cumulative probability threshold for stopping
        init_halt_bias: Initial bias for halt probability (negative = more iterations)
        gate_activation: Activation for gates ('sigmoid' or 'hard_sigmoid')
        candidate_activation: Activation for candidate state ('tanh' or 'gelu')
        use_layer_norm: Whether to apply LayerNorm after each iteration
        gradient_checkpointing: Whether to use gradient checkpointing for memory
        positional_mixing_type: Type of cross-position interaction ('conv' or 'attention')
        use_enhanced_global_halting: Whether to use enhanced halting with learnable weights
        use_learnable_loss_weights: Whether to use learnable loss weights
        ponder_loss_weight: Weight for the pondering cost loss
        stability_loss_weight: Weight for the stability loss
        sparsity_loss_weight: Weight for the sparsity loss
    """
    latent_dim: int = 512
    max_iterations: int = 8
    halt_threshold: float = 0.99
    init_halt_bias: float = -2.0  # Start with low halt probability
    gate_activation: Literal['sigmoid', 'hard_sigmoid'] = 'sigmoid'
    candidate_activation: Literal['tanh', 'gelu'] = 'tanh'
    use_layer_norm: bool = True
    gradient_checkpointing: bool = False

    # NEW: Positional mixing configuration
    # 'conv': Local 3-token window convolution (original)
    # 'attention': O(n) linear attention for global cross-position interaction
    positional_mixing_type: Literal['conv', 'attention'] = 'conv'

    # NEW: Enhanced global halting with learnable weights and attention pooling
    use_enhanced_global_halting: bool = False

    # NEW: Learnable loss weights (addresses "magic number" criticism)
    use_learnable_loss_weights: bool = False

    # Loss weights (can be overridden in trainer config)
    # If use_learnable_loss_weights=True, these become initial values
    ponder_loss_weight: float = 0.001
    stability_loss_weight: float = 0.1
    sparsity_loss_weight: float = 0.01


@dataclass
class DeepSeekMLAConfig:
    """Combined configuration for DeepSeek-style MLA + LRU model.

    This configuration combines MLA attention with optional LRU
    for latent space reasoning.
    """
    mla: MLAConfig = field(default_factory=MLAConfig)
    lru: LRUConfig = field(default_factory=LRUConfig)

    # Model-level settings
    num_hidden_layers: int = 24
    intermediate_size: int = 5632  # FFN intermediate size
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    tie_word_embeddings: bool = False
    vocab_size: int = 32000

    # LRU integration
    use_lru: bool = True
    lru_layers: Optional[list] = None  # Which layers to apply LRU, None = all

    @classmethod
    def from_qwen2_config(cls, qwen2_config, lru_config: Optional[LRUConfig] = None):
        """Create DeepSeekMLAConfig from Qwen2Config for architecture adaptation."""
        mla_config = MLAConfig(
            hidden_size=qwen2_config.hidden_size,
            num_attention_heads=qwen2_config.num_attention_heads,
            num_kv_heads=getattr(qwen2_config, 'num_key_value_heads', qwen2_config.num_attention_heads),
            # Derive latent dims based on compression ratio
            kv_latent_dim=qwen2_config.hidden_size // 4,
            q_latent_dim=qwen2_config.hidden_size * 3 // 4,
            rope_head_dim=64,
            head_dim=qwen2_config.hidden_size // qwen2_config.num_attention_heads,
            max_position_embeddings=qwen2_config.max_position_embeddings,
            rope_theta=getattr(qwen2_config, 'rope_theta', 10000.0),
            rope_scaling=getattr(qwen2_config, 'rope_scaling', None),
            attention_dropout=getattr(qwen2_config, 'attention_dropout', 0.0),
        )

        lru_cfg = lru_config or LRUConfig(latent_dim=mla_config.kv_latent_dim)

        return cls(
            mla=mla_config,
            lru=lru_cfg,
            num_hidden_layers=qwen2_config.num_hidden_layers,
            intermediate_size=qwen2_config.intermediate_size,
            hidden_act=getattr(qwen2_config, 'hidden_act', 'silu'),
            rms_norm_eps=getattr(qwen2_config, 'rms_norm_eps', 1e-6),
            initializer_range=getattr(qwen2_config, 'initializer_range', 0.02),
            use_cache=getattr(qwen2_config, 'use_cache', True),
            tie_word_embeddings=getattr(qwen2_config, 'tie_word_embeddings', False),
            vocab_size=qwen2_config.vocab_size,
        )


# Predefined configurations for different model sizes
QWEN_0_5B_MLA_CONFIG = DeepSeekMLAConfig(
    mla=MLAConfig(
        hidden_size=896,
        num_attention_heads=14,
        num_kv_heads=2,
        kv_latent_dim=256,      # ~4x compression
        q_latent_dim=512,
        rope_head_dim=64,
        head_dim=64,
        max_position_embeddings=32768,
    ),
    lru=LRUConfig(
        latent_dim=256,
        max_iterations=8,
        halt_threshold=0.99,
    ),
    num_hidden_layers=24,
    intermediate_size=4864,
    vocab_size=151936,
)

QWEN_1_5B_MLA_CONFIG = DeepSeekMLAConfig(
    mla=MLAConfig(
        hidden_size=1536,
        num_attention_heads=12,
        num_kv_heads=2,
        kv_latent_dim=384,
        q_latent_dim=1024,
        rope_head_dim=64,
        head_dim=128,
        max_position_embeddings=32768,
    ),
    lru=LRUConfig(
        latent_dim=384,
        max_iterations=8,
        halt_threshold=0.99,
    ),
    num_hidden_layers=28,
    intermediate_size=8960,
    vocab_size=151936,
)
