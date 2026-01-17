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
Multi-head Latent Attention (MLA) with Latent Reasoning Unit (LRU).

This module implements the core architecture for ZeroModel:
- MLA: Attention with compressed latent space for KV
- LRU: Iterative reasoning in latent space with ACT halting
- Decoupled RoPE: Position encoding only on part of head dimensions
"""

from .config import (
    MLAConfig,
    LRUConfig,
    DeepSeekMLAConfig,
    QWEN_0_5B_MLA_CONFIG,
    QWEN_1_5B_MLA_CONFIG,
)
from .rope import (
    DecoupledRotaryEmbedding,
    YaRNScaledRotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_decoupled,
    rotate_half,
)
from .mla_attention import (
    MLAAttention,
    MLAFlashAttention2,
)
from .lru import (
    LatentReasoningUnit,
    SimpleLRU,
    LRUOutput,
    PositionalMixing,
    GlobalHaltingUnit,
    UniversalTransformerLRU,
)
from .modeling_deepseek_mla import (
    DeepSeekMLAPretrainedConfig,
    DeepSeekMLAModel,
    DeepSeekMLAForCausalLM,
)

__all__ = [
    # Config
    'MLAConfig',
    'LRUConfig',
    'DeepSeekMLAConfig',
    'QWEN_0_5B_MLA_CONFIG',
    'QWEN_1_5B_MLA_CONFIG',
    # RoPE
    'DecoupledRotaryEmbedding',
    'YaRNScaledRotaryEmbedding',
    'apply_rotary_pos_emb',
    'apply_rotary_pos_emb_decoupled',
    'rotate_half',
    # Attention
    'MLAAttention',
    'MLAFlashAttention2',
    # LRU
    'LatentReasoningUnit',
    'SimpleLRU',
    'LRUOutput',
    # Models
    'DeepSeekMLAPretrainedConfig',
    'DeepSeekMLAModel',
    'DeepSeekMLAForCausalLM',
]
