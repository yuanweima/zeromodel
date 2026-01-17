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
"""Tests for parameter counting utilities."""

import pytest
import sys
from pathlib import Path

# Add utils path to allow direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "verl" / "utils"))

from param_counter import (
    count_standard_transformer_params,
    count_mla_params,
    count_mla_lru_params,
    design_matched_baseline,
    compare_configurations,
    ParameterBreakdown,
    QWEN_0_5B_CONFIG,
    MLA_LRU_CONFIG,
)


class TestParameterBreakdown:
    """Tests for ParameterBreakdown dataclass."""

    def test_breakdown_creation(self):
        """Test creating a breakdown object."""
        breakdown = ParameterBreakdown(
            embedding=100,
            attention=200,
            mlp=300,
            layer_norm=50,
            lm_head=100,
            lru=50,
            total=800,
        )
        assert breakdown.total == 800
        assert breakdown.embedding == 100

    def test_breakdown_str(self):
        """Test string representation."""
        breakdown = ParameterBreakdown(
            embedding=1000000,
            attention=2000000,
            mlp=3000000,
            layer_norm=100000,
            lm_head=1000000,
            lru=500000,
            total=7600000,
        )
        s = str(breakdown)
        assert "Embedding" in s
        assert "Attention" in s
        assert "MLP" in s
        assert "LRU" in s
        assert "Total" in s


class TestStandardTransformer:
    """Tests for standard transformer parameter counting."""

    def test_basic_count(self):
        """Test basic parameter counting."""
        breakdown = count_standard_transformer_params(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
        )

        assert breakdown.total > 0
        assert breakdown.embedding == 1000 * 256  # vocab * hidden
        assert breakdown.lru == 0  # No LRU in standard transformer

    def test_tied_embeddings(self):
        """Test with tied embeddings."""
        untied = count_standard_transformer_params(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            tie_embeddings=False,
        )

        tied = count_standard_transformer_params(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_attention_heads=4,
            num_kv_heads=4,
            tie_embeddings=True,
        )

        # Tied should have fewer params (no separate lm_head)
        assert tied.total < untied.total
        assert tied.lm_head == 0
        assert untied.lm_head == 1000 * 256

    def test_gqa(self):
        """Test Grouped Query Attention (fewer KV heads)."""
        full_attn = count_standard_transformer_params(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_attention_heads=8,
            num_kv_heads=8,  # Full attention
        )

        gqa = count_standard_transformer_params(
            vocab_size=1000,
            hidden_size=256,
            intermediate_size=512,
            num_layers=2,
            num_attention_heads=8,
            num_kv_heads=2,  # GQA with 2 KV heads
        )

        # GQA should have fewer attention params
        assert gqa.attention < full_attn.attention

    def test_qwen_config(self):
        """Test with Qwen-0.5B config."""
        breakdown = count_standard_transformer_params(**QWEN_0_5B_CONFIG)

        # Should be around 630M params
        assert 600_000_000 < breakdown.total < 700_000_000


class TestMLAParams:
    """Tests for MLA parameter counting."""

    def test_mla_vs_standard(self):
        """Test MLA parameter count vs standard attention."""
        standard = count_standard_transformer_params(**QWEN_0_5B_CONFIG)
        mla = count_mla_params(**MLA_LRU_CONFIG)

        # MLA should have different attention params due to latent compression
        assert mla.attention != standard.attention
        # But similar total (within 10%)
        ratio = mla.total / standard.total
        assert 0.9 < ratio < 1.1

    def test_latent_dim_effect(self):
        """Test effect of latent dimension on param count."""
        small_latent = count_mla_params(
            **{**MLA_LRU_CONFIG, 'kv_latent_dim': 128, 'q_latent_dim': 256}
        )
        large_latent = count_mla_params(
            **{**MLA_LRU_CONFIG, 'kv_latent_dim': 512, 'q_latent_dim': 1024}
        )

        # Larger latent should have more attention params
        assert large_latent.attention > small_latent.attention


class TestMLALRUParams:
    """Tests for MLA + LRU parameter counting."""

    def test_lru_adds_params(self):
        """Test that LRU adds parameters."""
        mla_only = count_mla_params(**MLA_LRU_CONFIG)
        mla_lru = count_mla_lru_params(**MLA_LRU_CONFIG)

        assert mla_lru.lru > 0
        assert mla_lru.total > mla_only.total

    def test_positional_mixing_effect(self):
        """Test positional mixing parameter contribution."""
        with_pos_mix = count_mla_lru_params(
            **MLA_LRU_CONFIG,
            use_positional_mixing=True,
        )
        without_pos_mix = count_mla_lru_params(
            **MLA_LRU_CONFIG,
            use_positional_mixing=False,
        )

        assert with_pos_mix.lru > without_pos_mix.lru

    def test_global_halting_effect(self):
        """Test global halting parameter contribution."""
        with_global = count_mla_lru_params(
            **MLA_LRU_CONFIG,
            use_global_halting=True,
        )
        without_global = count_mla_lru_params(
            **MLA_LRU_CONFIG,
            use_global_halting=False,
        )

        assert with_global.lru > without_global.lru

    def test_partial_layers(self):
        """Test LRU on subset of layers."""
        full_lru = count_mla_lru_params(**MLA_LRU_CONFIG, lru_layers=None)
        partial_lru = count_mla_lru_params(**MLA_LRU_CONFIG, lru_layers=12)

        assert partial_lru.lru < full_lru.lru
        # Partial should be ~half the LRU params (12/24 layers)
        ratio = partial_lru.lru / full_lru.lru
        assert 0.4 < ratio < 0.6


class TestDesignMatchedBaseline:
    """Tests for matched baseline design."""

    def test_matches_target(self):
        """Test that designed baseline matches target param count."""
        target = count_mla_lru_params(**MLA_LRU_CONFIG).total

        intermediate_size, breakdown = design_matched_baseline(
            target_params=target,
            **{k: v for k, v in QWEN_0_5B_CONFIG.items() if k != 'intermediate_size'}
        )

        # Should be within 1% of target
        ratio = breakdown.total / target
        assert 0.99 < ratio < 1.01

    def test_intermediate_size_multiple_of_64(self):
        """Test that intermediate_size is multiple of 64."""
        target = 700_000_000

        intermediate_size, _ = design_matched_baseline(
            target_params=target,
            **{k: v for k, v in QWEN_0_5B_CONFIG.items() if k != 'intermediate_size'}
        )

        assert intermediate_size % 64 == 0


class TestCompareConfigurations:
    """Tests for configuration comparison."""

    def test_basic_comparison(self):
        """Test basic comparison table generation."""
        configs = {
            'baseline': count_standard_transformer_params(**QWEN_0_5B_CONFIG),
            'mla_only': count_mla_params(**MLA_LRU_CONFIG),
            'mla_lru': count_mla_lru_params(**MLA_LRU_CONFIG),
        }

        result = compare_configurations(configs)

        assert 'baseline' in result
        assert 'mla_only' in result
        assert 'mla_lru' in result

    def test_with_reference(self):
        """Test comparison with reference config."""
        configs = {
            'baseline': count_standard_transformer_params(**QWEN_0_5B_CONFIG),
            'mla_lru': count_mla_lru_params(**MLA_LRU_CONFIG),
        }

        result = compare_configurations(configs, reference='baseline')

        # Should show delta percentages
        assert '%' in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
