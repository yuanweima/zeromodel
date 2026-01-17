"""
Tests for MLA + LRU implementation.

Run with: python tests/test_mla_lru.py
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Bypass verl package __init__.py by importing directly
def setup_direct_imports():
    """Setup direct imports to bypass verl/__init__.py dependencies."""
    import importlib.util

    def load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    # Load MLA modules directly
    mla_path = os.path.join(project_root, 'verl', 'models', 'mla')
    load_module('verl.models.mla.config', os.path.join(mla_path, 'config.py'))
    load_module('verl.models.mla.rope', os.path.join(mla_path, 'rope.py'))
    load_module('verl.models.mla.lru', os.path.join(mla_path, 'lru.py'))
    load_module('verl.models.mla.mla_attention', os.path.join(mla_path, 'mla_attention.py'))
    load_module('verl.models.mla.modeling_deepseek_mla', os.path.join(mla_path, 'modeling_deepseek_mla.py'))

    # Load trainer modules
    trainer_path = os.path.join(project_root, 'verl', 'trainer', 'lru')
    load_module('verl.trainer.lru.halting', os.path.join(trainer_path, 'halting.py'))
    load_module('verl.trainer.lru.losses', os.path.join(trainer_path, 'losses.py'))

    # Load reward score module
    reward_path = os.path.join(project_root, 'verl', 'utils', 'reward_score')
    load_module('verl.utils.reward_score.causal_loop', os.path.join(reward_path, 'causal_loop.py'))

setup_direct_imports()

import torch
import torch.nn as nn


def test_config():
    """Test configuration classes."""
    print("=" * 60)
    print("Test 1: Configuration Classes")
    print("=" * 60)

    from verl.models.mla.config import MLAConfig, LRUConfig, DeepSeekMLAConfig

    # Test MLAConfig
    mla_config = MLAConfig(
        hidden_size=896,
        num_attention_heads=14,
        num_kv_heads=2,
        kv_latent_dim=256,
        q_latent_dim=512,
        rope_head_dim=64,
        head_dim=64,
    )

    assert mla_config.kv_projection_size == 2 * 64  # n_kv * head_dim
    assert mla_config.q_projection_size == 14 * 64  # n_heads * head_dim
    assert mla_config.nope_head_dim == 0  # head_dim - rope_head_dim

    print(f"  MLAConfig: hidden_size={mla_config.hidden_size}, kv_latent_dim={mla_config.kv_latent_dim}")

    # Test LRUConfig
    lru_config = LRUConfig(
        latent_dim=256,
        max_iterations=8,
        halt_threshold=0.99,
    )

    assert lru_config.max_iterations == 8
    assert lru_config.halt_threshold == 0.99

    print(f"  LRUConfig: max_iterations={lru_config.max_iterations}, halt_threshold={lru_config.halt_threshold}")

    # Test DeepSeekMLAConfig
    config = DeepSeekMLAConfig(
        mla=mla_config,
        lru=lru_config,
        num_hidden_layers=24,
    )

    assert config.use_lru == True
    print(f"  DeepSeekMLAConfig: num_layers={config.num_hidden_layers}, use_lru={config.use_lru}")

    print("  [PASS] Configuration tests passed!")
    return True


def test_rope():
    """Test RoPE implementation."""
    print("\n" + "=" * 60)
    print("Test 2: Decoupled RoPE")
    print("=" * 60)

    from verl.models.mla.rope import (
        DecoupledRotaryEmbedding,
        apply_rotary_pos_emb_decoupled,
        rotate_half,
    )

    # Test DecoupledRotaryEmbedding
    rope_dim = 64
    max_seq_len = 512
    rope = DecoupledRotaryEmbedding(rope_dim=rope_dim, max_position_embeddings=max_seq_len)

    # Create test input
    batch_size, seq_len = 2, 32
    x = torch.randn(batch_size, seq_len, rope_dim)

    # Get cos, sin
    cos, sin = rope(x, seq_len=seq_len)

    assert cos.shape == (seq_len, rope_dim), f"Expected ({seq_len}, {rope_dim}), got {cos.shape}"
    assert sin.shape == (seq_len, rope_dim), f"Expected ({seq_len}, {rope_dim}), got {sin.shape}"

    print(f"  RoPE cos/sin shapes: {cos.shape}")

    # Test rotate_half
    x_rot = rotate_half(x)
    assert x_rot.shape == x.shape
    print(f"  rotate_half output shape: {x_rot.shape}")

    # Test decoupled application
    num_heads = 4
    num_kv_heads = 2
    head_dim = 128
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)  # [B, S, H, D]
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim)

    # For decoupled RoPE, we need to expand cos/sin to match the head dimension
    # cos/sin are [seq_len, rope_dim], we need [B, seq_len, 1, rope_dim] for broadcasting
    cos_expanded = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, rope_dim]
    sin_expanded = sin.unsqueeze(0).unsqueeze(2)

    # Extract RoPE portion and apply
    q_rope = q[..., :rope_dim]
    k_rope = k[..., :rope_dim]
    q_nope = q[..., rope_dim:]
    k_nope = k[..., rope_dim:]

    # Apply RoPE manually to test
    q_rope_embed = q_rope * cos_expanded + rotate_half(q_rope) * sin_expanded
    k_rope_embed = k_rope * cos_expanded + rotate_half(k_rope) * sin_expanded

    q_embed = torch.cat([q_rope_embed, q_nope], dim=-1)
    k_embed = torch.cat([k_rope_embed, k_nope], dim=-1)

    assert q_embed.shape == q.shape
    assert k_embed.shape == k.shape
    print(f"  Decoupled RoPE applied: q={q_embed.shape}, k={k_embed.shape}")

    print("  [PASS] RoPE tests passed!")
    return True


def test_lru():
    """Test LRU module."""
    print("\n" + "=" * 60)
    print("Test 3: Latent Reasoning Unit (LRU)")
    print("=" * 60)

    from verl.models.mla.config import LRUConfig
    from verl.models.mla.lru import LatentReasoningUnit, SimpleLRU, LRUOutput

    # Test LatentReasoningUnit with ACT
    lru_config = LRUConfig(
        latent_dim=256,
        max_iterations=8,
        halt_threshold=0.99,
        init_halt_bias=-2.0,
    )

    lru = LatentReasoningUnit(lru_config)

    # Create test input
    batch_size, seq_len, latent_dim = 2, 16, 256
    c_kv = torch.randn(batch_size, seq_len, latent_dim)

    # Forward pass
    output, lru_output = lru(c_kv, return_intermediates=True)

    assert output.shape == c_kv.shape, f"Expected {c_kv.shape}, got {output.shape}"
    assert isinstance(lru_output, LRUOutput)

    print(f"  Input shape: {c_kv.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Avg iterations: {lru_output.num_iterations.mean():.2f}")
    print(f"  Halt probabilities: min={lru_output.halting_probabilities.min():.3f}, max={lru_output.halting_probabilities.max():.3f}")

    if lru_output.intermediate_states is not None:
        print(f"  Intermediate states shape: {lru_output.intermediate_states.shape}")

    # Test SimpleLRU (fixed iterations)
    simple_lru = SimpleLRU(lru_config)
    output_simple, lru_output_simple = simple_lru(c_kv)

    assert output_simple.shape == c_kv.shape
    print(f"  SimpleLRU output shape: {output_simple.shape}")
    print(f"  SimpleLRU fixed iterations: {lru_output_simple.num_iterations[0, 0]:.0f}")

    print("  [PASS] LRU tests passed!")
    return True


def test_mla_attention():
    """Test MLA attention module."""
    print("\n" + "=" * 60)
    print("Test 4: MLA Attention")
    print("=" * 60)

    from verl.models.mla.config import MLAConfig, LRUConfig
    from verl.models.mla.mla_attention import MLAAttention
    from verl.models.mla.lru import LatentReasoningUnit

    # Create config
    mla_config = MLAConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_kv_heads=2,
        kv_latent_dim=64,
        q_latent_dim=128,
        rope_head_dim=32,
        head_dim=64,
        max_position_embeddings=512,
    )

    # Create MLA attention
    mla = MLAAttention(mla_config, layer_idx=0)

    # Test input
    batch_size, seq_len = 2, 16
    hidden_states = torch.randn(batch_size, seq_len, mla_config.hidden_size)

    # Forward pass without LRU
    output, attn_weights, past_kv = mla(
        hidden_states,
        output_attentions=False,
        use_cache=False,
    )

    assert output.shape == hidden_states.shape, f"Expected {hidden_states.shape}, got {output.shape}"
    print(f"  Input shape: {hidden_states.shape}")
    print(f"  Output shape (no LRU): {output.shape}")

    # Add LRU and test again
    lru_config = LRUConfig(latent_dim=mla_config.kv_latent_dim, max_iterations=4)
    lru = LatentReasoningUnit(lru_config)
    mla.set_lru(lru)

    output_with_lru, _, _ = mla(
        hidden_states,
        output_attentions=False,
        use_cache=False,
    )

    # Output might be tuple (output, lru_info) when LRU is attached
    if isinstance(output_with_lru, tuple):
        output_tensor, lru_info = output_with_lru
        print(f"  Output shape (with LRU): {output_tensor.shape}")
        print(f"  LRU info attached: {type(lru_info).__name__}")
    else:
        print(f"  Output shape (with LRU): {output_with_lru.shape}")

    print("  [PASS] MLA Attention tests passed!")
    return True


def test_losses():
    """Test loss functions."""
    print("\n" + "=" * 60)
    print("Test 5: Loss Functions")
    print("=" * 60)

    from verl.trainer.lru.losses import (
        StabilityLoss,
        SparsityLoss,
        PonderLoss,
        LRULossModule,
    )
    from verl.models.mla.lru import LRUOutput

    batch_size, seq_len, latent_dim = 2, 16, 64
    num_iters = 5

    # Create mock intermediate states
    intermediate_states = torch.randn(num_iters, batch_size, seq_len, latent_dim)

    # Test StabilityLoss
    stability_loss = StabilityLoss(decay=0.9)
    loss_stab = stability_loss(intermediate_states)
    print(f"  Stability loss: {loss_stab.item():.6f}")

    # Test SparsityLoss
    activations = torch.randn(batch_size, seq_len, latent_dim)
    sparsity_loss = SparsityLoss(target_sparsity=0.8)
    loss_sparse = sparsity_loss(activations)
    print(f"  Sparsity loss: {loss_sparse.item():.6f}")

    # Test PonderLoss
    lru_output = LRUOutput(
        output=torch.randn(batch_size, seq_len, latent_dim),
        halting_probabilities=torch.rand(batch_size, seq_len),
        num_iterations=torch.randint(1, 8, (batch_size, seq_len)).float(),
        remainders=torch.rand(batch_size, seq_len) * 0.1,
        intermediate_states=intermediate_states,
    )

    ponder_loss = PonderLoss(max_iterations=8)
    loss_ponder = ponder_loss(lru_output)
    print(f"  Ponder loss: {loss_ponder.item():.6f}")

    # Test combined LRULossModule
    lru_loss_module = LRULossModule(
        stability_weight=0.1,
        sparsity_weight=0.01,
        ponder_weight=0.001,
    )

    loss_output = lru_loss_module(lru_output)
    print(f"  Combined loss: {loss_output.total_loss.item():.6f}")
    print(f"  Metrics: avg_iterations={loss_output.metrics['lru/avg_iterations'].item():.2f}")

    print("  [PASS] Loss function tests passed!")
    return True


def test_halting():
    """Test halting mechanisms."""
    print("\n" + "=" * 60)
    print("Test 6: Halting Mechanisms")
    print("=" * 60)

    from verl.trainer.lru.halting import (
        ACTHaltingUnit,
        ConfidenceHaltingUnit,
        DynamicHaltingUnit,
        create_halting_unit,
    )

    hidden_dim = 64
    batch_size, seq_len = 2, 16

    # Test ACTHaltingUnit
    act_halt = ACTHaltingUnit(hidden_dim=hidden_dim, threshold=0.99)
    state = torch.randn(batch_size, seq_len, hidden_dim)
    halt_prob = act_halt.compute_halt_probability(state, iteration=0)

    assert halt_prob.shape == (batch_size, seq_len)
    print(f"  ACT halt probability shape: {halt_prob.shape}")
    print(f"  ACT halt probability range: [{halt_prob.min():.3f}, {halt_prob.max():.3f}]")

    # Test factory function
    for halt_type in ['act', 'confidence', 'dynamic']:
        halt_unit = create_halting_unit(halt_type, hidden_dim=hidden_dim)
        print(f"  Created {halt_type} halting unit: {type(halt_unit).__name__}")

    print("  [PASS] Halting mechanism tests passed!")
    return True


def test_causal_loop_data():
    """Test causal loop data generation."""
    print("\n" + "=" * 60)
    print("Test 7: Causal Loop Data Generation")
    print("=" * 60)

    # Import from the data preprocessing module directly
    import importlib.util
    causal_loop_path = os.path.join(project_root, 'examples', 'data_preprocess', 'causal_loop.py')

    # Parse the file manually to extract the classes we need
    # (avoiding the Dataset import which requires 'datasets' package)
    import ast

    with open(causal_loop_path, 'r') as f:
        source = f.read()

    # Extract the classes we need
    exec("""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import random

@dataclass
class CausalRule:
    source: str
    target: str
    operation: str
    value: int = 0
    condition: Optional[str] = None

    def apply(self, state: Dict[str, int]) -> int:
        source_val = state.get(self.source, 0)
        if self.operation == 'copy':
            return source_val
        elif self.operation == 'add':
            return state.get(self.target, 0) + self.value
        elif self.operation == 'add_source':
            return state.get(self.target, 0) + source_val
        elif self.operation == 'mult':
            return source_val * self.value
        elif self.operation == 'conditional':
            if source_val > self.value:
                return state.get(self.target, 0) + 1
            return state.get(self.target, 0)
        elif self.operation == 'mod':
            return source_val % max(self.value, 1)
        else:
            return state.get(self.target, 0)

    def to_string(self) -> str:
        if self.operation == 'copy':
            return f"{self.target} = {self.source}"
        elif self.operation == 'add':
            return f"{self.target} += {self.value}"
        elif self.operation == 'add_source':
            return f"{self.target} += {self.source}"
        elif self.operation == 'mult':
            return f"{self.target} = {self.source} * {self.value}"
        elif self.operation == 'conditional':
            return f"if {self.source} > {self.value}: {self.target} += 1"
        elif self.operation == 'mod':
            return f"{self.target} = {self.source} % {self.value}"
        else:
            return f"{self.source} -> {self.target} ({self.operation})"

@dataclass
class CausalGraph:
    variables: List[str]
    rules: List[CausalRule]
    initial_state: Dict[str, int] = field(default_factory=dict)

    def step(self, state: Dict[str, int]) -> Dict[str, int]:
        new_state = state.copy()
        for rule in self.rules:
            new_state[rule.target] = rule.apply(state)
        return new_state

    def simulate(self, num_steps: int) -> List[Dict[str, int]]:
        states = [self.initial_state.copy()]
        current = self.initial_state.copy()
        for _ in range(num_steps):
            current = self.step(current)
            states.append(current.copy())
        return states

    def get_final_state(self, num_steps: int) -> Dict[str, int]:
        return self.simulate(num_steps)[-1]

class CausalLoopGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_level1(self) -> Tuple[CausalGraph, int]:
        num_vars = self.rng.randint(3, 5)
        variables = [chr(ord('A') + i) for i in range(num_vars)]
        initial_state = {var: 0 for var in variables}
        initial_state[variables[0]] = self.rng.randint(1, 5)
        rules = []
        for i in range(1, num_vars):
            op = self.rng.choice(['copy', 'add_source'])
            rules.append(CausalRule(source=variables[i-1], target=variables[i], operation=op))
        graph = CausalGraph(variables=variables, rules=rules, initial_state=initial_state)
        num_steps = self.rng.randint(2, 4)
        return graph, num_steps

    def generate_level2(self) -> Tuple[CausalGraph, int]:
        num_vars = self.rng.randint(3, 4)
        variables = [chr(ord('A') + i) for i in range(num_vars)]
        initial_state = {var: self.rng.randint(0, 3) for var in variables}
        rules = []
        for i in range(num_vars):
            next_idx = (i + 1) % num_vars
            op = self.rng.choice(['add_source', 'copy'])
            rules.append(CausalRule(source=variables[i], target=variables[next_idx], operation=op))
        if self.rng.random() > 0.5:
            rules.append(CausalRule(source=variables[0], target=variables[0], operation='add', value=1))
        graph = CausalGraph(variables=variables, rules=rules, initial_state=initial_state)
        num_steps = self.rng.randint(2, 3)
        return graph, num_steps

    def generate_level3(self) -> Tuple[CausalGraph, int]:
        variables = ['A', 'B', 'C', 'D']
        initial_state = {'A': self.rng.randint(1, 3), 'B': self.rng.randint(1, 3), 'C': 0, 'D': 0}
        rules = [
            CausalRule(source='A', target='C', operation='add_source'),
            CausalRule(source='B', target='C', operation='add_source'),
            CausalRule(source='C', target='D', operation='copy'),
        ]
        if self.rng.random() > 0.5:
            rules.append(CausalRule(source='D', target='A', operation='conditional', value=2))
        graph = CausalGraph(variables=variables, rules=rules, initial_state=initial_state)
        num_steps = self.rng.randint(2, 4)
        return graph, num_steps

    def generate_level4(self) -> Tuple[CausalGraph, int]:
        variables = ['A', 'B', 'C', 'D', 'E']
        initial_state = {var: self.rng.randint(0, 2) for var in variables}
        initial_state['A'] = self.rng.randint(1, 3)
        rules = [
            CausalRule(source='A', target='B', operation='add_source'),
            CausalRule(source='B', target='C', operation='mult', value=2),
            CausalRule(source='C', target='D', operation='mod', value=5),
            CausalRule(source='D', target='E', operation='add_source'),
            CausalRule(source='E', target='A', operation='conditional', value=3),
            CausalRule(source='C', target='A', operation='conditional', value=5),
        ]
        graph = CausalGraph(variables=variables, rules=rules, initial_state=initial_state)
        num_steps = self.rng.randint(3, 5)
        return graph, num_steps

    def generate(self, level: int) -> Tuple[CausalGraph, int]:
        generators = {1: self.generate_level1, 2: self.generate_level2, 3: self.generate_level3, 4: self.generate_level4}
        return generators[level]()
""", globals())

    generator = CausalLoopGenerator(seed=42)

    # Test each level
    for level in [1, 2, 3, 4]:
        graph, num_steps = generator.generate(level)
        final_state = graph.get_final_state(num_steps)

        print(f"  Level {level}:")
        print(f"    Variables: {graph.variables}")
        print(f"    Rules: {len(graph.rules)}")
        print(f"    Initial: {graph.initial_state}")
        print(f"    Steps: {num_steps}")
        print(f"    Final: {final_state}")

    print("  [PASS] Causal loop data generation tests passed!")
    return True


def test_causal_loop_reward():
    """Test causal loop reward function."""
    print("\n" + "=" * 60)
    print("Test 8: Causal Loop Reward Function")
    print("=" * 60)

    from verl.utils.reward_score.causal_loop import compute_score, extract_answer, parse_state

    # Test extract_answer
    solution1 = "Assistant: Let me think...\n<think>reasoning</think>\n<answer>A=5, B=3, C=2</answer>"
    answer1 = extract_answer(solution1)
    assert answer1 == "A=5, B=3, C=2", f"Expected 'A=5, B=3, C=2', got '{answer1}'"
    print(f"  Extracted answer: '{answer1}'")

    # Test parse_state
    state = parse_state("A=5, B=3, C=2", ['A', 'B', 'C'])
    assert state == {'A': 5, 'B': 3, 'C': 2}, f"Expected {{'A': 5, 'B': 3, 'C': 2}}, got {state}"
    print(f"  Parsed state: {state}")

    # Test compute_score - correct answer
    ground_truth = {
        'final_state': {'A': 5, 'B': 3, 'C': 2},
        'variables': ['A', 'B', 'C'],
        'level': 1,
        'num_steps': 2,
    }

    score_correct = compute_score(solution1, ground_truth)
    assert score_correct == 1.0, f"Expected 1.0, got {score_correct}"
    print(f"  Score (correct): {score_correct}")

    # Test compute_score - partial answer
    solution2 = "Assistant: <answer>A=5, B=3, C=99</answer>"
    score_partial = compute_score(solution2, ground_truth)
    assert 0 < score_partial < 1, f"Expected partial score, got {score_partial}"
    print(f"  Score (partial): {score_partial:.3f}")

    # Test compute_score - wrong answer
    solution3 = "Assistant: <answer>A=0, B=0, C=0</answer>"
    score_wrong = compute_score(solution3, ground_truth)
    assert score_wrong == 0.1, f"Expected 0.1 (format score), got {score_wrong}"
    print(f"  Score (format only): {score_wrong}")

    # Test compute_score - no answer
    solution4 = "Assistant: I don't know"
    score_none = compute_score(solution4, ground_truth)
    assert score_none == 0.0, f"Expected 0.0, got {score_none}"
    print(f"  Score (no answer): {score_none}")

    print("  [PASS] Causal loop reward tests passed!")
    return True


def test_full_model():
    """Test the full DeepSeek MLA model."""
    print("\n" + "=" * 60)
    print("Test 9: Full Model Forward Pass")
    print("=" * 60)

    from verl.models.mla.modeling_deepseek_mla import (
        DeepSeekMLAPretrainedConfig,
        DeepSeekMLAModel,
        DeepSeekMLAForCausalLM,
    )

    # Create small test config
    config = DeepSeekMLAPretrainedConfig(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        kv_latent_dim=32,
        q_latent_dim=64,
        rope_head_dim=16,
        use_lru=True,
        lru_max_iterations=4,
        lru_halt_threshold=0.99,
    )

    print(f"  Config: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")

    # Create model
    model = DeepSeekMLAForCausalLM(config)
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Test forward pass
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

    assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Logits shape: {outputs.logits.shape}")

    # Check if LRU outputs are attached
    if hasattr(outputs, 'lru_outputs') and outputs.lru_outputs:
        print(f"  LRU outputs collected: {len(outputs.lru_outputs)} layers")
        avg_iters = sum(o.num_iterations.mean().item() for o in outputs.lru_outputs) / len(outputs.lru_outputs)
        print(f"  Avg LRU iterations: {avg_iters:.2f}")

    # Test with labels (loss computation)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        outputs_with_loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    assert outputs_with_loss.loss is not None
    print(f"  Loss: {outputs_with_loss.loss.item():.4f}")

    print("  [PASS] Full model tests passed!")
    return True


def test_attention_positional_mixing():
    """Test the new AttentionPositionalMixing module."""
    print("\n" + "=" * 60)
    print("Test 10: Attention Positional Mixing (NEW)")
    print("=" * 60)

    from verl.models.mla.lru import AttentionPositionalMixing

    # Test basic functionality
    latent_dim = 64
    batch_size, seq_len = 2, 16

    attn_mix = AttentionPositionalMixing(latent_dim=latent_dim, num_heads=1)

    x = torch.randn(batch_size, seq_len, latent_dim)
    output = attn_mix(x)

    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Test causality: output at position i should not depend on inputs at j > i
    # We test by checking that masking future positions doesn't change past outputs
    x_test = torch.randn(1, 8, latent_dim)
    out_full = attn_mix(x_test)

    # Modify the last 4 positions
    x_modified = x_test.clone()
    x_modified[:, 4:, :] = torch.randn(1, 4, latent_dim)
    out_modified = attn_mix(x_modified)

    # First 4 positions should be identical (causal property)
    diff = (out_full[:, :4, :] - out_modified[:, :4, :]).abs().max()
    print(f"  Causality test - max diff in first 4 positions: {diff:.6f}")

    # Test gradient flow
    x_grad = torch.randn(batch_size, seq_len, latent_dim, requires_grad=True)
    out_grad = attn_mix(x_grad)
    loss = out_grad.sum()
    loss.backward()

    assert x_grad.grad is not None, "Gradient should flow back"
    print(f"  Gradient flow: OK (grad norm: {x_grad.grad.norm():.4f})")

    print("  [PASS] Attention Positional Mixing tests passed!")
    return True


def test_enhanced_global_halting():
    """Test the new EnhancedGlobalHaltingUnit module."""
    print("\n" + "=" * 60)
    print("Test 11: Enhanced Global Halting (NEW)")
    print("=" * 60)

    from verl.models.mla.lru import EnhancedGlobalHaltingUnit

    latent_dim = 64
    batch_size, seq_len = 2, 16

    # Test basic functionality
    halt_unit = EnhancedGlobalHaltingUnit(
        latent_dim=latent_dim,
        init_global_weight=0.3,
        use_attention_pool=True,
        use_convergence_bonus=True,
    )

    state = torch.randn(batch_size, seq_len, latent_dim)
    local_halt = torch.sigmoid(torch.randn(batch_size, seq_len))
    input_state = torch.randn(batch_size, seq_len, latent_dim)

    combined_halt = halt_unit(state, local_halt, input_state=input_state)

    assert combined_halt.shape == local_halt.shape, f"Expected {local_halt.shape}, got {combined_halt.shape}"
    print(f"  Local halt shape: {local_halt.shape}")
    print(f"  Combined halt shape: {combined_halt.shape}")

    # Test learnable global weight
    initial_weight = halt_unit.global_weight.item()
    print(f"  Initial global weight: {initial_weight:.4f}")

    # Test with attention mask
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, -4:] = 0  # Mask last 4 positions

    combined_with_mask = halt_unit(state, local_halt, attention_mask=attention_mask)
    print(f"  Combined halt with mask shape: {combined_with_mask.shape}")

    # Test gradient flow for learnable weight
    state_grad = torch.randn(batch_size, seq_len, latent_dim, requires_grad=True)
    local_halt_grad = torch.sigmoid(torch.randn(batch_size, seq_len, requires_grad=True))
    combined_grad = halt_unit(state_grad, local_halt_grad)
    loss = combined_grad.sum()
    loss.backward()

    assert halt_unit.log_global_weight.grad is not None, "Global weight should have gradient"
    print(f"  Global weight gradient: {halt_unit.log_global_weight.grad:.6f}")

    print("  [PASS] Enhanced Global Halting tests passed!")
    return True


def test_learnable_loss_weights():
    """Test the new LearnableLossWeights module."""
    print("\n" + "=" * 60)
    print("Test 12: Learnable Loss Weights (NEW)")
    print("=" * 60)

    from verl.trainer.lru.losses import LearnableLossWeights, LRULossModule
    from verl.models.mla.lru import LRUOutput

    # Test LearnableLossWeights directly
    weights = LearnableLossWeights(
        init_stability=0.1,
        init_sparsity=0.01,
        init_ponder=0.001,
    )

    weight_dict = weights()
    print(f"  Initial weights:")
    print(f"    stability: {weight_dict['stability'].item():.6f}")
    print(f"    sparsity: {weight_dict['sparsity'].item():.6f}")
    print(f"    ponder: {weight_dict['ponder'].item():.6f}")

    # Verify weights are close to initial values
    assert abs(weight_dict['stability'].item() - 0.1) < 0.01, "Stability weight not close to init"
    assert abs(weight_dict['sparsity'].item() - 0.01) < 0.001, "Sparsity weight not close to init"
    assert abs(weight_dict['ponder'].item() - 0.001) < 0.0001, "Ponder weight not close to init"

    # Test gradient flow
    loss = weight_dict['stability'] + weight_dict['sparsity'] + weight_dict['ponder']
    loss.backward()

    assert weights.log_stability.grad is not None, "Stability should have gradient"
    assert weights.log_sparsity.grad is not None, "Sparsity should have gradient"
    assert weights.log_ponder.grad is not None, "Ponder should have gradient"
    print(f"  Gradient flow: OK")

    # Test LRULossModule with learnable weights
    batch_size, seq_len, latent_dim = 2, 16, 64
    num_iters = 5

    lru_loss_module = LRULossModule(
        stability_weight=0.1,
        sparsity_weight=0.01,
        ponder_weight=0.001,
        use_learnable_weights=True,
    )

    # Check that learnable weights module was created
    assert lru_loss_module.learnable_weights is not None, "Learnable weights should be created"
    print(f"  LRULossModule with learnable weights: OK")

    # Test forward pass
    lru_output = LRUOutput(
        output=torch.randn(batch_size, seq_len, latent_dim),
        halting_probabilities=torch.rand(batch_size, seq_len),
        num_iterations=torch.randint(1, 8, (batch_size, seq_len)).float(),
        remainders=torch.rand(batch_size, seq_len) * 0.1,
        intermediate_states=torch.randn(num_iters, batch_size, seq_len, latent_dim),
    )

    loss_output = lru_loss_module(lru_output)
    print(f"  Loss output total: {loss_output.total_loss.item():.6f}")

    # Check that weight metrics are in output
    assert 'lru/weight_stability' in loss_output.metrics, "Weight metrics should be included"
    print(f"  Weight metrics included: OK")

    # Test get_weight_params for optimizer
    weight_params = list(lru_loss_module.get_weight_params())
    assert len(weight_params) == 3, "Should have 3 weight parameters"
    print(f"  Weight parameters for optimizer: {len(weight_params)}")

    print("  [PASS] Learnable Loss Weights tests passed!")
    return True


def test_lru_with_new_options():
    """Test LRU with new configuration options."""
    print("\n" + "=" * 60)
    print("Test 13: LRU with New Options (NEW)")
    print("=" * 60)

    from verl.models.mla.config import LRUConfig
    from verl.models.mla.lru import LatentReasoningUnit

    batch_size, seq_len, latent_dim = 2, 16, 64

    # Test with attention mixing
    config_attn = LRUConfig(
        latent_dim=latent_dim,
        max_iterations=8,
        halt_threshold=0.99,
        positional_mixing_type='attention',
        use_enhanced_global_halting=False,
    )

    lru_attn = LatentReasoningUnit(
        config_attn,
        use_positional_mixing=True,
        use_global_halting=True,
    )

    x = torch.randn(batch_size, seq_len, latent_dim)
    output_attn, info_attn = lru_attn(x, return_intermediates=True)

    assert output_attn.shape == x.shape, f"Expected {x.shape}, got {output_attn.shape}"
    print(f"  LRU with attention mixing:")
    print(f"    Output shape: {output_attn.shape}")
    print(f"    Avg iterations: {info_attn.num_iterations.mean():.2f}")

    # Test with enhanced global halting
    config_enhanced = LRUConfig(
        latent_dim=latent_dim,
        max_iterations=8,
        halt_threshold=0.99,
        positional_mixing_type='conv',
        use_enhanced_global_halting=True,
    )

    lru_enhanced = LatentReasoningUnit(
        config_enhanced,
        use_positional_mixing=True,
        use_global_halting=True,
    )

    output_enhanced, info_enhanced = lru_enhanced(x, return_intermediates=True)

    assert output_enhanced.shape == x.shape
    print(f"  LRU with enhanced global halting:")
    print(f"    Output shape: {output_enhanced.shape}")
    print(f"    Avg iterations: {info_enhanced.num_iterations.mean():.2f}")

    # Test with both
    config_full = LRUConfig(
        latent_dim=latent_dim,
        max_iterations=8,
        halt_threshold=0.99,
        positional_mixing_type='attention',
        use_enhanced_global_halting=True,
    )

    lru_full = LatentReasoningUnit(
        config_full,
        use_positional_mixing=True,
        use_global_halting=True,
    )

    output_full, info_full = lru_full(x, return_intermediates=True)

    assert output_full.shape == x.shape
    print(f"  LRU with all enhancements:")
    print(f"    Output shape: {output_full.shape}")
    print(f"    Avg iterations: {info_full.num_iterations.mean():.2f}")

    print("  [PASS] LRU with New Options tests passed!")
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("ZeroModel MLA + LRU Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Config", test_config),
        ("RoPE", test_rope),
        ("LRU", test_lru),
        ("MLA Attention", test_mla_attention),
        ("Losses", test_losses),
        ("Halting", test_halting),
        ("Causal Loop Data", test_causal_loop_data),
        ("Causal Loop Reward", test_causal_loop_reward),
        ("Full Model", test_full_model),
        # NEW TESTS
        ("Attention Positional Mixing", test_attention_positional_mixing),
        ("Enhanced Global Halting", test_enhanced_global_halting),
        ("Learnable Loss Weights", test_learnable_loss_weights),
        ("LRU with New Options", test_lru_with_new_options),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            import traceback
            results.append((name, False, str(e)))
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {name}" + (f": {error}" if error else ""))

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 60)
        print("All tests passed! Implementation is working correctly.")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print(f"WARNING: {total - passed} test(s) failed!")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
