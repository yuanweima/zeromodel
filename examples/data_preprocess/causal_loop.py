"""
Causal Loop Prediction Task - Data Generation

This module generates causal reasoning problems with varying complexity levels.
The task requires models to trace causal chains and predict final states.

Difficulty Levels:
- Level 1: Linear chains (A→B→C→D)
- Level 2: Simple cycles (A→B→C→A)
- Level 3: Multi-path convergence (A→C, B→C, C→D)
- Level 4: Complex nested loops

Example Problem (Level 2):
    Initial: A=1, B=0, C=0
    Rules: A→B (copy), B→C (add 1), C→A (if C>1 then A+=1)
    After 3 steps: A=?, B=?, C=?
"""

import os
import json
import random
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from tqdm import tqdm
from datasets import Dataset


@dataclass
class CausalRule:
    """A causal rule defining how one variable affects another."""
    source: str  # Source variable
    target: str  # Target variable
    operation: str  # Operation type: 'copy', 'add', 'mult', 'conditional'
    value: int = 0  # Operation parameter
    condition: Optional[str] = None  # Condition for conditional operations

    def apply(self, state: Dict[str, int]) -> int:
        """Apply this rule to compute the new target value."""
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
            # Conditional: if source > threshold, apply operation
            if source_val > self.value:
                return state.get(self.target, 0) + 1
            return state.get(self.target, 0)
        elif self.operation == 'mod':
            return source_val % max(self.value, 1)
        else:
            return state.get(self.target, 0)

    def to_string(self) -> str:
        """Convert rule to human-readable string."""
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
    """A causal graph with variables and rules."""
    variables: List[str]
    rules: List[CausalRule]
    initial_state: Dict[str, int] = field(default_factory=dict)

    def step(self, state: Dict[str, int]) -> Dict[str, int]:
        """Execute one step of causal propagation."""
        new_state = state.copy()
        # Apply rules in order
        for rule in self.rules:
            new_state[rule.target] = rule.apply(state)
        return new_state

    def simulate(self, num_steps: int) -> List[Dict[str, int]]:
        """Simulate the causal graph for multiple steps."""
        states = [self.initial_state.copy()]
        current = self.initial_state.copy()
        for _ in range(num_steps):
            current = self.step(current)
            states.append(current.copy())
        return states

    def get_final_state(self, num_steps: int) -> Dict[str, int]:
        """Get the final state after simulation."""
        return self.simulate(num_steps)[-1]

    def to_problem_string(self, num_steps: int) -> str:
        """Convert to problem description string."""
        lines = []
        lines.append("Initial State:")
        for var in self.variables:
            lines.append(f"  {var} = {self.initial_state.get(var, 0)}")

        lines.append("\nCausal Rules (applied each step):")
        for i, rule in enumerate(self.rules, 1):
            lines.append(f"  {i}. {rule.to_string()}")

        lines.append(f"\nQuestion: After {num_steps} steps, what is the state of each variable?")

        return "\n".join(lines)


class CausalLoopGenerator:
    """Generator for causal loop problems at different difficulty levels."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_level1(self) -> Tuple[CausalGraph, int]:
        """Level 1: Linear chain (A→B→C→D)"""
        num_vars = self.rng.randint(3, 5)
        variables = [chr(ord('A') + i) for i in range(num_vars)]

        # Initial state: first variable has a value
        initial_state = {var: 0 for var in variables}
        initial_state[variables[0]] = self.rng.randint(1, 5)

        # Rules: each variable copies from previous
        rules = []
        for i in range(1, num_vars):
            op = self.rng.choice(['copy', 'add_source'])
            rules.append(CausalRule(
                source=variables[i-1],
                target=variables[i],
                operation=op,
            ))

        graph = CausalGraph(variables=variables, rules=rules, initial_state=initial_state)
        num_steps = self.rng.randint(2, 4)

        return graph, num_steps

    def generate_level2(self) -> Tuple[CausalGraph, int]:
        """Level 2: Simple cycle (A→B→C→A)"""
        num_vars = self.rng.randint(3, 4)
        variables = [chr(ord('A') + i) for i in range(num_vars)]

        # Initial state
        initial_state = {var: self.rng.randint(0, 3) for var in variables}

        # Rules: cycle through variables
        rules = []
        for i in range(num_vars):
            next_idx = (i + 1) % num_vars
            op = self.rng.choice(['add_source', 'copy'])
            rules.append(CausalRule(
                source=variables[i],
                target=variables[next_idx],
                operation=op,
            ))

        # Add one more operation to make it interesting
        if self.rng.random() > 0.5:
            rules.append(CausalRule(
                source=variables[0],
                target=variables[0],
                operation='add',
                value=1,
            ))

        graph = CausalGraph(variables=variables, rules=rules, initial_state=initial_state)
        num_steps = self.rng.randint(2, 3)

        return graph, num_steps

    def generate_level3(self) -> Tuple[CausalGraph, int]:
        """Level 3: Multi-path convergence (A→C, B→C, C→D)"""
        variables = ['A', 'B', 'C', 'D']

        # Initial state
        initial_state = {
            'A': self.rng.randint(1, 3),
            'B': self.rng.randint(1, 3),
            'C': 0,
            'D': 0,
        }

        # Rules: multiple paths converge
        rules = [
            CausalRule(source='A', target='C', operation='add_source'),
            CausalRule(source='B', target='C', operation='add_source'),
            CausalRule(source='C', target='D', operation='copy'),
        ]

        # Add feedback
        if self.rng.random() > 0.5:
            rules.append(CausalRule(
                source='D',
                target='A',
                operation='conditional',
                value=2,  # if D > 2, A += 1
            ))

        graph = CausalGraph(variables=variables, rules=rules, initial_state=initial_state)
        num_steps = self.rng.randint(2, 4)

        return graph, num_steps

    def generate_level4(self) -> Tuple[CausalGraph, int]:
        """Level 4: Complex nested loops"""
        variables = ['A', 'B', 'C', 'D', 'E']

        # Initial state
        initial_state = {var: self.rng.randint(0, 2) for var in variables}
        initial_state['A'] = self.rng.randint(1, 3)

        # Complex rules with nested dependencies
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
        """Generate a problem at the specified difficulty level."""
        generators = {
            1: self.generate_level1,
            2: self.generate_level2,
            3: self.generate_level3,
            4: self.generate_level4,
        }
        if level not in generators:
            raise ValueError(f"Unknown level: {level}. Must be 1-4.")
        return generators[level]()


def make_prefix(graph: CausalGraph, num_steps: int, template_type: str) -> str:
    """Create the prompt prefix for the model."""
    problem = graph.to_problem_string(num_steps)

    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Solve this causal reasoning problem:

{problem}

Show your step-by-step reasoning in <think> </think> tags. Then provide the final state of all variables in <answer> </answer> tags in the format: A=value, B=value, ...
Assistant: Let me trace through the causal rules step by step.
<think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful assistant that solves causal reasoning problems. You first think about the reasoning process step by step, then provide the answer.<|im_end|>
<|im_start|>user
Solve this causal reasoning problem:

{problem}

Show your step-by-step reasoning in <think> </think> tags. Then provide the final state of all variables in <answer> </answer> tags in the format: A=value, B=value, ...<|im_end|>
<|im_start|>assistant
Let me trace through the causal rules step by step.
<think>"""
    else:
        raise ValueError(f"Unknown template type: {template_type}")

    return prefix


def generate_dataset(
    num_samples: int,
    levels: List[int] = [1, 2, 3, 4],
    level_weights: Optional[List[float]] = None,
    seed: int = 42,
) -> List[Dict]:
    """Generate a dataset of causal loop problems.

    Args:
        num_samples: Number of samples to generate
        levels: List of difficulty levels to include
        level_weights: Optional weights for each level (default: uniform)
        seed: Random seed

    Returns:
        List of problem dictionaries
    """
    generator = CausalLoopGenerator(seed=seed)
    rng = random.Random(seed)

    if level_weights is None:
        level_weights = [1.0] * len(levels)

    # Normalize weights
    total = sum(level_weights)
    level_weights = [w / total for w in level_weights]

    samples = []
    for i in tqdm(range(num_samples), desc="Generating causal loop problems"):
        # Select level based on weights
        level = rng.choices(levels, weights=level_weights)[0]

        # Generate problem
        graph, num_steps = generator.generate(level)
        final_state = graph.get_final_state(num_steps)

        samples.append({
            'level': level,
            'variables': graph.variables,
            'rules': [r.to_string() for r in graph.rules],
            'initial_state': graph.initial_state,
            'num_steps': num_steps,
            'final_state': final_state,
            'problem_text': graph.to_problem_string(num_steps),
            '_graph': graph,  # For prefix generation
        })

    return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/causal_loop')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=50000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--levels', type=str, default='1,2,3,4',
                        help='Comma-separated list of difficulty levels')
    parser.add_argument('--level_weights', type=str, default=None,
                        help='Comma-separated weights for each level')
    parser.add_argument('--template_type', type=str, default='base',
                        choices=['base', 'qwen-instruct'])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Parse levels and weights
    levels = [int(x) for x in args.levels.split(',')]
    level_weights = None
    if args.level_weights:
        level_weights = [float(x) for x in args.level_weights.split(',')]

    data_source = 'causal_loop'

    # Generate datasets
    print(f"Generating {args.train_size} training samples...")
    train_samples = generate_dataset(
        num_samples=args.train_size,
        levels=levels,
        level_weights=level_weights,
        seed=args.seed,
    )

    print(f"Generating {args.test_size} test samples...")
    test_samples = generate_dataset(
        num_samples=args.test_size,
        levels=levels,
        level_weights=level_weights,
        seed=args.seed + 10000,  # Different seed for test
    )

    def make_map_fn(split):
        def process_fn(sample, idx):
            graph = sample['_graph']
            num_steps = sample['num_steps']
            question = make_prefix(graph, num_steps, template_type=args.template_type)

            # Ground truth for reward computation
            solution = {
                'final_state': sample['final_state'],
                'variables': sample['variables'],
                'level': sample['level'],
                'num_steps': sample['num_steps'],
            }

            data = {
                'data_source': data_source,
                'prompt': [{
                    'role': 'user',
                    'content': question,
                }],
                'ability': 'causal_reasoning',
                'reward_model': {
                    'style': 'rule',
                    'ground_truth': solution,
                },
                'extra_info': {
                    'split': split,
                    'index': idx,
                    'level': sample['level'],
                    'num_steps': sample['num_steps'],
                },
            }
            return data

        return process_fn

    # Process datasets
    train_data = [make_map_fn('train')(s, i) for i, s in enumerate(train_samples)]
    test_data = [make_map_fn('test')(s, i) for i, s in enumerate(test_samples)]

    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # Save
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print(f"Saved datasets to {local_dir}")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")

    # Print level distribution
    from collections import Counter
    train_levels = Counter(s['extra_info']['level'] for s in train_data)
    print(f"\nTrain level distribution: {dict(train_levels)}")

    # Save metadata
    metadata = {
        'levels': levels,
        'level_weights': level_weights,
        'template_type': args.template_type,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'seed': args.seed,
    }
    with open(os.path.join(local_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Copy to HDFS if specified
    if args.hdfs_dir:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied to HDFS: {args.hdfs_dir}")
