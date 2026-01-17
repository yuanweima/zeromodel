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
Comprehensive evaluation metrics for LRU ablation studies.

This module provides metrics specifically designed for:
1. Accuracy analysis across difficulty levels
2. Computational efficiency (iterations, FLOPs)
3. Reasoning quality (convergence, stability)
4. Generalization analysis

Usage:
    from verl.utils.evaluation_metrics import MetricsCollector, EvaluationReport

    collector = MetricsCollector()
    for batch in dataloader:
        outputs = model(batch)
        collector.add_batch(
            predictions=outputs.predictions,
            targets=batch.targets,
            lru_outputs=outputs.lru_outputs,
            metadata=batch.metadata,
        )

    report = collector.compute_all()
    print(report)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json


@dataclass
class LRUMetrics:
    """Metrics specific to LRU reasoning behavior."""
    avg_iterations: float  # Average iterations per token
    max_iterations: float  # Maximum iterations used
    min_iterations: float  # Minimum iterations used
    std_iterations: float  # Standard deviation of iterations
    convergence_ratio: float  # Ratio that converged before max_iter
    early_halt_ratio: float  # Ratio halting at iteration <= 2
    late_halt_ratio: float  # Ratio halting at iteration >= max_iter - 1
    ponder_cost: float  # Average normalized ponder cost

    # Per-layer statistics
    iterations_by_layer: Dict[int, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"""LRU Behavior Metrics:
  Avg iterations:     {self.avg_iterations:.2f} ± {self.std_iterations:.2f}
  Range:              [{self.min_iterations:.0f}, {self.max_iterations:.0f}]
  Convergence ratio:  {self.convergence_ratio:.1%}
  Early halt ratio:   {self.early_halt_ratio:.1%}
  Late halt ratio:    {self.late_halt_ratio:.1%}
  Ponder cost:        {self.ponder_cost:.4f}"""


@dataclass
class AccuracyMetrics:
    """Accuracy metrics by difficulty level and other dimensions."""
    overall_accuracy: float
    accuracy_by_level: Dict[int, float]
    accuracy_by_num_vars: Dict[int, float]
    accuracy_by_num_steps: Dict[int, float]

    # Detailed breakdown
    full_correct: int  # Fully correct predictions
    partial_correct: int  # Partially correct
    format_errors: int  # Valid format, all wrong
    no_answer: int  # No parseable answer
    total: int

    def __str__(self) -> str:
        lines = [f"Accuracy Metrics:"]
        lines.append(f"  Overall: {self.overall_accuracy:.1%} ({self.full_correct}/{self.total})")
        lines.append(f"  By level:")
        for level in sorted(self.accuracy_by_level.keys()):
            lines.append(f"    Level {level}: {self.accuracy_by_level[level]:.1%}")
        lines.append(f"  Breakdown:")
        lines.append(f"    Full correct:    {self.full_correct}")
        lines.append(f"    Partial correct: {self.partial_correct}")
        lines.append(f"    Format errors:   {self.format_errors}")
        lines.append(f"    No answer:       {self.no_answer}")
        return "\n".join(lines)


@dataclass
class EfficiencyMetrics:
    """Computational efficiency metrics."""
    avg_flops_per_token: float
    avg_latency_ms: float  # If timing is available
    memory_peak_mb: float  # Peak memory usage

    # Comparison to baseline (fixed iterations)
    flops_reduction_vs_fixed: float  # Relative to max_iterations
    speedup_vs_fixed: float

    # KV cache efficiency
    kv_compression_ratio: float  # Actual vs standard attention

    def __str__(self) -> str:
        return f"""Efficiency Metrics:
  Avg FLOPs/token:      {self.avg_flops_per_token:.2e}
  Avg latency:          {self.avg_latency_ms:.2f} ms
  Peak memory:          {self.memory_peak_mb:.1f} MB
  FLOPs reduction:      {self.flops_reduction_vs_fixed:.1%}
  Speedup vs fixed:     {self.speedup_vs_fixed:.2f}x
  KV compression:       {self.kv_compression_ratio:.1%}"""


@dataclass
class GeneralizationMetrics:
    """Metrics for analyzing generalization."""
    train_accuracy: float
    val_accuracy: float
    generalization_gap: float  # train - val accuracy

    # OOD performance (if available)
    ood_accuracy: Optional[float] = None
    ood_gap: Optional[float] = None  # val - ood accuracy

    # Length generalization
    accuracy_by_seq_length: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["Generalization Metrics:"]
        lines.append(f"  Train accuracy:     {self.train_accuracy:.1%}")
        lines.append(f"  Val accuracy:       {self.val_accuracy:.1%}")
        lines.append(f"  Generalization gap: {self.generalization_gap:+.1%}")
        if self.ood_accuracy is not None:
            lines.append(f"  OOD accuracy:       {self.ood_accuracy:.1%}")
            lines.append(f"  OOD gap:            {self.ood_gap:+.1%}")
        return "\n".join(lines)


@dataclass
class StabilityMetrics:
    """Training stability metrics."""
    loss_variance: float  # Variance in training loss
    gradient_norm_mean: float
    gradient_norm_std: float
    num_gradient_spikes: int  # Count of gradient > 3*mean

    # LRU-specific stability
    halt_prob_entropy: float  # Entropy of halting distribution
    iteration_consistency: float  # How consistent iterations are for similar inputs

    def __str__(self) -> str:
        return f"""Stability Metrics:
  Loss variance:          {self.loss_variance:.6f}
  Gradient norm:          {self.gradient_norm_mean:.4f} ± {self.gradient_norm_std:.4f}
  Gradient spikes:        {self.num_gradient_spikes}
  Halt prob entropy:      {self.halt_prob_entropy:.4f}
  Iteration consistency:  {self.iteration_consistency:.1%}"""


class MetricsCollector:
    """Collect and aggregate metrics during evaluation."""

    def __init__(self, max_iterations: int = 8):
        self.max_iterations = max_iterations

        # Raw data storage
        self.predictions: List[Any] = []
        self.targets: List[Any] = []
        self.correct: List[bool] = []
        self.partial_scores: List[float] = []

        # Metadata
        self.levels: List[int] = []
        self.num_vars: List[int] = []
        self.num_steps: List[int] = []
        self.seq_lengths: List[int] = []

        # LRU-specific
        self.iterations_used: List[float] = []
        self.halt_probs: List[List[float]] = []
        self.converged: List[bool] = []
        self.layer_iterations: List[Dict[int, float]] = []

        # Efficiency
        self.flops: List[float] = []
        self.latencies: List[float] = []

        # Error analysis
        self.format_errors: int = 0
        self.no_answer: int = 0

    def add_sample(
        self,
        prediction: Any,
        target: Any,
        correct: bool,
        partial_score: float = 0.0,
        level: int = 1,
        num_vars: int = 3,
        num_steps: int = 1,
        seq_length: int = 0,
        iterations_used: float = 0.0,
        halt_probs: Optional[List[float]] = None,
        converged: bool = False,
        layer_iterations: Optional[Dict[int, float]] = None,
        flops: float = 0.0,
        latency: float = 0.0,
        is_format_error: bool = False,
        is_no_answer: bool = False,
    ):
        """Add a single sample's results."""
        self.predictions.append(prediction)
        self.targets.append(target)
        self.correct.append(correct)
        self.partial_scores.append(partial_score)

        self.levels.append(level)
        self.num_vars.append(num_vars)
        self.num_steps.append(num_steps)
        self.seq_lengths.append(seq_length)

        self.iterations_used.append(iterations_used)
        self.halt_probs.append(halt_probs or [])
        self.converged.append(converged)
        self.layer_iterations.append(layer_iterations or {})

        self.flops.append(flops)
        self.latencies.append(latency)

        if is_format_error:
            self.format_errors += 1
        if is_no_answer:
            self.no_answer += 1

    def add_batch(
        self,
        predictions: List[Any],
        targets: List[Any],
        correct: List[bool],
        partial_scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict]] = None,
        lru_outputs: Optional[List[Dict]] = None,
        flops: Optional[List[float]] = None,
        latencies: Optional[List[float]] = None,
    ):
        """Add a batch of results."""
        batch_size = len(predictions)
        partial_scores = partial_scores or [0.0] * batch_size
        metadata = metadata or [{}] * batch_size
        lru_outputs = lru_outputs or [{}] * batch_size
        flops = flops or [0.0] * batch_size
        latencies = latencies or [0.0] * batch_size

        for i in range(batch_size):
            meta = metadata[i] if i < len(metadata) else {}
            lru_out = lru_outputs[i] if i < len(lru_outputs) else {}

            self.add_sample(
                prediction=predictions[i],
                target=targets[i],
                correct=correct[i],
                partial_score=partial_scores[i],
                level=meta.get('level', 1),
                num_vars=meta.get('num_vars', 3),
                num_steps=meta.get('num_steps', 1),
                seq_length=meta.get('seq_length', 0),
                iterations_used=lru_out.get('avg_iterations', 0.0),
                halt_probs=lru_out.get('halt_probs', None),
                converged=lru_out.get('converged', False),
                layer_iterations=lru_out.get('layer_iterations', None),
                flops=flops[i],
                latency=latencies[i],
                is_format_error=meta.get('is_format_error', False),
                is_no_answer=meta.get('is_no_answer', False),
            )

    def compute_accuracy_metrics(self) -> AccuracyMetrics:
        """Compute accuracy-related metrics."""
        total = len(self.correct)
        if total == 0:
            return AccuracyMetrics(
                overall_accuracy=0.0,
                accuracy_by_level={},
                accuracy_by_num_vars={},
                accuracy_by_num_steps={},
                full_correct=0,
                partial_correct=0,
                format_errors=0,
                no_answer=0,
                total=0,
            )

        full_correct = sum(self.correct)
        partial_correct = sum(1 for s in self.partial_scores if 0 < s < 1)

        # Accuracy by level
        acc_by_level = defaultdict(list)
        for c, lvl in zip(self.correct, self.levels):
            acc_by_level[lvl].append(c)
        accuracy_by_level = {lvl: np.mean(vals) for lvl, vals in acc_by_level.items()}

        # Accuracy by num vars
        acc_by_vars = defaultdict(list)
        for c, nv in zip(self.correct, self.num_vars):
            acc_by_vars[nv].append(c)
        accuracy_by_num_vars = {nv: np.mean(vals) for nv, vals in acc_by_vars.items()}

        # Accuracy by num steps
        acc_by_steps = defaultdict(list)
        for c, ns in zip(self.correct, self.num_steps):
            acc_by_steps[ns].append(c)
        accuracy_by_num_steps = {ns: np.mean(vals) for ns, vals in acc_by_steps.items()}

        return AccuracyMetrics(
            overall_accuracy=full_correct / total,
            accuracy_by_level=dict(accuracy_by_level),
            accuracy_by_num_vars=dict(accuracy_by_num_vars),
            accuracy_by_num_steps=dict(accuracy_by_num_steps),
            full_correct=full_correct,
            partial_correct=partial_correct,
            format_errors=self.format_errors,
            no_answer=self.no_answer,
            total=total,
        )

    def compute_lru_metrics(self) -> LRUMetrics:
        """Compute LRU-specific metrics."""
        if not self.iterations_used or all(i == 0 for i in self.iterations_used):
            return LRUMetrics(
                avg_iterations=0.0,
                max_iterations=0.0,
                min_iterations=0.0,
                std_iterations=0.0,
                convergence_ratio=0.0,
                early_halt_ratio=0.0,
                late_halt_ratio=0.0,
                ponder_cost=0.0,
                iterations_by_layer={},
            )

        iters = np.array(self.iterations_used)
        valid_iters = iters[iters > 0]

        if len(valid_iters) == 0:
            valid_iters = np.array([0.0])

        # Basic stats
        avg_iterations = np.mean(valid_iters)
        max_iter = np.max(valid_iters)
        min_iter = np.min(valid_iters)
        std_iterations = np.std(valid_iters)

        # Halting patterns
        convergence_ratio = np.mean(self.converged) if self.converged else 0.0
        early_halt_ratio = np.mean(valid_iters <= 2)
        late_halt_ratio = np.mean(valid_iters >= self.max_iterations - 1)

        # Ponder cost (normalized by max iterations)
        ponder_cost = avg_iterations / self.max_iterations

        # Per-layer iterations
        layer_iters = defaultdict(list)
        for li in self.layer_iterations:
            for layer, it in li.items():
                layer_iters[layer].append(it)
        iterations_by_layer = {layer: np.mean(vals) for layer, vals in layer_iters.items()}

        return LRUMetrics(
            avg_iterations=avg_iterations,
            max_iterations=max_iter,
            min_iterations=min_iter,
            std_iterations=std_iterations,
            convergence_ratio=convergence_ratio,
            early_halt_ratio=early_halt_ratio,
            late_halt_ratio=late_halt_ratio,
            ponder_cost=ponder_cost,
            iterations_by_layer=iterations_by_layer,
        )

    def compute_efficiency_metrics(
        self,
        baseline_flops_per_token: float = 1e9,
        fixed_iter_flops_per_token: float = 1e9,
        kv_dim_ratio: float = 0.25,
    ) -> EfficiencyMetrics:
        """Compute efficiency metrics."""
        avg_flops = np.mean(self.flops) if any(f > 0 for f in self.flops) else 0.0
        avg_latency = np.mean(self.latencies) if any(l > 0 for l in self.latencies) else 0.0

        # Compute relative metrics
        if fixed_iter_flops_per_token > 0 and avg_flops > 0:
            flops_reduction = 1 - (avg_flops / fixed_iter_flops_per_token)
        else:
            flops_reduction = 0.0

        # Speedup based on iterations
        lru_metrics = self.compute_lru_metrics()
        if lru_metrics.avg_iterations > 0 and self.max_iterations > 0:
            speedup = self.max_iterations / lru_metrics.avg_iterations
        else:
            speedup = 1.0

        return EfficiencyMetrics(
            avg_flops_per_token=avg_flops,
            avg_latency_ms=avg_latency * 1000,  # Convert to ms
            memory_peak_mb=0.0,  # Needs separate tracking
            flops_reduction_vs_fixed=flops_reduction,
            speedup_vs_fixed=speedup,
            kv_compression_ratio=kv_dim_ratio,
        )

    def compute_all(self) -> Dict[str, Any]:
        """Compute all metrics and return as dictionary."""
        return {
            'accuracy': self.compute_accuracy_metrics(),
            'lru': self.compute_lru_metrics(),
            'efficiency': self.compute_efficiency_metrics(),
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report for an experiment."""
    experiment_name: str
    accuracy: AccuracyMetrics
    lru: LRUMetrics
    efficiency: EfficiencyMetrics
    generalization: Optional[GeneralizationMetrics] = None
    stability: Optional[StabilityMetrics] = None

    def __str__(self) -> str:
        lines = ["=" * 80]
        lines.append(f"EVALUATION REPORT: {self.experiment_name}")
        lines.append("=" * 80)
        lines.append("")
        lines.append(str(self.accuracy))
        lines.append("")
        lines.append(str(self.lru))
        lines.append("")
        lines.append(str(self.efficiency))
        if self.generalization:
            lines.append("")
            lines.append(str(self.generalization))
        if self.stability:
            lines.append("")
            lines.append(str(self.stability))
        lines.append("=" * 80)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # Helper to convert numpy types to Python types
        def convert_keys(d):
            if isinstance(d, dict):
                return {str(k): convert_keys(v) for k, v in d.items()}
            elif isinstance(d, (np.integer, np.floating)):
                return float(d)
            return d

        result = {
            'experiment_name': self.experiment_name,
            'accuracy': {
                'overall': float(self.accuracy.overall_accuracy),
                'by_level': convert_keys(self.accuracy.accuracy_by_level),
                'by_num_vars': convert_keys(self.accuracy.accuracy_by_num_vars),
                'by_num_steps': convert_keys(self.accuracy.accuracy_by_num_steps),
                'full_correct': self.accuracy.full_correct,
                'partial_correct': self.accuracy.partial_correct,
                'format_errors': self.accuracy.format_errors,
                'no_answer': self.accuracy.no_answer,
                'total': self.accuracy.total,
            },
            'lru': {
                'avg_iterations': float(self.lru.avg_iterations),
                'max_iterations': float(self.lru.max_iterations),
                'min_iterations': float(self.lru.min_iterations),
                'std_iterations': float(self.lru.std_iterations),
                'convergence_ratio': float(self.lru.convergence_ratio),
                'early_halt_ratio': float(self.lru.early_halt_ratio),
                'late_halt_ratio': float(self.lru.late_halt_ratio),
                'ponder_cost': float(self.lru.ponder_cost),
            },
            'efficiency': {
                'avg_flops_per_token': float(self.efficiency.avg_flops_per_token),
                'avg_latency_ms': float(self.efficiency.avg_latency_ms),
                'memory_peak_mb': float(self.efficiency.memory_peak_mb),
                'flops_reduction_vs_fixed': float(self.efficiency.flops_reduction_vs_fixed),
                'speedup_vs_fixed': float(self.efficiency.speedup_vs_fixed),
                'kv_compression_ratio': float(self.efficiency.kv_compression_ratio),
            },
        }

        if self.generalization:
            result['generalization'] = {
                'train_accuracy': self.generalization.train_accuracy,
                'val_accuracy': self.generalization.val_accuracy,
                'generalization_gap': self.generalization.generalization_gap,
                'ood_accuracy': self.generalization.ood_accuracy,
                'ood_gap': self.generalization.ood_gap,
            }

        if self.stability:
            result['stability'] = {
                'loss_variance': self.stability.loss_variance,
                'gradient_norm_mean': self.stability.gradient_norm_mean,
                'gradient_norm_std': self.stability.gradient_norm_std,
                'num_gradient_spikes': self.stability.num_gradient_spikes,
            }

        return result

    def to_json(self, filepath: str):
        """Save report to JSON file."""
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, cls=NumpyEncoder)


def compute_difficulty_curve(
    accuracy_by_level: Dict[int, float],
) -> Dict[str, float]:
    """Analyze how accuracy degrades with difficulty.

    Returns:
        Dictionary with curve analysis metrics
    """
    levels = sorted(accuracy_by_level.keys())
    accuracies = [accuracy_by_level[l] for l in levels]

    if len(levels) < 2:
        return {'slope': 0.0, 'intercept': 1.0, 'r_squared': 0.0}

    # Linear regression
    x = np.array(levels, dtype=float)
    y = np.array(accuracies)

    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-10)
    intercept = (sum_y - slope * sum_x) / n

    # R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) + 1e-10
    r_squared = 1 - (ss_res / ss_tot)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'difficulty_resilience': max(0, 1 + slope),  # How well it handles harder problems
    }


def compute_iteration_efficiency(
    accuracy_by_iterations: Dict[int, float],
    iteration_counts: Dict[int, int],
) -> Dict[str, float]:
    """Analyze relationship between iterations and accuracy.

    Returns:
        Dictionary with iteration analysis metrics
    """
    if not accuracy_by_iterations:
        return {'optimal_iterations': 0, 'efficiency_score': 0.0}

    iters = sorted(accuracy_by_iterations.keys())
    accs = [accuracy_by_iterations[i] for i in iters]

    # Find elbow point (diminishing returns)
    if len(iters) < 2:
        optimal_iterations = iters[0]
    else:
        # Simple: find where accuracy improvement drops below threshold
        improvements = [accs[i+1] - accs[i] for i in range(len(accs)-1)]
        threshold = 0.01  # 1% improvement threshold
        optimal_iterations = iters[0]
        for i, imp in enumerate(improvements):
            if imp < threshold:
                optimal_iterations = iters[i]
                break
            optimal_iterations = iters[i + 1]

    # Efficiency: accuracy per iteration
    if sum(iteration_counts.values()) > 0:
        weighted_acc = sum(
            accuracy_by_iterations.get(i, 0) * count
            for i, count in iteration_counts.items()
        ) / sum(iteration_counts.values())
        weighted_iter = sum(
            i * count for i, count in iteration_counts.items()
        ) / sum(iteration_counts.values())
        efficiency_score = weighted_acc / max(weighted_iter, 1)
    else:
        efficiency_score = 0.0

    return {
        'optimal_iterations': optimal_iterations,
        'efficiency_score': efficiency_score,
    }


def compare_experiments(
    reports: Dict[str, EvaluationReport],
    baseline_name: str = 'baseline',
) -> str:
    """Generate comparison table across experiments.

    Args:
        reports: Dictionary mapping experiment names to reports
        baseline_name: Name of baseline experiment

    Returns:
        Formatted comparison table
    """
    if baseline_name not in reports:
        baseline_name = list(reports.keys())[0]

    baseline = reports[baseline_name]

    lines = ["=" * 100]
    lines.append("EXPERIMENT COMPARISON")
    lines.append("=" * 100)

    # Header
    header = f"{'Experiment':<25} {'Accuracy':>10} {'Δ Acc':>10} {'Avg Iter':>10} {'Converge':>10} {'Speedup':>10}"
    lines.append(header)
    lines.append("-" * 100)

    for name, report in reports.items():
        delta_acc = report.accuracy.overall_accuracy - baseline.accuracy.overall_accuracy
        delta_str = f"{delta_acc:+.1%}" if name != baseline_name else "---"

        row = f"{name:<25} {report.accuracy.overall_accuracy:>10.1%} {delta_str:>10} {report.lru.avg_iterations:>10.2f} {report.lru.convergence_ratio:>10.1%} {report.efficiency.speedup_vs_fixed:>10.2f}x"
        lines.append(row)

    lines.append("-" * 100)

    # Per-level breakdown
    lines.append("\nAccuracy by Difficulty Level:")
    levels = sorted(set(
        lvl for r in reports.values() for lvl in r.accuracy.accuracy_by_level.keys()
    ))

    header = f"{'Experiment':<25}" + "".join(f"{'L' + str(l):>10}" for l in levels)
    lines.append(header)
    lines.append("-" * (25 + 10 * len(levels)))

    for name, report in reports.items():
        row = f"{name:<25}"
        for level in levels:
            acc = report.accuracy.accuracy_by_level.get(level, 0)
            row += f"{acc:>10.1%}"
        lines.append(row)

    lines.append("=" * 100)

    return "\n".join(lines)


# =============================================================================
# Main: Example usage
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)

    # Simulate evaluation results
    n_samples = 500

    # Create collector and add simulated data
    collector = MetricsCollector(max_iterations=8)

    for i in range(n_samples):
        level = np.random.choice([1, 2, 3, 4], p=[0.3, 0.3, 0.25, 0.15])
        num_vars = level + 2  # 3-6 variables
        num_steps = level * 2  # 2-8 steps

        # Accuracy decreases with difficulty
        base_acc = 0.95 - 0.15 * (level - 1)
        correct = np.random.random() < base_acc

        # Iterations increase with difficulty
        avg_iter = min(2 + level * 1.5 + np.random.normal(0, 0.5), 8)
        converged = avg_iter < 7.5

        collector.add_sample(
            prediction=f"pred_{i}",
            target=f"target_{i}",
            correct=correct,
            partial_score=0.5 if not correct and np.random.random() < 0.3 else 0.0,
            level=level,
            num_vars=num_vars,
            num_steps=num_steps,
            iterations_used=avg_iter,
            converged=converged,
            flops=1e9 * avg_iter / 8,
            latency=0.001 * avg_iter,
        )

    # Compute and display metrics
    accuracy = collector.compute_accuracy_metrics()
    lru = collector.compute_lru_metrics()
    efficiency = collector.compute_efficiency_metrics()

    report = EvaluationReport(
        experiment_name="mla_lru_adaptive",
        accuracy=accuracy,
        lru=lru,
        efficiency=efficiency,
    )

    print(report)

    # Difficulty curve analysis
    curve = compute_difficulty_curve(accuracy.accuracy_by_level)
    print("\nDifficulty Curve Analysis:")
    print(f"  Slope: {curve['slope']:.4f} (accuracy drop per level)")
    print(f"  Intercept: {curve['intercept']:.4f}")
    print(f"  R²: {curve['r_squared']:.4f}")
    print(f"  Difficulty resilience: {curve['difficulty_resilience']:.4f}")

    # Save to JSON
    report.to_json('/tmp/eval_report.json')
    print("\nReport saved to /tmp/eval_report.json")
