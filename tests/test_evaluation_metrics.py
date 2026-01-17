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
"""Tests for evaluation metrics utilities."""

import pytest
import numpy as np
import json
import tempfile
import sys
from pathlib import Path

# Add utils path to allow direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "verl" / "utils"))

from evaluation_metrics import (
    MetricsCollector,
    EvaluationReport,
    AccuracyMetrics,
    LRUMetrics,
    EfficiencyMetrics,
    GeneralizationMetrics,
    StabilityMetrics,
    compute_difficulty_curve,
    compute_iteration_efficiency,
    compare_experiments,
)


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_add_single_sample(self):
        """Test adding a single sample."""
        collector = MetricsCollector()

        collector.add_sample(
            prediction="A=1, B=2",
            target="A=1, B=2",
            correct=True,
            level=1,
            iterations_used=3.5,
        )

        assert len(collector.correct) == 1
        assert collector.correct[0] == True
        assert collector.levels[0] == 1

    def test_add_batch(self):
        """Test adding a batch of samples."""
        collector = MetricsCollector()

        predictions = ["pred1", "pred2", "pred3"]
        targets = ["target1", "target2", "target3"]
        correct = [True, False, True]

        collector.add_batch(
            predictions=predictions,
            targets=targets,
            correct=correct,
        )

        assert len(collector.correct) == 3
        assert sum(collector.correct) == 2

    def test_compute_accuracy_metrics(self):
        """Test accuracy metrics computation."""
        collector = MetricsCollector()

        # Add samples with different levels
        for level in [1, 1, 1, 2, 2, 3]:
            correct = level == 1  # Only level 1 is correct
            collector.add_sample(
                prediction="pred",
                target="target",
                correct=correct,
                level=level,
            )

        metrics = collector.compute_accuracy_metrics()

        assert isinstance(metrics, AccuracyMetrics)
        assert metrics.total == 6
        assert metrics.full_correct == 3
        assert metrics.accuracy_by_level[1] == 1.0
        assert metrics.accuracy_by_level[2] == 0.0
        assert metrics.accuracy_by_level[3] == 0.0

    def test_compute_lru_metrics(self):
        """Test LRU metrics computation."""
        collector = MetricsCollector(max_iterations=8)

        # Add samples with varying iterations
        for i in range(10):
            iterations = 2 + i * 0.5
            collector.add_sample(
                prediction="pred",
                target="target",
                correct=True,
                iterations_used=iterations,
                converged=iterations < 7,
            )

        metrics = collector.compute_lru_metrics()

        assert isinstance(metrics, LRUMetrics)
        assert metrics.avg_iterations > 0
        assert metrics.min_iterations == 2.0
        assert 0 <= metrics.convergence_ratio <= 1

    def test_compute_efficiency_metrics(self):
        """Test efficiency metrics computation."""
        collector = MetricsCollector(max_iterations=8)

        for i in range(10):
            collector.add_sample(
                prediction="pred",
                target="target",
                correct=True,
                iterations_used=4.0,  # Half of max
                flops=1e9 * 4 / 8,
                latency=0.005,
            )

        metrics = collector.compute_efficiency_metrics()

        assert isinstance(metrics, EfficiencyMetrics)
        assert metrics.speedup_vs_fixed == pytest.approx(2.0, rel=0.1)

    def test_empty_collector(self):
        """Test metrics with empty collector."""
        collector = MetricsCollector()

        accuracy = collector.compute_accuracy_metrics()
        lru = collector.compute_lru_metrics()

        assert accuracy.total == 0
        assert accuracy.overall_accuracy == 0.0
        assert lru.avg_iterations == 0.0


class TestAccuracyMetrics:
    """Tests for AccuracyMetrics dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        metrics = AccuracyMetrics(
            overall_accuracy=0.75,
            accuracy_by_level={1: 0.9, 2: 0.7, 3: 0.5},
            accuracy_by_num_vars={3: 0.8, 4: 0.6},
            accuracy_by_num_steps={2: 0.85, 4: 0.65},
            full_correct=75,
            partial_correct=10,
            format_errors=5,
            no_answer=10,
            total=100,
        )

        s = str(metrics)

        assert "75.0%" in s
        assert "Level 1" in s
        assert "75" in s


class TestLRUMetrics:
    """Tests for LRUMetrics dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        metrics = LRUMetrics(
            avg_iterations=5.5,
            max_iterations=8.0,
            min_iterations=2.0,
            std_iterations=1.5,
            convergence_ratio=0.85,
            early_halt_ratio=0.1,
            late_halt_ratio=0.2,
            ponder_cost=0.6875,
            iterations_by_layer={0: 5.0, 1: 5.5, 2: 6.0},
        )

        s = str(metrics)

        assert "5.50" in s
        assert "85.0%" in s


class TestEvaluationReport:
    """Tests for EvaluationReport class."""

    def test_report_creation(self):
        """Test creating a report."""
        accuracy = AccuracyMetrics(
            overall_accuracy=0.75,
            accuracy_by_level={1: 0.9},
            accuracy_by_num_vars={3: 0.8},
            accuracy_by_num_steps={2: 0.85},
            full_correct=75,
            partial_correct=10,
            format_errors=5,
            no_answer=10,
            total=100,
        )

        lru = LRUMetrics(
            avg_iterations=5.5,
            max_iterations=8.0,
            min_iterations=2.0,
            std_iterations=1.5,
            convergence_ratio=0.85,
            early_halt_ratio=0.1,
            late_halt_ratio=0.2,
            ponder_cost=0.6875,
        )

        efficiency = EfficiencyMetrics(
            avg_flops_per_token=1e9,
            avg_latency_ms=5.0,
            memory_peak_mb=1000.0,
            flops_reduction_vs_fixed=0.3,
            speedup_vs_fixed=1.5,
            kv_compression_ratio=0.25,
        )

        report = EvaluationReport(
            experiment_name="test_exp",
            accuracy=accuracy,
            lru=lru,
            efficiency=efficiency,
        )

        assert report.experiment_name == "test_exp"
        s = str(report)
        assert "test_exp" in s

    def test_to_dict(self):
        """Test dictionary conversion."""
        accuracy = AccuracyMetrics(
            overall_accuracy=0.75,
            accuracy_by_level={1: 0.9},
            accuracy_by_num_vars={},
            accuracy_by_num_steps={},
            full_correct=75,
            partial_correct=10,
            format_errors=5,
            no_answer=10,
            total=100,
        )

        lru = LRUMetrics(
            avg_iterations=5.5,
            max_iterations=8.0,
            min_iterations=2.0,
            std_iterations=1.5,
            convergence_ratio=0.85,
            early_halt_ratio=0.1,
            late_halt_ratio=0.2,
            ponder_cost=0.6875,
        )

        efficiency = EfficiencyMetrics(
            avg_flops_per_token=1e9,
            avg_latency_ms=5.0,
            memory_peak_mb=1000.0,
            flops_reduction_vs_fixed=0.3,
            speedup_vs_fixed=1.5,
            kv_compression_ratio=0.25,
        )

        report = EvaluationReport(
            experiment_name="test",
            accuracy=accuracy,
            lru=lru,
            efficiency=efficiency,
        )

        d = report.to_dict()

        assert d['experiment_name'] == "test"
        assert 'accuracy' in d
        assert 'lru' in d
        assert 'efficiency' in d

    def test_to_json(self):
        """Test JSON serialization."""
        accuracy = AccuracyMetrics(
            overall_accuracy=0.75,
            accuracy_by_level={1: 0.9},
            accuracy_by_num_vars={},
            accuracy_by_num_steps={},
            full_correct=75,
            partial_correct=10,
            format_errors=5,
            no_answer=10,
            total=100,
        )

        lru = LRUMetrics(
            avg_iterations=5.5,
            max_iterations=8.0,
            min_iterations=2.0,
            std_iterations=1.5,
            convergence_ratio=0.85,
            early_halt_ratio=0.1,
            late_halt_ratio=0.2,
            ponder_cost=0.6875,
        )

        efficiency = EfficiencyMetrics(
            avg_flops_per_token=1e9,
            avg_latency_ms=5.0,
            memory_peak_mb=1000.0,
            flops_reduction_vs_fixed=0.3,
            speedup_vs_fixed=1.5,
            kv_compression_ratio=0.25,
        )

        report = EvaluationReport(
            experiment_name="test",
            accuracy=accuracy,
            lru=lru,
            efficiency=efficiency,
        )

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report.to_json(f.name)

            # Read back and verify
            with open(f.name) as rf:
                data = json.load(rf)

            assert data['experiment_name'] == "test"
            assert data['accuracy']['overall'] == 0.75


class TestDifficultyCurve:
    """Tests for difficulty curve analysis."""

    def test_linear_curve(self):
        """Test with linear accuracy degradation."""
        accuracy_by_level = {
            1: 0.9,
            2: 0.7,
            3: 0.5,
            4: 0.3,
        }

        curve = compute_difficulty_curve(accuracy_by_level)

        assert 'slope' in curve
        assert 'intercept' in curve
        assert 'r_squared' in curve
        assert curve['slope'] < 0  # Accuracy decreases with level
        assert curve['r_squared'] > 0.9  # Good linear fit

    def test_single_level(self):
        """Test with single level."""
        accuracy_by_level = {1: 0.9}

        curve = compute_difficulty_curve(accuracy_by_level)

        assert curve['slope'] == 0.0

    def test_difficulty_resilience(self):
        """Test difficulty resilience metric."""
        # Model that degrades slowly
        good_model = {1: 0.95, 2: 0.90, 3: 0.85, 4: 0.80}
        # Model that degrades quickly
        bad_model = {1: 0.95, 2: 0.70, 3: 0.45, 4: 0.20}

        good_curve = compute_difficulty_curve(good_model)
        bad_curve = compute_difficulty_curve(bad_model)

        assert good_curve['difficulty_resilience'] > bad_curve['difficulty_resilience']


class TestIterationEfficiency:
    """Tests for iteration efficiency analysis."""

    def test_basic_efficiency(self):
        """Test basic efficiency computation."""
        accuracy_by_iterations = {
            2: 0.5,
            4: 0.7,
            6: 0.8,
            8: 0.82,
        }
        iteration_counts = {
            2: 10,
            4: 30,
            6: 40,
            8: 20,
        }

        result = compute_iteration_efficiency(accuracy_by_iterations, iteration_counts)

        assert 'optimal_iterations' in result
        assert 'efficiency_score' in result

    def test_empty_input(self):
        """Test with empty input."""
        result = compute_iteration_efficiency({}, {})

        assert result['optimal_iterations'] == 0
        assert result['efficiency_score'] == 0.0


class TestCompareExperiments:
    """Tests for experiment comparison."""

    def test_basic_comparison(self):
        """Test basic comparison table."""
        accuracy1 = AccuracyMetrics(
            overall_accuracy=0.7,
            accuracy_by_level={1: 0.9, 2: 0.6},
            accuracy_by_num_vars={},
            accuracy_by_num_steps={},
            full_correct=70,
            partial_correct=10,
            format_errors=5,
            no_answer=15,
            total=100,
        )

        accuracy2 = AccuracyMetrics(
            overall_accuracy=0.8,
            accuracy_by_level={1: 0.95, 2: 0.7},
            accuracy_by_num_vars={},
            accuracy_by_num_steps={},
            full_correct=80,
            partial_correct=10,
            format_errors=5,
            no_answer=5,
            total=100,
        )

        lru = LRUMetrics(
            avg_iterations=5.0,
            max_iterations=8.0,
            min_iterations=2.0,
            std_iterations=1.5,
            convergence_ratio=0.8,
            early_halt_ratio=0.1,
            late_halt_ratio=0.2,
            ponder_cost=0.625,
        )

        efficiency = EfficiencyMetrics(
            avg_flops_per_token=1e9,
            avg_latency_ms=5.0,
            memory_peak_mb=1000.0,
            flops_reduction_vs_fixed=0.3,
            speedup_vs_fixed=1.6,
            kv_compression_ratio=0.25,
        )

        reports = {
            'baseline': EvaluationReport("baseline", accuracy1, lru, efficiency),
            'improved': EvaluationReport("improved", accuracy2, lru, efficiency),
        }

        comparison = compare_experiments(reports, baseline_name='baseline')

        assert "baseline" in comparison
        assert "improved" in comparison
        assert "EXPERIMENT COMPARISON" in comparison


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test complete evaluation pipeline."""
        np.random.seed(42)

        # Simulate evaluation
        collector = MetricsCollector(max_iterations=8)

        for i in range(100):
            level = np.random.choice([1, 2, 3, 4])
            correct = np.random.random() < (1.0 - level * 0.15)
            iterations = min(2 + level * 1.5 + np.random.normal(0, 0.5), 8)

            collector.add_sample(
                prediction=f"pred_{i}",
                target=f"target_{i}",
                correct=correct,
                level=level,
                num_vars=level + 2,
                num_steps=level * 2,
                iterations_used=iterations,
                converged=iterations < 7,
                flops=1e9 * iterations / 8,
                latency=0.001 * iterations,
            )

        # Compute all metrics
        accuracy = collector.compute_accuracy_metrics()
        lru = collector.compute_lru_metrics()
        efficiency = collector.compute_efficiency_metrics()

        # Create report
        report = EvaluationReport(
            experiment_name="integration_test",
            accuracy=accuracy,
            lru=lru,
            efficiency=efficiency,
        )

        # Verify all components
        assert accuracy.total == 100
        assert 0 < accuracy.overall_accuracy < 1
        assert len(accuracy.accuracy_by_level) > 0

        assert lru.avg_iterations > 0
        assert 0 <= lru.convergence_ratio <= 1

        assert efficiency.speedup_vs_fixed > 0

        # Test serialization
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            report.to_json(f.name)
            assert Path(f.name).exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
