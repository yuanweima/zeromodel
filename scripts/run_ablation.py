#!/usr/bin/env python
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
Automated ablation study runner for MLA + LRU experiments.

This script orchestrates the complete ablation study including:
1. Running all experiment configurations
2. Collecting and aggregating results
3. Computing statistical significance
4. Generating publication-ready reports

Usage:
    # Run all experiments
    python scripts/run_ablation.py --config verl/trainer/config/lru_trainer.yaml

    # Run specific experiments
    python scripts/run_ablation.py --experiments baseline mla_only mla_lru_adaptive

    # Generate report from existing results
    python scripts/run_ablation.py --report-only --results-dir outputs/ablation

    # Dry run (show what would be run)
    python scripts/run_ablation.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str
    config_overrides: Dict[str, Any]
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    priority: int = 0  # Higher = run first


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    name: str
    seed: int
    metrics: Dict[str, float]
    checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None
    duration_seconds: float = 0.0
    status: str = "pending"  # pending, running, completed, failed


class AblationRunner:
    """Orchestrates the ablation study."""

    def __init__(
        self,
        config_path: str,
        output_dir: str = "outputs/ablation",
        seeds: List[int] = None,
        dry_run: bool = False,
    ):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.seeds = seeds or [42, 123, 456]
        self.dry_run = dry_run

        # Load base configuration
        with open(self.config_path) as f:
            self.base_config = yaml.safe_load(f)

        # Parse experiment definitions
        self.experiments = self._parse_experiments()

        # Results storage
        self.results: Dict[str, List[ExperimentResult]] = {}

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.output_dir / f"ablation_{datetime.now():%Y%m%d_%H%M%S}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _parse_experiments(self) -> Dict[str, ExperimentConfig]:
        """Parse experiment definitions from config."""
        experiments = {}

        exp_defs = self.base_config.get('experiments', {})
        for name, config in exp_defs.items():
            experiments[name] = ExperimentConfig(
                name=name,
                description=config.get('description', ''),
                config_overrides=config,
            )

        return experiments

    def _merge_config(
        self,
        base: Dict,
        overrides: Dict,
    ) -> Dict:
        """Deep merge configuration dictionaries."""
        result = base.copy()

        for key, value in overrides.items():
            if key == 'description':
                continue  # Skip description field
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value

        return result

    def _generate_experiment_config(
        self,
        exp: ExperimentConfig,
        seed: int,
    ) -> Dict:
        """Generate full config for an experiment run."""
        # Start with base config (excluding experiments section)
        config = {k: v for k, v in self.base_config.items() if k != 'experiments'}

        # Merge experiment overrides
        config = self._merge_config(config, exp.config_overrides)

        # Set seed
        if 'training' not in config:
            config['training'] = {}
        config['training']['seed'] = seed

        # Set output directory
        exp_output = self.output_dir / exp.name / f"seed_{seed}"
        config['training']['output_dir'] = str(exp_output)

        # Update wandb name
        if 'logging' in config and 'wandb' in config['logging']:
            config['logging']['wandb']['name'] = f"{exp.name}_s{seed}"

        return config

    def run_single_experiment(
        self,
        exp: ExperimentConfig,
        seed: int,
    ) -> ExperimentResult:
        """Run a single experiment with given seed."""
        self.logger.info(f"Running experiment: {exp.name} (seed={seed})")
        self.logger.info(f"  Description: {exp.description}")

        result = ExperimentResult(
            name=exp.name,
            seed=seed,
            metrics={},
            status="running",
        )

        if self.dry_run:
            self.logger.info("  [DRY RUN] Would run experiment")
            result.status = "skipped"
            return result

        try:
            # Generate config
            config = self._generate_experiment_config(exp, seed)

            # Save config to temp file
            exp_dir = self.output_dir / exp.name / f"seed_{seed}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            config_file = exp_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            result.log_path = str(exp_dir / "train.log")

            # Run training
            start_time = datetime.now()

            # Build command
            cmd = [
                "python", "-m", "verl.trainer.lru.train",
                "--config", str(config_file),
            ]

            self.logger.info(f"  Command: {' '.join(cmd)}")

            # Execute with timeout
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600 * 24,  # 24 hour timeout
                cwd=str(PROJECT_ROOT),
            )

            duration = (datetime.now() - start_time).total_seconds()
            result.duration_seconds = duration

            if process.returncode != 0:
                self.logger.error(f"  Experiment failed: {process.stderr[-500:]}")
                result.status = "failed"
                # Save error log
                with open(exp_dir / "error.log", 'w') as f:
                    f.write(process.stderr)
                return result

            # Load results from checkpoint
            checkpoint_dir = exp_dir / "checkpoints"
            if checkpoint_dir.exists():
                result.checkpoint_path = str(checkpoint_dir / "best_model")

            # Load metrics from eval results
            eval_file = exp_dir / "eval_results.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    result.metrics = json.load(f)

            result.status = "completed"
            self.logger.info(f"  Completed in {duration:.1f}s")

        except subprocess.TimeoutExpired:
            self.logger.error(f"  Experiment timed out")
            result.status = "timeout"
        except Exception as e:
            self.logger.error(f"  Experiment error: {e}")
            result.status = "failed"

        return result

    def run_all_experiments(
        self,
        experiment_names: Optional[List[str]] = None,
        parallel: int = 1,
    ):
        """Run all (or selected) experiments."""
        # Filter experiments
        if experiment_names:
            experiments = {n: e for n, e in self.experiments.items() if n in experiment_names}
        else:
            experiments = self.experiments

        # Sort by priority
        sorted_exps = sorted(experiments.values(), key=lambda e: -e.priority)

        total_runs = len(sorted_exps) * len(self.seeds)
        self.logger.info(f"Starting ablation study: {len(sorted_exps)} experiments × {len(self.seeds)} seeds = {total_runs} runs")

        completed = 0
        for exp in sorted_exps:
            self.results[exp.name] = []

            for seed in self.seeds:
                result = self.run_single_experiment(exp, seed)
                self.results[exp.name].append(result)

                completed += 1
                self.logger.info(f"Progress: {completed}/{total_runs}")

                # Save intermediate results
                self._save_results()

        self.logger.info("Ablation study complete!")

    def _save_results(self):
        """Save current results to file."""
        results_file = self.output_dir / "results.json"

        data = {}
        for name, results in self.results.items():
            data[name] = [
                {
                    'seed': r.seed,
                    'metrics': r.metrics,
                    'status': r.status,
                    'duration': r.duration_seconds,
                    'checkpoint': r.checkpoint_path,
                }
                for r in results
            ]

        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)

    def load_results(self, results_dir: Optional[str] = None):
        """Load results from previous run."""
        results_dir = Path(results_dir) if results_dir else self.output_dir
        results_file = results_dir / "results.json"

        if not results_file.exists():
            self.logger.error(f"Results file not found: {results_file}")
            return

        with open(results_file) as f:
            data = json.load(f)

        self.results = {}
        for name, results in data.items():
            self.results[name] = [
                ExperimentResult(
                    name=name,
                    seed=r['seed'],
                    metrics=r.get('metrics', {}),
                    status=r.get('status', 'unknown'),
                    duration_seconds=r.get('duration', 0),
                    checkpoint_path=r.get('checkpoint'),
                )
                for r in results
            ]

    def generate_report(self) -> str:
        """Generate comprehensive ablation study report."""
        lines = []
        lines.append("=" * 100)
        lines.append("ABLATION STUDY REPORT")
        lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}")
        lines.append("=" * 100)
        lines.append("")

        # Summary table
        lines.append("EXPERIMENT SUMMARY")
        lines.append("-" * 100)

        header = f"{'Experiment':<30} {'Seeds':>6} {'Acc Mean':>10} {'Acc Std':>10} {'Iter Mean':>10} {'Status':>12}"
        lines.append(header)
        lines.append("-" * 100)

        import numpy as np

        for name, results in self.results.items():
            completed = [r for r in results if r.status == 'completed']
            n_completed = len(completed)

            if completed:
                accs = [r.metrics.get('accuracy', 0) for r in completed]
                iters = [r.metrics.get('avg_iterations', 0) for r in completed]

                acc_mean = np.mean(accs)
                acc_std = np.std(accs)
                iter_mean = np.mean(iters)

                status = f"{n_completed}/{len(results)} done"
            else:
                acc_mean = 0
                acc_std = 0
                iter_mean = 0
                status = "No results"

            row = f"{name:<30} {n_completed:>6} {acc_mean:>10.1%} {acc_std:>10.4f} {iter_mean:>10.2f} {status:>12}"
            lines.append(row)

        lines.append("-" * 100)
        lines.append("")

        # Try to compute statistical significance
        try:
            from verl.utils.statistical_tests import StatisticalReport, multiple_comparison_correction

            # Find baseline
            baseline_name = 'baseline' if 'baseline' in self.results else list(self.results.keys())[0]
            baseline_results = self.results.get(baseline_name, [])
            baseline_accs = [r.metrics.get('accuracy', 0) for r in baseline_results if r.status == 'completed']

            if len(baseline_accs) >= 2:
                lines.append("STATISTICAL ANALYSIS")
                lines.append("-" * 100)

                report = StatisticalReport(np.array(baseline_accs), baseline_name=baseline_name)

                for name, results in self.results.items():
                    if name == baseline_name:
                        continue
                    accs = [r.metrics.get('accuracy', 0) for r in results if r.status == 'completed']
                    if len(accs) >= 2:
                        report.add_comparison(name, np.array(accs))

                lines.append(report.generate())
                lines.append("")

        except ImportError:
            lines.append("(Statistical analysis unavailable - missing dependencies)")

        lines.append("=" * 100)

        return "\n".join(lines)

    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        import numpy as np

        lines = [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Ablation Study Results on Causal Loop Task}",
            r"\label{tab:ablation-full}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Configuration & Params & Accuracy & $\Delta$ Acc & Avg Iter & Conv. Rate & Speedup \\",
            r"\midrule",
        ]

        # Find baseline for comparison
        baseline_name = 'baseline' if 'baseline' in self.results else None
        baseline_acc = None

        if baseline_name and self.results.get(baseline_name):
            completed = [r for r in self.results[baseline_name] if r.status == 'completed']
            if completed:
                baseline_acc = np.mean([r.metrics.get('accuracy', 0) for r in completed])

        for name, results in self.results.items():
            completed = [r for r in results if r.status == 'completed']
            if not completed:
                continue

            accs = [r.metrics.get('accuracy', 0) for r in completed]
            iters = [r.metrics.get('avg_iterations', 0) for r in completed]
            conv = [r.metrics.get('convergence_ratio', 0) for r in completed]
            params = completed[0].metrics.get('params', '---')

            acc_mean = np.mean(accs)
            acc_std = np.std(accs)
            iter_mean = np.mean(iters)
            conv_mean = np.mean(conv)

            # Delta vs baseline
            if baseline_acc is not None and name != baseline_name:
                delta = acc_mean - baseline_acc
                delta_str = f"{delta:+.1%}"
            else:
                delta_str = "---"

            # Speedup (8 / avg_iter for fixed max_iter=8)
            speedup = 8 / iter_mean if iter_mean > 0 else 1.0

            row = f"{name} & {params} & {acc_mean:.1%} $\\pm$ {acc_std:.3f} & {delta_str} & {iter_mean:.2f} & {conv_mean:.1%} & {speedup:.2f}$\\times$ \\\\"
            lines.append(row)

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run MLA+LRU ablation study")

    parser.add_argument(
        "--config",
        type=str,
        default="verl/trainer/config/lru_trainer.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Generate report from existing results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory with existing results (for --report-only)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX table",
    )

    args = parser.parse_args()

    # Create runner
    runner = AblationRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        seeds=args.seeds,
        dry_run=args.dry_run,
    )

    if args.report_only:
        # Load and report existing results
        runner.load_results(args.results_dir or args.output_dir)
        print(runner.generate_report())

        if args.latex:
            print("\n" + "=" * 80)
            print("LaTeX Table:")
            print("=" * 80)
            print(runner.generate_latex_table())
    else:
        # Run experiments
        runner.run_all_experiments(args.experiments)

        # Generate report
        print(runner.generate_report())

        if args.latex:
            print("\n" + runner.generate_latex_table())

        # Save report
        report_file = Path(args.output_dir) / "report.txt"
        with open(report_file, 'w') as f:
            f.write(runner.generate_report())
        print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
