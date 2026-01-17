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
Statistical significance testing utilities for ablation studies.

This module provides rigorous statistical tools for:
1. Bootstrap confidence intervals
2. Paired statistical tests (t-test, Wilcoxon)
3. Effect size calculations (Cohen's d, Cliff's delta)
4. Multiple comparison corrections (Bonferroni, Holm, Benjamini-Hochberg)
5. Power analysis for experiment planning

Usage:
    from verl.utils.statistical_tests import (
        bootstrap_ci,
        paired_comparison,
        multiple_comparison_correction,
        StatisticalReport,
    )

    # Compare two models
    result = paired_comparison(model_a_scores, model_b_scores)
    print(result)

    # Generate full report for ablation study
    report = StatisticalReport(baseline_scores)
    report.add_comparison("mla_only", mla_scores)
    report.add_comparison("mla_lru", lru_scores)
    print(report.generate())
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum


class StatTestType(Enum):
    """Types of statistical tests."""
    PAIRED_TTEST = "paired_t_test"
    WILCOXON = "wilcoxon_signed_rank"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


class EffectSizeType(Enum):
    """Types of effect size measures."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    CLIFFS_DELTA = "cliffs_delta"
    GLASS_DELTA = "glass_delta"


@dataclass
class ConfidenceInterval:
    """Confidence interval result."""
    lower: float
    upper: float
    point_estimate: float
    confidence_level: float
    method: str

    def __str__(self) -> str:
        return f"{self.point_estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}] ({self.confidence_level*100:.0f}% CI, {self.method})"

    def contains(self, value: float) -> bool:
        """Check if value falls within confidence interval."""
        return self.lower <= value <= self.upper


@dataclass
class ComparisonResult:
    """Result of a pairwise statistical comparison."""
    test_type: StatTestType
    statistic: float
    p_value: float
    effect_size: float
    effect_size_type: EffectSizeType
    ci: Optional[ConfidenceInterval] = None
    significant: bool = False
    alpha: float = 0.05

    # Sample statistics
    mean_a: float = 0.0
    mean_b: float = 0.0
    std_a: float = 0.0
    std_b: float = 0.0
    n_samples: int = 0

    def __str__(self) -> str:
        sig_str = "✓" if self.significant else "✗"
        return f"""Comparison Result ({self.test_type.value}):
  Statistic: {self.statistic:.4f}
  P-value: {self.p_value:.6f} {sig_str} (α={self.alpha})
  Effect size ({self.effect_size_type.value}): {self.effect_size:.4f}
  Mean A: {self.mean_a:.4f} ± {self.std_a:.4f}
  Mean B: {self.mean_b:.4f} ± {self.std_b:.4f}
  Difference: {self.mean_b - self.mean_a:+.4f}
  N samples: {self.n_samples}"""


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = "bca",
    random_state: Optional[int] = None,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval.

    Args:
        data: 1D array of observations
        statistic_fn: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95)
        method: Bootstrap method - "percentile", "basic", or "bca" (bias-corrected accelerated)
        random_state: Random seed for reproducibility

    Returns:
        ConfidenceInterval with bounds and point estimate
    """
    if random_state is not None:
        np.random.seed(random_state)

    data = np.asarray(data)
    n = len(data)
    point_estimate = statistic_fn(data)

    # Generate bootstrap samples
    bootstrap_indices = np.random.randint(0, n, size=(n_bootstrap, n))
    bootstrap_samples = data[bootstrap_indices]
    bootstrap_stats = np.array([statistic_fn(s) for s in bootstrap_samples])

    alpha = 1 - confidence_level

    if method == "percentile":
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    elif method == "basic":
        lower = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        upper = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * alpha / 2)

    elif method == "bca":
        # Bias-corrected and accelerated bootstrap
        # Bias correction
        z0 = _norm_ppf(np.mean(bootstrap_stats < point_estimate))

        # Acceleration (jackknife estimate)
        jackknife_stats = np.array([
            statistic_fn(np.delete(data, i)) for i in range(n)
        ])
        jack_mean = np.mean(jackknife_stats)
        num = np.sum((jack_mean - jackknife_stats) ** 3)
        denom = 6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5
        a = num / denom if denom != 0 else 0

        # Adjusted percentiles
        z_alpha_lower = _norm_ppf(alpha / 2)
        z_alpha_upper = _norm_ppf(1 - alpha / 2)

        alpha1 = _norm_cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha2 = _norm_cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))

        lower = np.percentile(bootstrap_stats, 100 * alpha1)
        upper = np.percentile(bootstrap_stats, 100 * alpha2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return ConfidenceInterval(
        lower=lower,
        upper=upper,
        point_estimate=point_estimate,
        confidence_level=confidence_level,
        method=f"bootstrap_{method}",
    )


def paired_ttest(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Perform paired t-test.

    Args:
        a: Scores from model A
        b: Scores from model B
        alpha: Significance level

    Returns:
        ComparisonResult with test statistics
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) != len(b):
        raise ValueError("Arrays must have same length for paired test")

    n = len(a)
    diff = b - a
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    # T-statistic
    t_stat = mean_diff / se_diff if se_diff > 0 else 0

    # P-value (two-tailed)
    df = n - 1
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df))

    # Effect size (Cohen's d for paired samples)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    # Confidence interval for mean difference
    t_critical = _t_ppf(1 - alpha / 2, df)
    ci = ConfidenceInterval(
        lower=mean_diff - t_critical * se_diff,
        upper=mean_diff + t_critical * se_diff,
        point_estimate=mean_diff,
        confidence_level=1 - alpha,
        method="t_distribution",
    )

    return ComparisonResult(
        test_type=StatTestType.PAIRED_TTEST,
        statistic=t_stat,
        p_value=p_value,
        effect_size=cohens_d,
        effect_size_type=EffectSizeType.COHENS_D,
        ci=ci,
        significant=p_value < alpha,
        alpha=alpha,
        mean_a=np.mean(a),
        mean_b=np.mean(b),
        std_a=np.std(a, ddof=1),
        std_b=np.std(b, ddof=1),
        n_samples=n,
    )


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
) -> ComparisonResult:
    """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        a: Scores from model A
        b: Scores from model B
        alpha: Significance level

    Returns:
        ComparisonResult with test statistics
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) != len(b):
        raise ValueError("Arrays must have same length for paired test")

    diff = b - a

    # Remove zeros
    nonzero_mask = diff != 0
    diff_nonzero = diff[nonzero_mask]
    n = len(diff_nonzero)

    if n == 0:
        return ComparisonResult(
            test_type=StatTestType.WILCOXON,
            statistic=0,
            p_value=1.0,
            effect_size=0,
            effect_size_type=EffectSizeType.CLIFFS_DELTA,
            significant=False,
            alpha=alpha,
            mean_a=np.mean(a),
            mean_b=np.mean(b),
            std_a=np.std(a, ddof=1),
            std_b=np.std(b, ddof=1),
            n_samples=len(a),
        )

    # Rank absolute differences
    abs_diff = np.abs(diff_nonzero)
    ranks = _rankdata(abs_diff)

    # Signed ranks
    signed_ranks = np.where(diff_nonzero > 0, ranks, -ranks)

    # W+ (sum of positive ranks)
    w_plus = np.sum(signed_ranks[signed_ranks > 0])
    w_minus = abs(np.sum(signed_ranks[signed_ranks < 0]))

    # Test statistic (smaller of W+ and W-)
    w_stat = min(w_plus, w_minus)

    # Normal approximation for p-value (n > 20)
    if n >= 20:
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w_stat - mean_w) / std_w
        p_value = 2 * _norm_cdf(z)  # Two-tailed
    else:
        # For small samples, use approximation
        # (For production, use exact distribution or scipy)
        mean_w = n * (n + 1) / 4
        std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (w_stat - mean_w) / std_w
        p_value = 2 * _norm_cdf(z)

    # Cliff's delta (non-parametric effect size)
    cliffs_delta = _cliffs_delta(a, b)

    return ComparisonResult(
        test_type=StatTestType.WILCOXON,
        statistic=w_stat,
        p_value=p_value,
        effect_size=cliffs_delta,
        effect_size_type=EffectSizeType.CLIFFS_DELTA,
        significant=p_value < alpha,
        alpha=alpha,
        mean_a=np.mean(a),
        mean_b=np.mean(b),
        std_a=np.std(a, ddof=1),
        std_b=np.std(b, ddof=1),
        n_samples=len(a),
    )


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> ComparisonResult:
    """Perform permutation test for difference in means.

    Args:
        a: Scores from model A
        b: Scores from model B
        n_permutations: Number of permutations
        alpha: Significance level
        random_state: Random seed

    Returns:
        ComparisonResult with test statistics
    """
    if random_state is not None:
        np.random.seed(random_state)

    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) != len(b):
        raise ValueError("Arrays must have same length for paired test")

    n = len(a)
    observed_diff = np.mean(b) - np.mean(a)

    # Combined data for permutation
    combined = np.concatenate([a, b])

    # Generate permutation distribution
    perm_diffs = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = np.random.permutation(combined)
        perm_a = perm[:n]
        perm_b = perm[n:]
        perm_diffs[i] = np.mean(perm_b) - np.mean(perm_a)

    # P-value (two-tailed)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    # Effect size
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    cohens_d = observed_diff / pooled_std if pooled_std > 0 else 0

    # Bootstrap CI for difference
    diff = b - a
    ci = bootstrap_ci(diff, np.mean, confidence_level=1 - alpha, random_state=random_state)

    return ComparisonResult(
        test_type=StatTestType.PERMUTATION,
        statistic=observed_diff,
        p_value=p_value,
        effect_size=cohens_d,
        effect_size_type=EffectSizeType.COHENS_D,
        ci=ci,
        significant=p_value < alpha,
        alpha=alpha,
        mean_a=np.mean(a),
        mean_b=np.mean(b),
        std_a=np.std(a, ddof=1),
        std_b=np.std(b, ddof=1),
        n_samples=n,
    )


def paired_comparison(
    a: np.ndarray,
    b: np.ndarray,
    test_type: str = "auto",
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> ComparisonResult:
    """Perform paired statistical comparison with automatic test selection.

    Args:
        a: Scores from model A (baseline)
        b: Scores from model B (experimental)
        test_type: "auto", "ttest", "wilcoxon", or "permutation"
        alpha: Significance level
        random_state: Random seed for permutation/bootstrap

    Returns:
        ComparisonResult with appropriate test
    """
    a = np.asarray(a)
    b = np.asarray(b)

    if test_type == "auto":
        # Check normality of differences
        diff = b - a
        _, p_normal = _shapiro_wilk_approx(diff)

        if p_normal > 0.05 and len(a) >= 30:
            test_type = "ttest"
        elif len(a) >= 20:
            test_type = "wilcoxon"
        else:
            test_type = "permutation"

    if test_type == "ttest":
        return paired_ttest(a, b, alpha)
    elif test_type == "wilcoxon":
        return wilcoxon_test(a, b, alpha)
    elif test_type == "permutation":
        return permutation_test(a, b, alpha=alpha, random_state=random_state)
    else:
        raise ValueError(f"Unknown test type: {test_type}")


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "holm",
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """Apply multiple comparison correction.

    Args:
        p_values: List of p-values
        method: "bonferroni", "holm", "hochberg", or "fdr_bh" (Benjamini-Hochberg)
        alpha: Family-wise error rate (or FDR for fdr_bh)

    Returns:
        Tuple of (adjusted_p_values, significant_flags)
    """
    n = len(p_values)
    p_values = np.asarray(p_values)

    if method == "bonferroni":
        adjusted = np.minimum(p_values * n, 1.0)
        significant = adjusted < alpha

    elif method == "holm":
        # Holm-Bonferroni step-down
        sorted_idx = np.argsort(p_values)
        adjusted = np.zeros(n)

        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(p_values[idx] * (n - i), 1.0)

        # Ensure monotonicity
        sorted_adjusted = adjusted[sorted_idx]
        for i in range(1, n):
            sorted_adjusted[i] = max(sorted_adjusted[i], sorted_adjusted[i-1])
        adjusted[sorted_idx] = sorted_adjusted

        significant = adjusted < alpha

    elif method == "hochberg":
        # Hochberg step-up
        sorted_idx = np.argsort(p_values)[::-1]  # Descending
        adjusted = np.zeros(n)

        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = min(p_values[idx] * (i + 1), 1.0)

        # Ensure monotonicity (reversed)
        sorted_adjusted = adjusted[sorted_idx]
        for i in range(1, n):
            sorted_adjusted[i] = min(sorted_adjusted[i], sorted_adjusted[i-1])
        adjusted[sorted_idx] = sorted_adjusted

        significant = adjusted < alpha

    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_values)
        adjusted = np.zeros(n)

        for i, idx in enumerate(sorted_idx):
            adjusted[idx] = p_values[idx] * n / (i + 1)

        # Ensure monotonicity (from largest to smallest)
        sorted_adjusted = adjusted[sorted_idx]
        for i in range(n - 2, -1, -1):
            sorted_adjusted[i] = min(sorted_adjusted[i], sorted_adjusted[i + 1])
        adjusted[sorted_idx] = np.minimum(sorted_adjusted, 1.0)

        significant = adjusted < alpha

    else:
        raise ValueError(f"Unknown correction method: {method}")

    return adjusted.tolist(), significant.tolist()


def effect_size_interpretation(d: float, effect_type: EffectSizeType = EffectSizeType.COHENS_D) -> str:
    """Interpret effect size magnitude.

    Args:
        d: Effect size value
        effect_type: Type of effect size measure

    Returns:
        Interpretation string
    """
    d_abs = abs(d)

    if effect_type in [EffectSizeType.COHENS_D, EffectSizeType.HEDGES_G, EffectSizeType.GLASS_DELTA]:
        # Cohen's benchmarks
        if d_abs < 0.2:
            magnitude = "negligible"
        elif d_abs < 0.5:
            magnitude = "small"
        elif d_abs < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
    else:  # Cliff's delta
        if d_abs < 0.147:
            magnitude = "negligible"
        elif d_abs < 0.33:
            magnitude = "small"
        elif d_abs < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"

    direction = "positive" if d > 0 else "negative" if d < 0 else "zero"
    return f"{magnitude} {direction} effect (d={d:.3f})"


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    n: Optional[int] = None,
) -> Dict[str, float]:
    """Power analysis for paired samples.

    Either calculates required sample size for given power,
    or calculates achieved power for given sample size.

    Args:
        effect_size: Expected Cohen's d
        alpha: Significance level
        power: Desired statistical power (default 0.8)
        n: Sample size (if None, calculates required n for power)

    Returns:
        Dictionary with power analysis results
    """
    z_alpha = _norm_ppf(1 - alpha / 2)

    if n is None:
        # Calculate required sample size
        z_beta = _norm_ppf(power)
        n_required = ((z_alpha + z_beta) / effect_size) ** 2
        n_required = int(np.ceil(n_required))

        return {
            "required_n": n_required,
            "effect_size": effect_size,
            "alpha": alpha,
            "target_power": power,
        }
    else:
        # Calculate achieved power
        noncentrality = effect_size * np.sqrt(n)
        achieved_power = 1 - _norm_cdf(z_alpha - noncentrality)

        return {
            "n": n,
            "effect_size": effect_size,
            "alpha": alpha,
            "achieved_power": achieved_power,
        }


@dataclass
class StatisticalReport:
    """Generate comprehensive statistical report for ablation studies."""

    baseline_name: str
    baseline_scores: np.ndarray
    comparisons: Dict[str, ComparisonResult] = field(default_factory=dict)
    alpha: float = 0.05
    correction_method: str = "holm"

    def __init__(
        self,
        baseline_scores: np.ndarray,
        baseline_name: str = "baseline",
        alpha: float = 0.05,
        correction_method: str = "holm",
    ):
        self.baseline_name = baseline_name
        self.baseline_scores = np.asarray(baseline_scores)
        self.comparisons = {}
        self.alpha = alpha
        self.correction_method = correction_method

    def add_comparison(
        self,
        name: str,
        scores: np.ndarray,
        test_type: str = "auto",
    ) -> ComparisonResult:
        """Add a comparison to the report.

        Args:
            name: Name of the experimental condition
            scores: Scores from experimental condition
            test_type: Statistical test to use

        Returns:
            ComparisonResult
        """
        result = paired_comparison(
            self.baseline_scores,
            np.asarray(scores),
            test_type=test_type,
            alpha=self.alpha,
        )
        self.comparisons[name] = result
        return result

    def generate(self) -> str:
        """Generate formatted statistical report.

        Returns:
            Formatted report string
        """
        if not self.comparisons:
            return "No comparisons added to report."

        lines = ["=" * 90]
        lines.append("STATISTICAL ANALYSIS REPORT")
        lines.append("=" * 90)
        lines.append("")

        # Baseline summary
        lines.append(f"Baseline: {self.baseline_name}")
        lines.append(f"  Mean: {np.mean(self.baseline_scores):.4f} ± {np.std(self.baseline_scores, ddof=1):.4f}")
        lines.append(f"  N samples: {len(self.baseline_scores)}")
        ci = bootstrap_ci(self.baseline_scores)
        lines.append(f"  95% CI: [{ci.lower:.4f}, {ci.upper:.4f}]")
        lines.append("")

        # Multiple comparison correction
        p_values = [r.p_value for r in self.comparisons.values()]
        adjusted_p, significant = multiple_comparison_correction(
            p_values, method=self.correction_method, alpha=self.alpha
        )

        # Comparison table
        lines.append("-" * 90)
        header = f"{'Condition':<25} {'Mean':>10} {'Δ':>10} {'p-value':>12} {'adj-p':>12} {'Effect':>10} {'Sig':>5}"
        lines.append(header)
        lines.append("-" * 90)

        for i, (name, result) in enumerate(self.comparisons.items()):
            diff = result.mean_b - result.mean_a
            sig_mark = "***" if adjusted_p[i] < 0.001 else "**" if adjusted_p[i] < 0.01 else "*" if adjusted_p[i] < 0.05 else ""

            row = f"{name:<25} {result.mean_b:>10.4f} {diff:>+10.4f} {result.p_value:>12.6f} {adjusted_p[i]:>12.6f} {result.effect_size:>10.3f} {sig_mark:>5}"
            lines.append(row)

        lines.append("-" * 90)
        lines.append(f"Significance levels: * p<0.05, ** p<0.01, *** p<0.001 (after {self.correction_method} correction)")
        lines.append("")

        # Effect size summary
        lines.append("Effect Size Interpretations:")
        for name, result in self.comparisons.items():
            interp = effect_size_interpretation(result.effect_size, result.effect_size_type)
            lines.append(f"  {name}: {interp}")

        lines.append("")
        lines.append("=" * 90)

        return "\n".join(lines)

    def to_latex(self) -> str:
        """Generate LaTeX table for paper.

        Returns:
            LaTeX formatted table string
        """
        # Multiple comparison correction
        p_values = [r.p_value for r in self.comparisons.values()]
        adjusted_p, _ = multiple_comparison_correction(
            p_values, method=self.correction_method, alpha=self.alpha
        )

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Configuration & Accuracy & $\Delta$ & p-value & Effect Size \\",
            r"\midrule",
        ]

        # Baseline row
        lines.append(f"{self.baseline_name} & {np.mean(self.baseline_scores):.3f} $\\pm$ {np.std(self.baseline_scores, ddof=1):.3f} & --- & --- & --- \\\\")

        # Comparison rows
        for i, (name, result) in enumerate(self.comparisons.items()):
            diff = result.mean_b - result.mean_a
            sig = ""
            if adjusted_p[i] < 0.001:
                sig = "^{***}"
            elif adjusted_p[i] < 0.01:
                sig = "^{**}"
            elif adjusted_p[i] < 0.05:
                sig = "^{*}"

            lines.append(f"{name} & {result.mean_b:.3f} $\\pm$ {result.std_b:.3f} & {diff:+.3f}{sig} & {adjusted_p[i]:.4f} & {result.effect_size:.3f} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\small",
            f"\\item Note: $^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$ after {self.correction_method} correction.",
            r"\end{tablenotes}",
            r"\end{table}",
        ])

        return "\n".join(lines)


# =============================================================================
# Helper functions (pure Python implementations to avoid scipy dependency)
# =============================================================================

def _norm_ppf(p: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Percent point function (inverse CDF) of standard normal distribution."""
    # Approximation using Abramowitz and Stegun formula 26.2.23
    p = np.asarray(p)
    scalar_input = p.ndim == 0
    p = np.atleast_1d(p).astype(float)

    result = np.zeros_like(p)
    result[p <= 0] = float('-inf')
    result[p >= 1] = float('inf')
    result[p == 0.5] = 0.0

    # Handle values in (0, 1)
    mask = (p > 0) & (p < 1) & (p != 0.5)

    # Split into p < 0.5 and p >= 0.5
    p_work = p[mask].copy()
    sign = np.where(p_work < 0.5, -1, 1)
    p_work = np.where(p_work < 0.5, 1 - p_work, p_work)

    t = np.sqrt(-2 * np.log(1 - p_work))

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    result[mask] = sign * (t - (c0 + c1*t + c2*t**2) / (1 + d1*t + d2*t**2 + d3*t**3))

    if scalar_input:
        return float(result[0])
    return result


def _norm_cdf(x: float) -> float:
    """CDF of standard normal distribution."""
    # Error function approximation
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x: float) -> float:
    """Error function approximation."""
    # Horner form of approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911

    sign = 1 if x >= 0 else -1
    x = abs(x)

    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)

    return sign * y


def _t_cdf(t: float, df: int) -> float:
    """CDF of t-distribution (approximation)."""
    # Use normal approximation for large df
    if df >= 30:
        return _norm_cdf(t)

    # For smaller df, use approximation
    x = df / (df + t * t)
    return 1 - 0.5 * _incomplete_beta(df / 2, 0.5, x)


def _t_ppf(p: float, df: int) -> float:
    """Inverse CDF of t-distribution (approximation)."""
    # Use normal approximation for large df
    if df >= 30:
        return _norm_ppf(p)

    # Newton-Raphson iteration
    x = _norm_ppf(p)
    for _ in range(10):
        fx = _t_cdf(x, df) - p
        if abs(fx) < 1e-10:
            break
        # Approximate derivative
        dfx = (_t_cdf(x + 1e-6, df) - _t_cdf(x - 1e-6, df)) / 2e-6
        if dfx == 0:
            break
        x = x - fx / dfx

    return x


def _incomplete_beta(a: float, b: float, x: float) -> float:
    """Incomplete beta function approximation."""
    # Simple approximation for the specific case we need
    if x <= 0:
        return 0
    if x >= 1:
        return 1

    # Use continued fraction for better accuracy
    # This is a simplified version
    result = (x ** a) * ((1 - x) ** b) / a

    # Regularize
    from math import gamma
    beta_ab = gamma(a) * gamma(b) / gamma(a + b)

    return result / beta_ab


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Assign ranks to data, handling ties with average."""
    sorter = np.argsort(x)

    ranks = np.empty_like(sorter, dtype=float)
    ranks[sorter] = np.arange(1, len(x) + 1, dtype=float)

    # Handle ties
    unique_vals, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            mask = inverse == i
            ranks[mask] = np.mean(ranks[mask])

    return ranks


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cliff's delta (non-parametric effect size)."""
    n_a, n_b = len(a), len(b)

    # Count dominance
    greater = 0
    less = 0

    for x in a:
        for y in b:
            if y > x:
                greater += 1
            elif y < x:
                less += 1

    return (greater - less) / (n_a * n_b)


def _shapiro_wilk_approx(data: np.ndarray) -> Tuple[float, float]:
    """Approximate Shapiro-Wilk test for normality.

    Returns (statistic, p_value). This is a simplified approximation.
    For production use, use scipy.stats.shapiro.
    """
    n = len(data)
    if n < 3:
        return 1.0, 1.0

    # Sort data
    x_sorted = np.sort(data)

    # Simplified calculation
    m = np.arange(1, n + 1)
    expected_normal = _norm_ppf((m - 0.375) / (n + 0.25))

    # Correlation with expected normal order statistics
    r = np.corrcoef(x_sorted, expected_normal)[0, 1]
    w = r ** 2

    # P-value approximation (very rough)
    # For n >= 20, W is approximately normal
    if n >= 20:
        mean_w = 0.9  # Approximate expected value under null
        std_w = 0.05  # Approximate std
        z = (w - mean_w) / std_w
        p_value = 2 * (1 - _norm_cdf(abs(z)))
    else:
        # For small samples, use rough heuristic
        p_value = 1.0 if w > 0.9 else 0.05 if w > 0.8 else 0.01

    return w, p_value


# =============================================================================
# Main: Example usage
# =============================================================================

if __name__ == "__main__":
    # Example: Compare ablation configurations
    np.random.seed(42)

    # Simulate scores (in practice, load from experiment results)
    n_samples = 100

    baseline_scores = np.random.normal(0.65, 0.15, n_samples)
    mla_only_scores = baseline_scores + np.random.normal(0.02, 0.05, n_samples)
    mla_lru_scores = baseline_scores + np.random.normal(0.08, 0.06, n_samples)
    mla_lru_no_pos_scores = baseline_scores + np.random.normal(0.05, 0.05, n_samples)

    # Create report
    report = StatisticalReport(baseline_scores, baseline_name="baseline")
    report.add_comparison("mla_only", mla_only_scores)
    report.add_comparison("mla_lru_adaptive", mla_lru_scores)
    report.add_comparison("mla_lru_no_pos_mix", mla_lru_no_pos_scores)

    print(report.generate())
    print("\n" + "=" * 90)
    print("LaTeX Table:")
    print("=" * 90)
    print(report.to_latex())

    # Power analysis
    print("\n" + "=" * 90)
    print("Power Analysis:")
    print("=" * 90)

    # How many samples needed to detect d=0.3 with 80% power?
    power_result = power_analysis(effect_size=0.3, alpha=0.05, power=0.8)
    print(f"To detect d=0.3 with 80% power: n = {power_result['required_n']}")

    # What power do we have with n=100 to detect d=0.3?
    power_result = power_analysis(effect_size=0.3, alpha=0.05, n=100)
    print(f"With n=100, power to detect d=0.3: {power_result['achieved_power']:.2%}")
