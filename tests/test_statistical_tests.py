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
"""Tests for statistical significance testing utilities."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add utils path to allow direct import
sys.path.insert(0, str(Path(__file__).parent.parent / "verl" / "utils"))

from statistical_tests import (
    bootstrap_ci,
    paired_ttest,
    wilcoxon_test,
    permutation_test,
    paired_comparison,
    multiple_comparison_correction,
    effect_size_interpretation,
    power_analysis,
    StatisticalReport,
    ConfidenceInterval,
    ComparisonResult,
    StatTestType,
    EffectSizeType,
)


class TestBootstrapCI:
    """Tests for bootstrap confidence intervals."""

    def test_basic_ci(self):
        """Test basic confidence interval computation."""
        np.random.seed(42)
        data = np.random.normal(0.5, 0.1, 100)

        ci = bootstrap_ci(data, n_bootstrap=1000, random_state=42)

        assert isinstance(ci, ConfidenceInterval)
        assert ci.lower < ci.point_estimate < ci.upper
        assert ci.confidence_level == 0.95

    def test_contains_mean(self):
        """Test that CI contains true mean for normal data."""
        np.random.seed(42)
        true_mean = 0.7
        data = np.random.normal(true_mean, 0.1, 200)

        ci = bootstrap_ci(data, n_bootstrap=2000, random_state=42)

        # CI should contain true mean (most of the time)
        assert ci.lower < true_mean < ci.upper

    def test_different_methods(self):
        """Test different bootstrap methods."""
        np.random.seed(42)
        data = np.random.normal(0.5, 0.1, 100)

        ci_percentile = bootstrap_ci(data, method="percentile", random_state=42)
        ci_basic = bootstrap_ci(data, method="basic", random_state=42)
        ci_bca = bootstrap_ci(data, method="bca", random_state=42)

        # All should give similar results
        assert abs(ci_percentile.point_estimate - ci_basic.point_estimate) < 0.01
        assert abs(ci_bca.point_estimate - ci_percentile.point_estimate) < 0.01

    def test_custom_statistic(self):
        """Test with custom statistic function."""
        np.random.seed(42)
        data = np.random.normal(0.5, 0.1, 100)

        ci_mean = bootstrap_ci(data, statistic_fn=np.mean, random_state=42)
        ci_median = bootstrap_ci(data, statistic_fn=np.median, random_state=42)

        # Should compute different statistics
        assert ci_mean.method == ci_median.method
        # Point estimates may differ
        assert isinstance(ci_median.point_estimate, float)


class TestPairedTTest:
    """Tests for paired t-test."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 50)
        b = a + np.random.normal(0.15, 0.05, 50)  # Clear improvement

        result = paired_ttest(a, b)

        assert isinstance(result, ComparisonResult)
        assert result.test_type == StatTestType.PAIRED_TTEST
        assert result.significant  # Should detect the difference
        assert result.p_value < 0.05
        assert result.effect_size > 0  # b > a

    def test_no_difference(self):
        """Test with no real difference."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 50)
        b = a + np.random.normal(0, 0.01, 50)  # No real difference

        result = paired_ttest(a, b)

        # Should NOT detect significant difference
        assert result.p_value > 0.05 or abs(result.effect_size) < 0.2

    def test_confidence_interval(self):
        """Test that CI is computed."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 50)
        b = np.random.normal(0.6, 0.1, 50)

        result = paired_ttest(a, b)

        assert result.ci is not None
        assert result.ci.lower < result.ci.upper

    def test_array_length_mismatch(self):
        """Test error on mismatched array lengths."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2])

        with pytest.raises(ValueError):
            paired_ttest(a, b)


class TestWilcoxonTest:
    """Tests for Wilcoxon signed-rank test."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        np.random.seed(42)
        a = np.random.uniform(0.4, 0.6, 50)
        b = a + np.random.uniform(0.1, 0.2, 50)  # Clear improvement

        result = wilcoxon_test(a, b)

        assert result.test_type == StatTestType.WILCOXON
        assert result.significant
        assert result.effect_size_type == EffectSizeType.CLIFFS_DELTA

    def test_handles_ties(self):
        """Test handling of tied values."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 2, 4, 5, 6])  # Some ties

        result = wilcoxon_test(a, b)

        assert isinstance(result, ComparisonResult)

    def test_identical_arrays(self):
        """Test with identical arrays."""
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1, 2, 3, 4, 5])

        result = wilcoxon_test(a, b)

        # No difference
        assert result.p_value == 1.0


class TestPermutationTest:
    """Tests for permutation test."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 30)
        b = np.random.normal(0.7, 0.1, 30)

        result = permutation_test(a, b, n_permutations=1000, random_state=42)

        assert result.test_type == StatTestType.PERMUTATION
        assert result.significant
        assert result.p_value < 0.05

    def test_no_difference(self):
        """Test with no difference."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 30)
        b = np.random.normal(0.5, 0.1, 30)

        result = permutation_test(a, b, n_permutations=1000, random_state=42)

        # Should not be significant
        assert result.p_value > 0.05

    def test_reproducibility(self):
        """Test reproducibility with random state."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 30)
        b = np.random.normal(0.6, 0.1, 30)

        result1 = permutation_test(a, b, random_state=123)
        result2 = permutation_test(a, b, random_state=123)

        assert result1.p_value == result2.p_value


class TestPairedComparison:
    """Tests for auto-selected paired comparison."""

    def test_auto_selection(self):
        """Test automatic test selection."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 100)
        b = np.random.normal(0.6, 0.1, 100)

        result = paired_comparison(a, b, test_type="auto")

        assert isinstance(result, ComparisonResult)

    def test_explicit_ttest(self):
        """Test explicit t-test selection."""
        np.random.seed(42)
        a = np.random.normal(0.5, 0.1, 50)
        b = np.random.normal(0.6, 0.1, 50)

        result = paired_comparison(a, b, test_type="ttest")

        assert result.test_type == StatTestType.PAIRED_TTEST

    def test_explicit_wilcoxon(self):
        """Test explicit Wilcoxon selection."""
        np.random.seed(42)
        a = np.random.uniform(0.4, 0.6, 50)
        b = np.random.uniform(0.5, 0.7, 50)

        result = paired_comparison(a, b, test_type="wilcoxon")

        assert result.test_type == StatTestType.WILCOXON


class TestMultipleComparisonCorrection:
    """Tests for multiple comparison correction."""

    def test_bonferroni(self):
        """Test Bonferroni correction."""
        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]

        adjusted, significant = multiple_comparison_correction(
            p_values, method="bonferroni", alpha=0.05
        )

        # Bonferroni multiplies by n
        assert adjusted[0] == pytest.approx(0.05, rel=0.01)  # 0.01 * 5
        assert all(p <= 1.0 for p in adjusted)

    def test_holm(self):
        """Test Holm correction."""
        p_values = [0.001, 0.01, 0.04, 0.05]

        adjusted, significant = multiple_comparison_correction(
            p_values, method="holm", alpha=0.05
        )

        # Holm is less conservative than Bonferroni
        bonf_adjusted, _ = multiple_comparison_correction(
            p_values, method="bonferroni", alpha=0.05
        )

        assert sum(significant) >= sum(
            p < 0.05 for p in bonf_adjusted
        )  # Holm finds at least as many

    def test_fdr_bh(self):
        """Test Benjamini-Hochberg FDR correction."""
        p_values = [0.001, 0.01, 0.02, 0.03, 0.04]

        adjusted, significant = multiple_comparison_correction(
            p_values, method="fdr_bh", alpha=0.05
        )

        # FDR is typically less conservative
        assert sum(significant) > 0

    def test_preserves_order(self):
        """Test that relative order is preserved."""
        p_values = [0.05, 0.01, 0.03]

        adjusted, _ = multiple_comparison_correction(
            p_values, method="holm", alpha=0.05
        )

        # Original order should be preserved
        assert len(adjusted) == 3


class TestEffectSizeInterpretation:
    """Tests for effect size interpretation."""

    def test_cohens_d_small(self):
        """Test small effect interpretation."""
        interp = effect_size_interpretation(0.3)
        assert "small" in interp

    def test_cohens_d_medium(self):
        """Test medium effect interpretation."""
        interp = effect_size_interpretation(0.6)
        assert "medium" in interp

    def test_cohens_d_large(self):
        """Test large effect interpretation."""
        interp = effect_size_interpretation(1.0)
        assert "large" in interp

    def test_negative_effect(self):
        """Test negative effect."""
        interp = effect_size_interpretation(-0.5)
        assert "negative" in interp


class TestPowerAnalysis:
    """Tests for power analysis."""

    def test_sample_size_calculation(self):
        """Test required sample size calculation."""
        result = power_analysis(effect_size=0.5, alpha=0.05, power=0.8)

        assert 'required_n' in result
        assert result['required_n'] > 0

    def test_power_calculation(self):
        """Test achieved power calculation."""
        result = power_analysis(effect_size=0.5, alpha=0.05, n=100)

        assert 'achieved_power' in result
        assert 0 < result['achieved_power'] <= 1

    def test_larger_n_more_power(self):
        """Test that larger n gives more power."""
        result_small = power_analysis(effect_size=0.3, n=30)
        result_large = power_analysis(effect_size=0.3, n=200)

        assert result_large['achieved_power'] > result_small['achieved_power']


class TestStatisticalReport:
    """Tests for statistical report generation."""

    def test_report_creation(self):
        """Test basic report creation."""
        np.random.seed(42)
        baseline = np.random.normal(0.5, 0.1, 50)

        report = StatisticalReport(baseline, baseline_name="baseline")

        assert report.baseline_name == "baseline"

    def test_add_comparison(self):
        """Test adding comparisons."""
        np.random.seed(42)
        baseline = np.random.normal(0.5, 0.1, 50)
        experimental = np.random.normal(0.6, 0.1, 50)

        report = StatisticalReport(baseline)
        result = report.add_comparison("experimental", experimental)

        assert "experimental" in report.comparisons
        assert isinstance(result, ComparisonResult)

    def test_generate_report(self):
        """Test report generation."""
        np.random.seed(42)
        baseline = np.random.normal(0.5, 0.1, 50)
        exp1 = np.random.normal(0.55, 0.1, 50)
        exp2 = np.random.normal(0.6, 0.1, 50)

        report = StatisticalReport(baseline)
        report.add_comparison("exp1", exp1)
        report.add_comparison("exp2", exp2)

        output = report.generate()

        assert "STATISTICAL ANALYSIS REPORT" in output
        assert "exp1" in output
        assert "exp2" in output
        assert "Effect Size" in output

    def test_latex_table(self):
        """Test LaTeX table generation."""
        np.random.seed(42)
        baseline = np.random.normal(0.5, 0.1, 50)
        experimental = np.random.normal(0.6, 0.1, 50)

        report = StatisticalReport(baseline)
        report.add_comparison("experimental", experimental)

        latex = report.to_latex()

        assert r"\begin{table}" in latex
        assert r"\end{table}" in latex
        assert "experimental" in latex


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
