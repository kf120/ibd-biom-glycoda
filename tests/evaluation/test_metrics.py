"""Tests for scoring metrics."""

import warnings

import numpy as np
import pytest

from ibd_biom_glycoda.evaluation.metrics import (
    ALL_METRIC_NAMES,
    SUPPORT_KEYS,
    compute_scoring_metrics,
    summarize_scoring_metrics,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestComputeScoringMetricsGating:
    @pytest.fixture
    def binary_data(self, rng):
        y_true = rng.integers(0, 2, size=150)
        y_proba = rng.uniform(0, 1, size=150)
        return y_true, y_proba

    @pytest.fixture
    def multiclass_data(self, rng):
        y_true = rng.integers(0, 3, size=150)
        raw = rng.uniform(0, 1, size=(150, 3))
        y_proba = raw / raw.sum(axis=1, keepdims=True)
        return y_true, y_proba

    def test_metrics_none_returns_every_metric_name(self, binary_data):
        y_true, y_proba = binary_data
        result = compute_scoring_metrics(y_true, y_proba, metrics=None)
        assert set(result.keys()) == set(ALL_METRIC_NAMES) | set(SUPPORT_KEYS)

    def test_subset_returns_only_requested_keys_plus_support(self, binary_data):
        """Support counts bypass the metric filter by design: no metric should
        be obtainable without the sample it was computed on."""
        y_true, y_proba = binary_data
        result = compute_scoring_metrics(y_true, y_proba, metrics=['AUROC'])
        assert set(result.keys()) == {'AUROC', *SUPPORT_KEYS}

    @pytest.mark.parametrize(
        "metrics",
        [
            ['AUROC'],
            ['AUPRC'],
            ['LogLoss'],
            ['Brier'],
            ['Sensitivity', 'Specificity', 'MCC', 'BAcc', 'Precision', 'FPR', 'FNR', 'MCR'],
        ],
    )
    def test_binary_gated_subset_matches_full_computation(self, binary_data, metrics):
        y_true, y_proba = binary_data
        full = compute_scoring_metrics(y_true, y_proba, metrics=None)
        subset = compute_scoring_metrics(y_true, y_proba, metrics=metrics)

        assert set(subset.keys()) == set(metrics) | set(SUPPORT_KEYS)
        for key in metrics:
            assert subset[key] == pytest.approx(full[key], abs=1e-12, nan_ok=True)

    @pytest.mark.parametrize(
        "metrics",
        [
            ['AUROC'],
            ['AUPRC'],
            ['Brier'],
            ['Sensitivity', 'Specificity', 'MCC', 'BAcc'],
        ],
    )
    def test_multiclass_gated_subset_matches_full_computation(self, multiclass_data, metrics):
        y_true, y_proba = multiclass_data
        full = compute_scoring_metrics(y_true, y_proba, metrics=None)
        subset = compute_scoring_metrics(y_true, y_proba, metrics=metrics)

        assert set(subset.keys()) == set(metrics) | set(SUPPORT_KEYS)
        for key in metrics:
            assert subset[key] == pytest.approx(full[key], abs=1e-12, nan_ok=True)

    def test_unknown_metric_name_is_silently_ignored(self, binary_data):
        """Matches pre-existing behaviour: unrecognised names are dropped,
        not errors."""
        y_true, y_proba = binary_data
        result = compute_scoring_metrics(y_true, y_proba, metrics=['AUROC', 'NotAMetric'])
        assert set(result.keys()) == {'AUROC', *SUPPORT_KEYS}


class TestSummarizeScoringMetrics:
    """SEM must not be rounded before use as an inverse-variance meta-analysis
    weight, since `compute_pooled_stats` skips SEM <= 0."""

    def test_sem_below_display_precision_is_not_rounded_to_zero(self):
        # True SEM here is under 0.0005, so rounding to 3 dp would collapse it
        # to 0.000 and drop this group from the pool.
        metrics_list = [
            {'AUROC': 0.81000},
            {'AUROC': 0.81002},
            {'AUROC': 0.80999},
            {'AUROC': 0.81001},
            {'AUROC': 0.81000},
        ]
        result = summarize_scoring_metrics(metrics_list)
        assert 0 < result['AUROC']['SEM'] < 0.0005

    def test_mean_and_ci_remain_rounded_for_display(self):
        metrics_list = [{'AUROC': 0.812345}, {'AUROC': 0.809876}, {'AUROC': 0.815432}]
        result = summarize_scoring_metrics(metrics_list)
        assert result['AUROC']['Mean'] == round(result['AUROC']['Mean'], 3)
        lower, upper = result['AUROC']['CI']
        assert lower == round(lower, 3)
        assert upper == round(upper, 3)

    def test_sem_matches_full_precision_scipy_value(self):
        from scipy.stats import sem as scipy_sem

        values = [0.71, 0.79, 0.72, 0.70, 0.815]
        result = summarize_scoring_metrics([{'AUROC': v} for v in values])
        assert result['AUROC']['SEM'] == pytest.approx(scipy_sem(values), abs=1e-12)


class TestSummarizeScoringMetricsSeedOrderIndependence:
    """Metrics must be the union across seeds, not just the first seed's keys,
    or the outcome depends on which seed happens to run first."""

    def test_metric_missing_from_first_seed_is_not_silently_dropped(self):
        # Reading only the first seed's keys would drop 'AUROC' entirely here.
        metrics_list = [{'LogLoss': 0.5}, {'AUROC': 0.8, 'LogLoss': 0.5}]
        with pytest.warns(UserWarning, match="AUROC"):
            result = summarize_scoring_metrics(metrics_list)
        assert 'AUROC' in result
        assert 'LogLoss' in result

    def test_metric_missing_from_a_later_seed_does_not_raise(self):
        # Indexing every seed with the first seed's keys would raise KeyError here.
        metrics_list = [{'AUROC': 0.8, 'LogLoss': 0.5}, {'LogLoss': 0.5}]
        with pytest.warns(UserWarning, match="AUROC"):
            result = summarize_scoring_metrics(metrics_list)
        assert 'AUROC' in result

    def test_result_is_independent_of_seed_order(self):
        forward = [{'LogLoss': 0.5}, {'AUROC': 0.8, 'LogLoss': 0.6}]
        backward = [{'AUROC': 0.8, 'LogLoss': 0.6}, {'LogLoss': 0.5}]
        with pytest.warns(UserWarning):
            result_forward = summarize_scoring_metrics(forward)
        with pytest.warns(UserWarning):
            result_backward = summarize_scoring_metrics(backward)
        assert set(result_forward) == set(result_backward) == {'AUROC', 'LogLoss'}

    def test_metric_present_in_every_seed_raises_no_warning(self):
        metrics_list = [{'AUROC': 0.8}, {'AUROC': 0.81}, {'AUROC': 0.79}]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = summarize_scoring_metrics(metrics_list)
        assert 'AUROC' in result
