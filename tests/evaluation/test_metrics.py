"""Tests for scoring metrics and Brier decomposition."""

import warnings

import numpy as np
import pytest

from ibd_biom_glycoda.evaluation.metrics import (
    ALL_METRIC_NAMES,
    compute_brier_score_decomposition,
    compute_scoring_metrics,
    summarize_scoring_metrics,
)


def _reference_brier_decomposition(y_true, y_proba):
    """Compute Brier components with a loop-based test oracle."""
    from sklearn.metrics import brier_score_loss

    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = brier_score_loss(y_true, y_proba)
    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1 - base_rate)

    unique_preds = np.unique(y_proba)
    reliability = 0.0
    resolution = 0.0

    for pred_val in unique_preds:
        mask = y_proba == pred_val
        n_k = np.sum(mask)
        if n_k == 0:
            continue
        observed_freq = np.mean(y_true[mask])
        reliability += (n_k / len(y_true)) * (pred_val - observed_freq) ** 2
        resolution += (n_k / len(y_true)) * (observed_freq - base_rate) ** 2

    resolution_ratio = resolution / uncertainty if uncertainty > 0 else np.nan
    decomposition_check = uncertainty - resolution + reliability

    return {
        'brier': brier,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'resolution_ratio': resolution_ratio,
        'decomposition_check': decomposition_check,
    }


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestComputeBrierScoreDecomposition:
    def test_matches_reference_loop_with_ties(self, rng):
        y_true = rng.integers(0, 2, size=200)
        # Round probabilities so several samples share exact values (ties),
        # exercising the bincount grouping path.
        y_proba = np.round(rng.uniform(0, 1, size=200), 2)

        result = compute_brier_score_decomposition(y_true, y_proba)
        reference = _reference_brier_decomposition(y_true, y_proba)

        for key in reference:
            assert result[key] == pytest.approx(reference[key], abs=1e-12, nan_ok=True)

    def test_matches_reference_loop_all_unique(self, rng):
        y_true = rng.integers(0, 2, size=50)
        y_proba = rng.uniform(0, 1, size=50)  # continuous -> all unique w.h.p.
        assert len(np.unique(y_proba)) == 50

        result = compute_brier_score_decomposition(y_true, y_proba)
        reference = _reference_brier_decomposition(y_true, y_proba)

        for key in reference:
            assert result[key] == pytest.approx(reference[key], abs=1e-12, nan_ok=True)

    def test_degenerate_when_all_predictions_unique(self, rng):
        """Unique predictions produce singleton-bin degeneracy."""
        y_true = rng.integers(0, 2, size=30)
        y_proba = rng.uniform(0, 1, size=30)

        result = compute_brier_score_decomposition(y_true, y_proba)

        assert result['reliability'] == pytest.approx(result['brier'], abs=1e-9)
        assert result['resolution'] == pytest.approx(result['uncertainty'], abs=1e-9)

    def test_single_unique_probability(self):
        # base rate == the single predicted probability (0.5) so both terms
        # collapse to zero: the one bin's observed frequency equals both the
        # prediction and the base rate.
        y_true = np.array([0, 1, 1, 0])
        y_proba = np.full(4, 0.5)

        result = compute_brier_score_decomposition(y_true, y_proba)

        assert result['reliability'] == pytest.approx(0.0, abs=1e-12)
        assert result['resolution'] == pytest.approx(0.0, abs=1e-12)


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
        assert set(result.keys()) == set(ALL_METRIC_NAMES)

    def test_subset_returns_only_requested_keys(self, binary_data):
        y_true, y_proba = binary_data
        result = compute_scoring_metrics(y_true, y_proba, metrics=['AUROC'])
        assert set(result.keys()) == {'AUROC'}

    @pytest.mark.parametrize(
        "metrics",
        [
            ['AUROC'],
            ['AUPRC'],
            ['LogLoss'],
            ['Brier'],
            ['ECE', 'Reliability', 'Resolution', 'Uncertainty', 'Resolution Ratio'],
            ['Sensitivity', 'Specificity', 'MCC', 'BAcc', 'Precision', 'FPR', 'FNR', 'MCR'],
        ],
    )
    def test_binary_gated_subset_matches_full_computation(self, binary_data, metrics):
        y_true, y_proba = binary_data
        full = compute_scoring_metrics(y_true, y_proba, metrics=None)
        subset = compute_scoring_metrics(y_true, y_proba, metrics=metrics)

        assert set(subset.keys()) == set(metrics)
        for key in metrics:
            assert subset[key] == pytest.approx(full[key], abs=1e-12, nan_ok=True)

    @pytest.mark.parametrize(
        "metrics",
        [
            ['AUROC'],
            ['AUPRC'],
            ['Brier'],
            ['ECE', 'Reliability', 'Resolution', 'Uncertainty', 'Resolution Ratio'],
            ['Sensitivity', 'Specificity', 'MCC', 'BAcc'],
        ],
    )
    def test_multiclass_gated_subset_matches_full_computation(self, multiclass_data, metrics):
        y_true, y_proba = multiclass_data
        full = compute_scoring_metrics(y_true, y_proba, metrics=None)
        subset = compute_scoring_metrics(y_true, y_proba, metrics=metrics)

        assert set(subset.keys()) == set(metrics)
        for key in metrics:
            assert subset[key] == pytest.approx(full[key], abs=1e-12, nan_ok=True)

    def test_resolution_ratio_is_populated_and_correct(self, binary_data):
        y_true, y_proba = binary_data
        result = compute_scoring_metrics(
            y_true, y_proba, metrics=['Resolution Ratio', 'Resolution', 'Uncertainty']
        )
        expected = result['Resolution'] / result['Uncertainty']
        assert result['Resolution Ratio'] == pytest.approx(expected, abs=1e-12)

    def test_unknown_metric_name_is_silently_ignored(self, binary_data):
        """Matches pre-existing behaviour: unrecognised names are dropped,
        not errors."""
        y_true, y_proba = binary_data
        result = compute_scoring_metrics(y_true, y_proba, metrics=['AUROC', 'NotAMetric'])
        assert set(result.keys()) == {'AUROC'}


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
