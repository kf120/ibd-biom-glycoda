# -*- coding: utf-8 -*-
"""Known-answer tests for the conventions inside ``compute_scoring_metrics``.

These conventions are choices, not derivations, and every one of them is
invisible at the call site. Each test below fixes one against a hand-computed
value so that changing it is deliberate:

* what a metric returns when its denominator is empty;
* whether the binary and multiclass branches put Brier on the same scale;
* which side of a 0.5 threshold a probability of exactly 0.5 falls on.
"""

import numpy as np
import pytest

from ibd_biom_glycoda.evaluation.metrics import compute_scoring_metrics, is_lower_better


class TestZeroDenominatorConventions:
    """Empty denominators return 0.0, not NaN.

    A macro average therefore treats "undefined here" as "scored zero here",
    which pulls the average down rather than leaving it undefined. Any subgroup
    table built on these values needs its support counts read alongside.
    """

    def test_precision_is_zero_when_nothing_is_predicted_positive(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])  # every prediction below 0.5

        result = compute_scoring_metrics(
            y_true, y_proba, metrics=['Precision', 'Sensitivity', 'Specificity']
        )

        assert result['Precision'] == 0.0        # tp / (tp + fp) with tp = fp = 0
        assert result['Sensitivity'] == 0.0      # tp / (tp + fn) = 0 / 2
        assert result['Specificity'] == 1.0      # tn / (tn + fp) = 2 / 2

    def test_sensitivity_is_zero_when_there_are_no_positive_cases(self):
        y_true = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.6, 0.2, 0.9])

        result = compute_scoring_metrics(y_true, y_proba, metrics=['Sensitivity', 'FNR'])

        assert result['Sensitivity'] == 0.0
        assert result['FNR'] == 0.0

    def test_hand_computed_confusion_metrics(self):
        #                    below   above   above   below
        y_true = np.array([0,      0,      1,      1])
        y_proba = np.array([0.10,  0.90,   0.80,   0.20])
        # tp = 1, fp = 1, tn = 1, fn = 1
        result = compute_scoring_metrics(
            y_true, y_proba,
            metrics=['Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'MCR'],
        )

        assert result['Sensitivity'] == pytest.approx(0.5)
        assert result['Specificity'] == pytest.approx(0.5)
        assert result['Precision'] == pytest.approx(0.5)
        assert result['FPR'] == pytest.approx(0.5)
        assert result['FNR'] == pytest.approx(0.5)
        assert result['MCR'] == pytest.approx(0.5)


class TestThresholdBoundary:
    def test_probability_of_exactly_one_half_is_classified_negative(self):
        """The binary branch uses ``> 0.5``, so 0.5 itself predicts the control class."""
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])

        result = compute_scoring_metrics(y_true, y_proba, metrics=['Sensitivity', 'Specificity'])

        assert result['Sensitivity'] == 0.0
        assert result['Specificity'] == 1.0


class TestBrierScale:
    """The two branches report Brier on different scales.

    The binary branch uses ``brier_score_loss`` (range 0-1). The multiclass
    branch sums the squared error over class columns (range 0-2). The branch is
    chosen by how many classes the *model* has, so the same two-class data can
    yield either value. Results from the two branches are not comparable.
    """

    def test_binary_branch_matches_the_hand_computed_mean_squared_error(self):
        y_true = np.array([1, 0])
        y_proba = np.array([0.8, 0.3])
        expected = ((1 - 0.8) ** 2 + (0 - 0.3) ** 2) / 2

        result = compute_scoring_metrics(y_true, y_proba, metrics=['Brier'])

        assert result['Brier'] == pytest.approx(expected)

    def test_multiclass_branch_is_the_summed_form_and_is_twice_the_binary_value(self):
        y_true = np.array([1, 0])
        # Same two-class problem, but routed through the multiclass branch by
        # declaring a third model class that never occurs.
        y_proba_2col = np.array([[0.2, 0.8], [0.7, 0.3]])
        y_proba_3col = np.column_stack([y_proba_2col, np.zeros(2)])

        binary = compute_scoring_metrics(y_true, y_proba_2col, metrics=['Brier'])
        multiclass = compute_scoring_metrics(
            y_true, y_proba_3col, metrics=['Brier'], model_classes=[0, 1, 2]
        )

        assert multiclass['Brier'] == pytest.approx(2 * binary['Brier'])


class TestIsLowerBetter:
    @pytest.mark.parametrize("metric", ['LogLoss', 'Brier', 'FPR', 'FNR', 'MCR'])
    def test_loss_like_metrics_are_lower_better(self, metric):
        assert is_lower_better(metric)

    @pytest.mark.parametrize("metric", ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'MCC', 'BAcc'])
    def test_performance_metrics_are_higher_better(self, metric):
        assert not is_lower_better(metric)

    @pytest.mark.parametrize(
        "metric",
        [
            'Calibration Slope',
            'Calibration Intercept',
            'CalibrationSlope',
            'CalibrationIntercept',
        ],
    )
    def test_calibration_targets_are_not_classified_as_lower_better(self, metric):
        """These target a value (1 and 0), not a direction.

        The keyword list must not claim a direction for them, or a plot would
        rank a slope of 0.5 above a slope of 1.0. Both the spaced display form
        and the metric key are covered, since either can reach a plot label.
        """
        assert not is_lower_better(metric)

    def test_model_based_c_statistic_is_not_classified_as_lower_better(self):
        assert not is_lower_better('ModelBasedAUROC')

    def test_npv_is_not_classified_as_lower_better(self):
        assert not is_lower_better('NPV')


class TestRequestedMetricsAreHonoured:
    def test_every_advertised_metric_is_computable_on_a_minimal_binary_input(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=60)
        y_proba = rng.uniform(0, 1, size=60)

        result = compute_scoring_metrics(y_true, y_proba, metrics=None)

        assert result, "no metrics returned"
        assert all(v is not None for v in result.values())

    def test_removed_calibration_metrics_are_no_longer_advertised(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=60)
        y_proba = rng.uniform(0, 1, size=60)

        result = compute_scoring_metrics(y_true, y_proba, metrics=None)

        for removed in ('ECE', 'Reliability', 'Resolution', 'Uncertainty', 'Resolution Ratio'):
            assert removed not in result
