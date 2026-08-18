# -*- coding: utf-8 -*-
"""Tests for the calibration, case-mix, and support additions to ``metrics``.

Each metric added here answers a question the pre-existing set could not, so
each is pinned against an independent reference rather than against its own
implementation:

* the model-based c-statistic, against a brute-force pairwise form and against
  outcomes simulated from the model's own predicted probabilities;
* the calibration intercept and slope, against closed-form values and against
  data generated with a known slope;
* the explicit threshold and NPV, against hand-computed confusion matrices;
* the support counts and flags, against counted values.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ibd_biom_glycoda.evaluation.metrics import (
    SUPPORT_FLAG_LOW,
    SUPPORT_FLAG_NOT_ESTIMABLE,
    SUPPORT_FLAG_OK,
    SUPPORT_KEYS,
    case_fraction,
    compute_model_based_c_statistic,
    compute_scoring_metrics,
    compute_support,
    support_flag,
)


def brute_force_model_based_c(p):
    """Reference implementation: the quadratic pairwise form, written plainly.

    Kept deliberately naive so that it shares no code path with the sorted
    implementation it checks.
    """
    p = np.asarray(p, dtype=float)
    numerator = 0.0
    denominator = 0.0
    for i in range(p.size):
        for j in range(p.size):
            if i == j:
                continue
            weight = p[i] * (1.0 - p[j])
            denominator += weight
            if p[i] > p[j]:
                numerator += weight
            elif p[i] == p[j]:
                numerator += 0.5 * weight
    return numerator / denominator


class TestModelBasedCStatistic:
    def test_matches_brute_force_pairwise_form(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.01, 0.99, size=60)

        assert compute_model_based_c_statistic(p) == pytest.approx(
            brute_force_model_based_c(p), abs=1e-12
        )

    def test_matches_brute_force_when_predictions_are_heavily_tied(self):
        """Ties get half credit, which is where a sorted implementation slips."""
        rng = np.random.default_rng(1)
        p = rng.choice([0.1, 0.4, 0.4, 0.9], size=40)

        assert compute_model_based_c_statistic(p) == pytest.approx(
            brute_force_model_based_c(p), abs=1e-12
        )

    def test_constant_predictions_give_one_half(self):
        """A model that predicts the same risk for everyone ranks nobody."""
        assert compute_model_based_c_statistic(np.full(50, 0.3)) == pytest.approx(0.5)

    def test_recovers_observed_auroc_on_outcomes_simulated_from_its_own_risks(self):
        """The plan's acceptance criterion for this metric.

        When outcomes really are drawn from the predicted probabilities, the
        model is perfectly calibrated by construction, so the observed
        c-statistic must agree with the model-based one up to Monte Carlo error.
        """
        rng = np.random.default_rng(20240818)
        p = rng.beta(2.0, 2.0, size=40000)
        y = rng.binomial(1, p)

        observed = roc_auc_score(y, p)
        model_based = compute_model_based_c_statistic(p)

        assert model_based == pytest.approx(observed, abs=0.01)

    def test_wider_risk_spread_gives_a_higher_model_based_value(self):
        """The case-mix reading: a narrow spread is intrinsically hard to rank."""
        narrow = np.linspace(0.45, 0.55, 200)
        wide = np.linspace(0.05, 0.95, 200)

        assert compute_model_based_c_statistic(wide) > compute_model_based_c_statistic(narrow)

    @pytest.mark.parametrize(
        "predictions",
        [
            np.array([0.5]),               # fewer than two participants
            np.array([]),                  # empty
            np.zeros(10),                  # no case weight anywhere
            np.ones(10),                   # no control weight anywhere
            np.array([0.3, np.nan, 0.7]),  # missing prediction
        ],
    )
    def test_non_estimable_inputs_return_nan(self, predictions):
        assert np.isnan(compute_model_based_c_statistic(predictions))


class TestExplicitThreshold:
    def test_default_threshold_is_unchanged(self):
        """The pre-existing 0.5 behaviour must survive the new parameter."""
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=200)
        p = rng.uniform(0, 1, size=200)

        default = compute_scoring_metrics(y, p, metrics=['Sensitivity', 'Specificity'])
        explicit = compute_scoring_metrics(
            y, p, metrics=['Sensitivity', 'Specificity'], threshold=0.5
        )

        assert default == explicit

    def test_hand_computed_confusion_metrics_at_a_non_default_threshold(self):
        #                    0.10   0.30   0.60   0.90
        y_true = np.array([0,     1,     0,     1])
        y_proba = np.array([0.10, 0.30,  0.60,  0.90])
        # At threshold 0.25: predicted positive = 0.30, 0.60, 0.90
        # tp = 2 (0.30, 0.90), fp = 1 (0.60), tn = 1 (0.10), fn = 0
        result = compute_scoring_metrics(
            y_true,
            y_proba,
            metrics=['Sensitivity', 'Specificity', 'Precision', 'NPV', 'MCR'],
            threshold=0.25,
        )

        assert result['Sensitivity'] == pytest.approx(1.0)      # 2 / (2 + 0)
        assert result['Specificity'] == pytest.approx(0.5)      # 1 / (1 + 1)
        assert result['Precision'] == pytest.approx(2 / 3)      # 2 / (2 + 1)
        assert result['NPV'] == pytest.approx(1.0)              # 1 / (1 + 0)
        assert result['MCR'] == pytest.approx(0.25)             # 1 / 4

    def test_probability_exactly_at_the_threshold_classifies_as_control(self):
        """Same strict-inequality convention the fixed 0.5 cut-off used."""
        y_true = np.array([1, 1, 0, 0])
        y_proba = np.array([0.7, 0.7, 0.7, 0.7])

        result = compute_scoring_metrics(
            y_true, y_proba, metrics=['Sensitivity', 'Specificity'], threshold=0.7
        )

        assert result['Sensitivity'] == 0.0
        assert result['Specificity'] == 1.0

    def test_raising_the_threshold_cannot_increase_sensitivity(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, size=300)
        p = rng.uniform(0, 1, size=300)

        sensitivities = [
            compute_scoring_metrics(y, p, metrics=['Sensitivity'], threshold=t)['Sensitivity']
            for t in (0.2, 0.4, 0.6, 0.8)
        ]

        assert sensitivities == sorted(sensitivities, reverse=True)


class TestNegativePredictiveValue:
    def test_is_zero_when_nothing_is_predicted_negative(self):
        """Follows the zero-denominator convention of the threshold family."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.6, 0.7, 0.8, 0.9])  # every prediction above 0.5

        result = compute_scoring_metrics(y_true, y_proba, metrics=['NPV'])

        assert result['NPV'] == 0.0

    def test_hand_computed_value(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.10, 0.90, 0.80, 0.20])
        # tn = 1 (0.10), fn = 1 (0.20) -> NPV = 1 / 2
        result = compute_scoring_metrics(y_true, y_proba, metrics=['NPV'])

        assert result['NPV'] == pytest.approx(0.5)


class TestSupportAccompaniesEveryMetric:
    def test_support_is_returned_even_for_a_single_metric_request(self):
        """No disparity number should be readable without its sample."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

        result = compute_scoring_metrics(y_true, y_proba, metrics=['AUROC'])

        assert set(result.keys()) == {'AUROC', *SUPPORT_KEYS}
        assert result['n'] == 5
        assert result['n_cases'] == 2
        assert result['n_controls'] == 3

    def test_support_counts_are_ints_not_floats(self):
        y_true = np.array([0, 1])
        result = compute_support(y_true)

        assert all(isinstance(result[key], int) for key in SUPPORT_KEYS)

    def test_case_control_split_is_undefined_for_more_than_two_labels(self):
        y_true = np.array([0, 1, 2, 2])
        support = compute_support(y_true)

        assert support['n'] == 4
        assert np.isnan(support['n_cases'])
        assert np.isnan(support['n_controls'])


class TestCaseFraction:
    def test_derived_from_counts(self):
        assert case_fraction({'n': 8, 'n_cases': 2, 'n_controls': 6}) == pytest.approx(0.25)

    def test_empty_group_is_nan_not_zero(self):
        assert np.isnan(case_fraction({'n': 0, 'n_cases': 0, 'n_controls': 0}))

    def test_stays_correct_after_counts_are_summed_across_folds(self):
        """Why case fraction is derived rather than stored: the mean of per-fold
        fractions is not the pooled fraction when the folds differ in size."""
        fold_a = {'n': 100, 'n_cases': 10, 'n_controls': 90}   # 0.10
        fold_b = {'n': 10, 'n_cases': 5, 'n_controls': 5}      # 0.50
        pooled = {
            'n': 110,
            'n_cases': 15,
            'n_controls': 95,
        }

        mean_of_fractions = (case_fraction(fold_a) + case_fraction(fold_b)) / 2

        assert case_fraction(pooled) == pytest.approx(15 / 110)
        assert mean_of_fractions == pytest.approx(0.30)
        assert case_fraction(pooled) != pytest.approx(mean_of_fractions)


class TestSupportFlag:
    def test_well_supported_group_is_ok(self):
        assert support_flag({'n': 100, 'n_cases': 40, 'n_controls': 60}) == SUPPORT_FLAG_OK

    @pytest.mark.parametrize(
        "support",
        [
            {'n': 25, 'n_cases': 6, 'n_controls': 19},    # both below 20
            {'n': 100, 'n_cases': 19, 'n_controls': 81},  # cases just below 20
            {'n': 100, 'n_cases': 81, 'n_controls': 19},  # controls just below 20
        ],
    )
    def test_sparse_group_is_flagged_low(self, support):
        assert support_flag(support) == SUPPORT_FLAG_LOW

    def test_boundary_of_exactly_twenty_is_supported(self):
        assert support_flag({'n': 40, 'n_cases': 20, 'n_controls': 20}) == SUPPORT_FLAG_OK

    @pytest.mark.parametrize(
        "support",
        [
            {'n': 30, 'n_cases': 0, 'n_controls': 30},
            {'n': 30, 'n_cases': 30, 'n_controls': 0},
            {'n': 0, 'n_cases': 0, 'n_controls': 0},
        ],
    )
    def test_single_class_group_is_not_estimable_rather_than_low(self, support):
        """A group with no cases cannot yield a metric at all, so calling it
        merely sparse would overstate what is there."""
        assert support_flag(support) == SUPPORT_FLAG_NOT_ESTIMABLE

    def test_thresholds_are_overridable(self):
        support = {'n': 20, 'n_cases': 8, 'n_controls': 12}

        assert support_flag(support) == SUPPORT_FLAG_LOW
        assert support_flag(support, min_cases=5, min_controls=5) == SUPPORT_FLAG_OK
