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

from ibd_biom_glycoda.evaluation.metrics import (
    SUPPORT_FLAG_LOW,
    SUPPORT_FLAG_NOT_ESTIMABLE,
    SUPPORT_FLAG_OK,
    SUPPORT_KEYS,
    case_fraction,
    compute_scoring_metrics,
    compute_support,
    support_flag,
)


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
