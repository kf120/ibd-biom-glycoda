# -*- coding: utf-8 -*-
"""Contract for combining per-fold subgroup results.

Subgroup metrics are averaged across folds. A subgroup that cannot be scored in
some folds therefore has a mean built from fewer numbers than its neighbours,
and the folds that drop out are not a random sample -- a subgroup goes
unestimable in exactly the folds where it was smallest or most imbalanced.

The aggregation must therefore make three things observable: how many folds the
subgroup appeared in, how many of those it could be scored in, and how many
participants stood behind it.
"""

import numpy as np
import pytest

from ibd_biom_glycoda.analysis.loco_cv import (
    _fold_average_metric_dicts,
    compute_fold_averaged_subgroup_metrics,
)
from ibd_biom_glycoda.evaluation.subgroup_analysis import ESTIMABLE_KEY, SUPPORT_KEYS


class TestFoldAverageMetricDicts:
    def test_support_counts_are_summed_and_metrics_averaged(self):
        per_fold = [
            {'n': 10, 'n_cases': 4, 'n_controls': 6, ESTIMABLE_KEY: 1.0, 'AUROC': 0.8},
            {'n': 20, 'n_cases': 6, 'n_controls': 14, ESTIMABLE_KEY: 1.0, 'AUROC': 0.6},
        ]

        result = _fold_average_metric_dicts(per_fold)

        assert result['n'] == 30            # summed, not averaged
        assert result['n_cases'] == 10
        assert result['n_controls'] == 20
        assert result['AUROC'] == pytest.approx(0.7)
        assert result['n_folds'] == 2
        assert result['n_folds_estimable'] == 2

    def test_metric_mean_ignores_unestimable_folds_but_records_them(self):
        per_fold = [
            {'n': 10, 'n_cases': 5, 'n_controls': 5, ESTIMABLE_KEY: 1.0, 'AUROC': 0.9},
            {'n': 4, 'n_cases': 0, 'n_controls': 4, ESTIMABLE_KEY: 0.0, 'AUROC': np.nan},
            {'n': 12, 'n_cases': 6, 'n_controls': 6, ESTIMABLE_KEY: 1.0, 'AUROC': 0.7},
        ]

        result = _fold_average_metric_dicts(per_fold)

        # The mean is over the 2 estimable folds, not all 3 ...
        assert result['AUROC'] == pytest.approx(0.8)
        # ... and that denominator is reported rather than left implicit.
        assert result['n_folds'] == 3
        assert result['n_folds_estimable'] == 2
        # Support still counts every participant, including the unscored fold.
        assert result['n'] == 26
        assert result['n_cases'] == 11

    def test_estimable_flag_is_not_leaked_as_a_metric(self):
        per_fold = [{'n': 4, 'n_cases': 2, 'n_controls': 2, ESTIMABLE_KEY: 1.0, 'AUROC': 0.5}]

        result = _fold_average_metric_dicts(per_fold)

        assert ESTIMABLE_KEY not in result

    def test_subgroup_never_estimable_still_reports_its_support(self):
        per_fold = [
            {'n': 3, 'n_cases': 0, 'n_controls': 3, ESTIMABLE_KEY: 0.0, 'AUROC': np.nan},
            {'n': 5, 'n_cases': 0, 'n_controls': 5, ESTIMABLE_KEY: 0.0, 'AUROC': np.nan},
        ]

        result = _fold_average_metric_dicts(per_fold)

        assert result['n'] == 8
        assert result['n_folds_estimable'] == 0
        assert 'AUROC' not in result  # no estimate is offered where none exists

    def test_empty_input_returns_empty(self):
        assert _fold_average_metric_dicts([]) == {}


class TestFoldAveragedSubgroupMetrics:
    @pytest.fixture
    def subgroup_names(self):
        return {
            'age': {0: '<40', 1: '>40'},
            'sex': {0: 'M', 1: 'F'},
            'location': {0: 'UK', 1: 'US'},
        }

    def test_subgroup_unestimable_in_one_fold_is_not_silently_dropped(self, subgroup_names):
        """The defect this contract exists for.

        Fold 1 contains both classes for sex=F; fold 2 contains only controls.
        The subgroup must survive with its support intact and a fold count that
        shows the mean rests on one fold.
        """
        #            fold 1: 4 samples, both sexes mixed | fold 2: sex=F all controls
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 0])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.4, 0.3])
        fold_id = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        #  Z_bin columns are [Sex, Age_bin]
        sex = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        age = np.zeros(8, dtype=int)
        Z_bin = np.column_stack([sex, age])
        V = np.tile(np.array([1.0, 0.0]), (8, 1))

        result = compute_fold_averaged_subgroup_metrics(
            y_true, y_pred, fold_id, Z_bin, V,
            subgroup_var='sex', metrics=['AUROC'], subgroup_names=subgroup_names,
        )

        assert set(result) == {'M', 'F'}
        assert result['F']['n'] == 4            # 2 from each fold
        assert result['F']['n_folds'] == 2
        assert result['F']['n_folds_estimable'] == 1
        assert result['M']['n_folds_estimable'] == 2

    def test_intersection_cells_report_support_for_every_combination(self, subgroup_names):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.9, 0.3, 0.8])
        fold_id = np.array([1, 1, 1, 1])
        # every participant is <40 and male: the other three cells are empty
        Z_bin = np.column_stack([np.zeros(4, dtype=int), np.zeros(4, dtype=int)])
        V = np.tile(np.array([1.0, 0.0]), (4, 1))

        result = compute_fold_averaged_subgroup_metrics(
            y_true, y_pred, fold_id, Z_bin, V,
            subgroup_var='age_sex', metrics=['AUROC'], subgroup_names=subgroup_names,
        )

        assert result['<40']['M']['n'] == 4
        assert result['<40']['M']['n_folds_estimable'] == 1

    def test_support_counts_sum_to_the_participants_supplied(self, subgroup_names):
        rng = np.random.default_rng(3)
        n = 60
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.uniform(0, 1, size=n)
        fold_id = rng.integers(1, 4, size=n)
        Z_bin = np.column_stack([rng.integers(0, 2, size=n), rng.integers(0, 2, size=n)])
        V = np.tile(np.array([1.0, 0.0]), (n, 1))

        result = compute_fold_averaged_subgroup_metrics(
            y_true, y_pred, fold_id, Z_bin, V,
            subgroup_var='sex', metrics=['AUROC'], subgroup_names=subgroup_names,
        )

        assert sum(group['n'] for group in result.values()) == n
        assert sum(group['n_cases'] for group in result.values()) == int(y_true.sum())

    def test_support_keys_are_the_documented_set(self):
        assert SUPPORT_KEYS == ('n', 'n_cases', 'n_controls')
