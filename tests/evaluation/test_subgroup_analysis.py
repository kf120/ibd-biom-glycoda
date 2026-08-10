"""Tests for subgroup and intersection metrics."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ibd_biom_glycoda.evaluation.subgroup_analysis import (
    calculate_intersection_performance,
    calculate_subgroup_performance,
    plot_between_test_cohort_metrics_subgroup,
)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def two_group_data(rng):
    n = 120
    true_labels = rng.integers(0, 2, size=n)
    pred_proba = rng.uniform(0, 1, size=n)
    # Two roughly balanced subgroups, each with both classes present.
    subgroup_labels = rng.integers(0, 2, size=n)
    return true_labels, pred_proba, subgroup_labels


class TestCalculateSubgroupPerformance:
    def test_default_metrics_include_sensitivity_and_specificity(self, two_group_data):
        true_labels, pred_proba, subgroup_labels = two_group_data

        result = calculate_subgroup_performance(true_labels, pred_proba, subgroup_labels)

        assert len(result) > 0
        for subgroup_metrics in result.values():
            assert 'Sensitivity' in subgroup_metrics
            assert 'Specificity' in subgroup_metrics

    def test_default_metrics_include_resolution_ratio(self, two_group_data):
        true_labels, pred_proba, subgroup_labels = two_group_data

        result = calculate_subgroup_performance(true_labels, pred_proba, subgroup_labels)

        for subgroup_metrics in result.values():
            assert 'Resolution Ratio' in subgroup_metrics
            assert subgroup_metrics['Resolution Ratio'] is not None

    def test_single_class_subgroup_is_skipped(self, rng):
        # Subgroup 1 has only class 0 -> must be excluded, not raise.
        true_labels = np.array([0, 1, 0, 1, 0, 0, 0])
        pred_proba = rng.uniform(0, 1, size=7)
        subgroup_labels = np.array([0, 0, 0, 0, 1, 1, 1])

        result = calculate_subgroup_performance(true_labels, pred_proba, subgroup_labels)

        assert set(result.keys()) == {0}

    def test_subgroup_names_used_as_keys(self, two_group_data):
        true_labels, pred_proba, subgroup_labels = two_group_data
        names = {0: 'Male', 1: 'Female'}

        result = calculate_subgroup_performance(
            true_labels, pred_proba, subgroup_labels, subgroup_names=names
        )

        assert set(result.keys()) <= {'Male', 'Female'}

    def test_explicit_metrics_still_restricts_output(self, two_group_data):
        true_labels, pred_proba, subgroup_labels = two_group_data

        result = calculate_subgroup_performance(
            true_labels, pred_proba, subgroup_labels, metrics=['AUROC']
        )

        for subgroup_metrics in result.values():
            assert set(subgroup_metrics.keys()) == {'AUROC'}


class TestCalculateIntersectionPerformance:
    @pytest.fixture
    def age_sex_data(self, rng):
        n = 200
        true_labels = rng.integers(0, 2, size=n)
        pred_proba = rng.uniform(0, 1, size=n)
        age_labels = rng.integers(0, 2, size=n)
        sex_labels = rng.integers(0, 2, size=n)
        return true_labels, pred_proba, age_labels, sex_labels

    def test_default_metrics_include_sensitivity_and_specificity(self, age_sex_data):
        true_labels, pred_proba, age_labels, sex_labels = age_sex_data

        result = calculate_intersection_performance(
            true_labels, pred_proba, age_labels, sex_labels
        )

        found_any = False
        for sex_dict in result.values():
            for metrics in sex_dict.values():
                found_any = True
                assert 'Sensitivity' in metrics
                assert 'Specificity' in metrics
        assert found_any

    def test_nested_structure_is_age_then_sex(self, age_sex_data):
        true_labels, pred_proba, age_labels, sex_labels = age_sex_data
        age_names = {0: '<40', 1: '>40'}
        sex_names = {0: 'M', 1: 'F'}

        result = calculate_intersection_performance(
            true_labels,
            pred_proba,
            age_labels,
            sex_labels,
            age_subgroup_names=age_names,
            sex_subgroup_names=sex_names,
        )

        assert set(result.keys()) <= {'<40', '>40'}
        for sex_dict in result.values():
            assert set(sex_dict.keys()) <= {'M', 'F'}


class TestPlotBetweenTestCohortMetricsSubgroupModelMatching:
    """Model selection must match the parsed estimator exactly: 'LR' is a
    substring of 'XB+CLR', so substring matching picks the wrong model's row."""

    @staticmethod
    def _meta_summary_with_lr_xb_clr_collision():
        # XB+CLR deliberately precedes LR+CLR: both match a substring 'LR', so
        # this ordering is what exposes the bug via the downstream .iloc[0].
        df_group = pd.DataFrame([
            {'Model': 'XB+CLR', 'Metric': 'AUROC', 'Pooled Mean': 0.99,
             'Pooled SEM': 0.01, 'Lower CI': 0.97, 'Upper CI': 1.00},
            {'Model': 'LR+CLR', 'Metric': 'AUROC', 'Pooled Mean': 0.50,
             'Pooled SEM': 0.01, 'Lower CI': 0.48, 'Upper CI': 0.52},
            {'Model': 'XB+CLR', 'Metric': 'ECE', 'Pooled Mean': 0.10,
             'Pooled SEM': 0.01, 'Lower CI': 0.08, 'Upper CI': 0.12},
            {'Model': 'LR+CLR', 'Metric': 'ECE', 'Pooled Mean': 0.60,
             'Pooled SEM': 0.01, 'Lower CI': 0.58, 'Upper CI': 0.62},
        ])
        return {'subgroups': {'test_out': {'age_sex': {'<40 | M': df_group}}}}

    def test_lr_bar_uses_lr_row_not_xb_clr_collision(self):
        meta_summary = self._meta_summary_with_lr_xb_clr_collision()

        fig, ax, ax2 = plot_between_test_cohort_metrics_subgroup(
            meta_summary,
            model_names=('LR', 'XB'),
            subgroup_var='age_sex',
            metrics=('AUROC', 'ECE'),
            set_name='test_out',
            groups=['<40 | M'],
            preprocessor_names=['CLR'],
        )
        try:
            heights = [p.get_height() for p in ax.patches]
            assert heights == pytest.approx([0.50, 0.99])
        finally:
            plt.close(fig)
