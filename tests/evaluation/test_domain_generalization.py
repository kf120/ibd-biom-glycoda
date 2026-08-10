"""Tests for LOCO domain-generalization plotting helpers."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ibd_biom_glycoda.evaluation.domain_generalization import (
    parse_pipeline,
    plot_between_test_cohort_metrics_overall,
)


class TestParsePipeline:
    def test_plus_separated_label(self):
        assert parse_pipeline('LR+CLR') == ('CLR', 'LR')

    def test_lr_is_not_confused_with_clr(self):
        """'LR' must not be treated as a match for the 'CLR' preprocessor name."""
        preprocessor, estimator = parse_pipeline('XB+CLR')
        assert estimator == 'XB'
        assert preprocessor == 'CLR'


class TestPlotBetweenTestCohortMetricsOverallModelMatching:
    """Model selection must match the parsed estimator exactly: 'LR' is a
    substring of 'XB+CLR', so substring matching picks the wrong model's row."""

    @staticmethod
    def _meta_summary_with_lr_xb_clr_collision():
        # XB+CLR deliberately precedes LR+CLR: both match a substring 'LR', so
        # this ordering is what exposes the bug via the downstream .iloc[0].
        df_set = pd.DataFrame([
            {'Model': 'XB+CLR', 'Metric': 'AUROC', 'Pooled Mean': 0.99,
             'Pooled SEM': 0.01, 'Lower CI': 0.97, 'Upper CI': 1.00},
            {'Model': 'LR+CLR', 'Metric': 'AUROC', 'Pooled Mean': 0.50,
             'Pooled SEM': 0.01, 'Lower CI': 0.48, 'Upper CI': 0.52},
            {'Model': 'XB+CLR', 'Metric': 'LogLoss', 'Pooled Mean': 0.10,
             'Pooled SEM': 0.01, 'Lower CI': 0.08, 'Upper CI': 0.12},
            {'Model': 'LR+CLR', 'Metric': 'LogLoss', 'Pooled Mean': 0.60,
             'Pooled SEM': 0.01, 'Lower CI': 0.58, 'Upper CI': 0.62},
        ])
        return {'disease': {'test_out': df_set}}

    def test_lr_bar_uses_lr_row_not_xb_clr_collision(self):
        meta_summary = self._meta_summary_with_lr_xb_clr_collision()

        fig, ax, ax2 = plot_between_test_cohort_metrics_overall(
            meta_summary,
            model_names=('LR', 'XB'),
            metrics=('AUROC', 'LogLoss'),
            set_name='test_out',
            preprocessor_names=['CLR'],
        )
        try:
            # ax.bar() is called once per model in model_names order (LR, XB),
            # so patches[0] is LR's AUROC bar and patches[1] is XB's.
            heights = [p.get_height() for p in ax.patches]
            assert heights == pytest.approx([0.50, 0.99])
        finally:
            plt.close(fig)

    def test_swapped_model_order_still_selects_correct_rows(self):
        """Requesting XB before LR must not change which row LR's bar uses."""
        meta_summary = self._meta_summary_with_lr_xb_clr_collision()

        fig, ax, ax2 = plot_between_test_cohort_metrics_overall(
            meta_summary,
            model_names=('XB', 'LR'),
            metrics=('AUROC', 'LogLoss'),
            set_name='test_out',
            preprocessor_names=['CLR'],
        )
        try:
            heights = [p.get_height() for p in ax.patches]
            assert heights == pytest.approx([0.99, 0.50])
        finally:
            plt.close(fig)
