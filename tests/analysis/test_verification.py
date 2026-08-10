"""Tests for LOCO result snapshot comparisons."""

import numpy as np
import pandas as pd
import pytest

from ibd_biom_glycoda.analysis.verification import (
    compare_loco_snapshots,
    diff_dataframe,
    diff_nested_results,
    flag_exceeding_threshold,
    load_snapshot,
    snapshot_results,
    summarize_diff_magnitudes,
)


class TestSnapshotRoundtrip:
    def test_snapshot_and_load_nested_structure(self, tmp_path):
        results = {
            'UK': {'test_out': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.85]})}},
        }
        path = tmp_path / "snapshots" / "baseline.joblib"

        snapshot_results(results, path)
        loaded = load_snapshot(path)

        pd.testing.assert_frame_equal(loaded['UK']['test_out']['disease'], results['UK']['test_out']['disease'])

    def test_load_missing_snapshot_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_snapshot(tmp_path / "does_not_exist.joblib")


class TestDiffDataframe:
    def test_numeric_column_delta(self):
        baseline = pd.DataFrame({'Model': ['LR', 'XB'], 'AUROC Mean': [0.850, 0.900]})
        current = pd.DataFrame({'Model': ['LR', 'XB'], 'AUROC Mean': [0.852, 0.900]})

        diff = diff_dataframe(baseline, current, path="cohort/test_out/disease")

        row = diff[(diff['Model'] == 'LR') & (diff['Field'] == 'AUROC Mean')].iloc[0]
        assert row['Delta'] == pytest.approx(0.002, abs=1e-9)
        assert row['AbsDelta'] == pytest.approx(0.002, abs=1e-9)
        assert row['Changed']

        row_xb = diff[(diff['Model'] == 'XB') & (diff['Field'] == 'AUROC Mean')].iloc[0]
        assert row_xb['Delta'] == pytest.approx(0.0, abs=1e-9)
        assert not row_xb['Changed']

    def test_tuple_ci_column_delta(self):
        baseline = pd.DataFrame({'Model': ['LR'], 'Metric': ['AUROC'], 'CI': [(0.80, 0.90)]})
        current = pd.DataFrame({'Model': ['LR'], 'Metric': ['AUROC'], 'CI': [(0.81, 0.92)]})

        diff = diff_dataframe(baseline, current, path="p")

        row = diff[diff['Field'] == 'CI'].iloc[0]
        # max(|0.81-0.80|, |0.92-0.90|) == 0.02
        assert row['AbsDelta'] == pytest.approx(0.02, abs=1e-9)
        assert row['Changed']

    def test_categorical_column_reports_changed_without_numeric_delta(self):
        baseline = pd.DataFrame({'Model': ['LR'], 'Metric': ['AUROC'], 'Heterogeneity': ['Low']})
        current = pd.DataFrame({'Model': ['LR'], 'Metric': ['AUROC'], 'Heterogeneity': ['High']})

        diff = diff_dataframe(baseline, current, path="p")

        row = diff[diff['Field'] == 'Heterogeneity'].iloc[0]
        assert row['Changed']
        assert pd.isna(row['AbsDelta'])

    def test_worst_cohort_style_column_does_not_break_row_alignment(self):
        """Align rows independently of changing non-key string fields."""
        baseline = pd.DataFrame({
            'Model': ['LR'], 'Metric': ['AUROC'], 'Pooled Mean': [0.85], 'Worst Cohort': ['UK'],
        })
        current = pd.DataFrame({
            'Model': ['LR'], 'Metric': ['AUROC'], 'Pooled Mean': [0.86], 'Worst Cohort': ['US'],
        })

        diff = diff_dataframe(baseline, current, path="p")

        assert set(diff['RowStatus']) == {'both'}
        mean_row = diff[diff['Field'] == 'Pooled Mean'].iloc[0]
        assert mean_row['Delta'] == pytest.approx(0.01, abs=1e-9)

    def test_row_present_only_in_current_is_reported(self):
        baseline = pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.85]})
        current = pd.DataFrame({'Model': ['LR', 'XB'], 'AUROC Mean': [0.85, 0.90]})

        diff = diff_dataframe(baseline, current, path="p")

        xb_rows = diff[diff['Model'] == 'XB']
        assert (xb_rows['RowStatus'] == 'current_only').all()
        assert xb_rows[xb_rows['Field'] == 'AUROC Mean'].iloc[0]['Current'] == pytest.approx(0.90)

    def test_both_empty_returns_empty_frame(self):
        diff = diff_dataframe(pd.DataFrame(), pd.DataFrame(), path="p")
        assert diff.empty

    def test_missing_key_columns_raises(self):
        baseline = pd.DataFrame({'SomeValue': [1.0]})
        current = pd.DataFrame({'SomeValue': [2.0]})
        with pytest.raises(ValueError):
            diff_dataframe(baseline, current, path="p")


class TestDiffNestedResults:
    def test_walks_dict_of_dict_of_dataframe(self):
        baseline = {
            'UK': {'test_out': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.85]})}},
            'US': {'test_out': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.80]})}},
        }
        current = {
            'UK': {'test_out': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.852]})}},
            'US': {'test_out': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.80]})}},
        }

        diff = diff_nested_results(baseline, current)

        uk_row = diff[diff['Path'] == 'UK/test_out/disease'].iloc[0]
        assert uk_row['AbsDelta'] == pytest.approx(0.002, abs=1e-9)
        us_row = diff[diff['Path'] == 'US/test_out/disease'].iloc[0]
        assert us_row['AbsDelta'] == pytest.approx(0.0, abs=1e-9)

    def test_key_present_only_in_one_side_is_still_reported(self):
        baseline = {'UK': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.85]})}}
        current = {
            'UK': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.85]})},
            'DE': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.70]})},
        }

        diff = diff_nested_results(baseline, current)

        de_rows = diff[diff['Path'] == 'DE/disease']
        assert (de_rows['RowStatus'] == 'current_only').all()

    def test_unsupported_leaf_type_raises(self):
        with pytest.raises(TypeError):
            diff_nested_results({'a': 1.0}, {'a': 2.0})


class TestSummarizeDiffMagnitudes:
    def test_reports_max_and_median_per_field(self):
        baseline = pd.DataFrame({'Model': ['LR', 'XB'], 'AUROC Mean': [0.85, 0.90], 'LogLoss Mean': [0.30, 0.28]})
        current = pd.DataFrame({'Model': ['LR', 'XB'], 'AUROC Mean': [0.852, 0.905], 'LogLoss Mean': [0.30001, 0.28002]})

        diff = diff_dataframe(baseline, current, path="p")
        summary = summarize_diff_magnitudes(diff)

        assert summary.loc['AUROC Mean', 'max'] == pytest.approx(0.005, abs=1e-9)
        assert summary.loc['LogLoss Mean', 'max'] == pytest.approx(0.00002, abs=1e-9)


class TestFlagExceedingThreshold:
    def test_flags_only_rows_beyond_threshold(self):
        baseline = pd.DataFrame({'Model': ['LR', 'XB'], 'AUROC Mean': [0.85, 0.90]})
        current = pd.DataFrame({'Model': ['LR', 'XB'], 'AUROC Mean': [0.852, 0.950]})

        diff = diff_dataframe(baseline, current, path="p")
        flagged = flag_exceeding_threshold(diff, thresholds={'AUROC': 0.0025})

        lr_row = flagged[flagged['Model'] == 'LR'].iloc[0]
        xb_row = flagged[flagged['Model'] == 'XB'].iloc[0]
        assert not lr_row['Exceeds']
        assert xb_row['Exceeds']

    def test_field_matching_no_threshold_key_is_never_flagged(self):
        baseline = pd.DataFrame({'Model': ['LR'], 'MCC Mean': [0.5]})
        current = pd.DataFrame({'Model': ['LR'], 'MCC Mean': [0.9]})

        diff = diff_dataframe(baseline, current, path="p")
        flagged = flag_exceeding_threshold(diff, thresholds={'AUROC': 0.0025})

        assert not flagged.iloc[0]['Exceeds']

    def test_matches_metric_column_for_long_summary_table(self):
        baseline = pd.DataFrame(
            {'Model': ['LR'], 'Metric': ['AUROC'], 'Pooled Mean': [0.85]}
        )
        current = pd.DataFrame(
            {'Model': ['LR'], 'Metric': ['AUROC'], 'Pooled Mean': [0.90]}
        )

        diff = diff_dataframe(baseline, current, path="p")
        flagged = flag_exceeding_threshold(diff, thresholds={'AUROC': 0.01})

        pooled_mean = flagged[flagged['Field'] == 'Pooled Mean'].iloc[0]
        assert pooled_mean['Threshold'] == pytest.approx(0.01)
        assert pooled_mean['Exceeds']


class TestCompareLocoSnapshots:
    def test_loads_diffs_and_flags_in_one_call(self, tmp_path):
        baseline = {'UK': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.85]})}}
        current = {'UK': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.95]})}}
        baseline_path = tmp_path / "baseline.joblib"
        current_path = tmp_path / "current.joblib"
        snapshot_results(baseline, baseline_path)
        snapshot_results(current, current_path)

        diff, magnitudes, flagged = compare_loco_snapshots(
            baseline_path, current_path, thresholds={'AUROC': 0.01}
        )

        assert diff.loc[diff['Field'] == 'AUROC Mean', 'AbsDelta'].iloc[0] == pytest.approx(0.10, abs=1e-9)
        assert magnitudes.loc['AUROC Mean', 'max'] == pytest.approx(0.10, abs=1e-9)
        assert flagged[flagged['Field'] == 'AUROC Mean']['Exceeds'].iloc[0]

    def test_without_thresholds_nothing_is_flagged(self, tmp_path):
        baseline = {'UK': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.85]})}}
        current = {'UK': {'disease': pd.DataFrame({'Model': ['LR'], 'AUROC Mean': [0.99]})}}
        baseline_path = tmp_path / "baseline.joblib"
        current_path = tmp_path / "current.joblib"
        snapshot_results(baseline, baseline_path)
        snapshot_results(current, current_path)

        _, _, flagged = compare_loco_snapshots(baseline_path, current_path)

        assert not flagged['Exceeds'].any()
