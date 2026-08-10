# -*- coding: utf-8 -*-
"""Snapshot and compare nested LOCO result tables."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Available columns used to align result rows.
DEFAULT_KEY_COLUMNS = ('Model', 'Metric', 'Group')

DIFF_COLUMNS = ['Path', 'Field', 'Baseline', 'Current', 'Delta', 'AbsDelta', 'Changed', 'RowStatus']


def snapshot_results(results, path):
    """Persist a nested dict-of-DataFrames results structure to disk.

    Parameters
    ----------
    results : dict
        Nested results structure (e.g. output of ``summarize_loco_results_all``
        or ``aggregate_loco_results_per_test_cohort``).
    path : str or Path
        Destination file. Parent directories are created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, path)


def load_snapshot(path):
    """Load a result snapshot."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No snapshot found at '{path}'.")
    return joblib.load(path)


def _column_kind(series):
    """Classify a DataFrame column as 'numeric', 'tuple' (e.g. a CI pair), or 'other'."""
    non_null = series.dropna()
    if non_null.empty:
        return 'other'
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    if all(isinstance(v, (tuple, list)) and len(v) == 2 for v in non_null):
        return 'tuple'
    return 'other'


def diff_dataframe(baseline_df, current_df, path, key_columns=DEFAULT_KEY_COLUMNS):
    """Diff two result tables sharing the same row-identifier columns.

    Parameters
    ----------
    baseline_df, current_df : pd.DataFrame
        Tables to compare. Either may be an empty/missing DataFrame if a
        path only exists on one side.
    path : str
        Label identifying where in the nested structure this table lives,
        carried through to the 'Path' column of the report.
    key_columns : sequence of str, default DEFAULT_KEY_COLUMNS
        Column names, if present, used to align rows between the two
        tables. Every other column is treated as a value to diff.

    Returns
    -------
    pd.DataFrame
        One row per (aligned row, value column), with columns
        ``DIFF_COLUMNS``.
    """
    baseline_is_df = isinstance(baseline_df, pd.DataFrame)
    current_is_df = isinstance(current_df, pd.DataFrame)

    if not baseline_is_df and not current_is_df:
        return pd.DataFrame(columns=DIFF_COLUMNS)

    # Preserve columns when one side of the comparison is missing.
    if not baseline_is_df:
        baseline_df = pd.DataFrame(columns=current_df.columns)
    if not current_is_df:
        current_df = pd.DataFrame(columns=baseline_df.columns)

    if baseline_df.empty and current_df.empty:
        return pd.DataFrame(columns=DIFF_COLUMNS)

    key_cols = [c for c in key_columns if c in baseline_df.columns or c in current_df.columns]
    if not key_cols:
        raise ValueError(
            f"No key columns found among {key_columns} for path '{path}'. "
            "Pass explicit key_columns matching this table's identifier columns."
        )

    value_cols_baseline = [c for c in baseline_df.columns if c not in key_cols]
    value_cols_current = [c for c in current_df.columns if c not in key_cols]
    all_value_cols = sorted(set(value_cols_baseline) | set(value_cols_current))

    value_kind = {}
    for col in all_value_cols:
        source = baseline_df if col in baseline_df.columns else current_df
        value_kind[col] = _column_kind(source[col])

    b = baseline_df.rename(columns={c: f'{c}__baseline' for c in value_cols_baseline})
    c = current_df.rename(columns={c: f'{c}__current' for c in value_cols_current})
    merged = b.merge(c, on=key_cols, how='outer', indicator=True)
    merged['_merge'] = merged['_merge'].map({
        'left_only': 'baseline_only', 'right_only': 'current_only', 'both': 'both',
    })

    records = []
    for _, row in merged.iterrows():
        key_values = {k: row[k] for k in key_cols}
        row_status = row['_merge']

        for col in all_value_cols:
            b_col, c_col = f'{col}__baseline', f'{col}__current'
            b_val = row[b_col] if b_col in merged.columns else np.nan
            c_val = row[c_col] if c_col in merged.columns else np.nan
            kind = value_kind[col]

            if kind == 'numeric':
                has_both = pd.notna(b_val) and pd.notna(c_val)
                delta = (c_val - b_val) if has_both else np.nan
                abs_delta = abs(delta) if pd.notna(delta) else np.nan
                changed = bool(has_both and abs_delta > 0)
            elif kind == 'tuple':
                has_both = isinstance(b_val, (tuple, list)) and isinstance(c_val, (tuple, list))
                abs_delta = max(abs(c_val[i] - b_val[i]) for i in range(2)) if has_both else np.nan
                delta = abs_delta
                changed = bool(has_both and abs_delta > 0)
            else:
                has_either = pd.notna(b_val) or pd.notna(c_val)
                changed = bool(has_either and b_val != c_val)
                delta = np.nan
                abs_delta = np.nan

            records.append({
                'Path': path, **key_values, 'Field': col,
                'Baseline': b_val, 'Current': c_val,
                'Delta': delta, 'AbsDelta': abs_delta,
                'Changed': changed, 'RowStatus': row_status,
            })

    return pd.DataFrame(records)


def diff_nested_results(baseline, current, path="", key_columns=DEFAULT_KEY_COLUMNS):
    """Recursively diff two nested dict-of-DataFrames results structures.

    Parameters
    ----------
    baseline, current : dict or pd.DataFrame
        Structures to compare. Dict nodes are matched by key (a key
        present on only one side still produces rows, via the DataFrame
        diff's baseline_only/current_only RowStatus); DataFrame leaves are
        diffed with :func:`diff_dataframe`.
    path : str, default ""
        Internal accumulator for the dict-key path; leave at the default
        when calling directly.
    key_columns : sequence of str, default DEFAULT_KEY_COLUMNS
        Forwarded to :func:`diff_dataframe`.

    Returns
    -------
    pd.DataFrame
        Concatenated diff report across every DataFrame leaf, columns
        ``DIFF_COLUMNS``.
    """
    if isinstance(baseline, pd.DataFrame) or isinstance(current, pd.DataFrame):
        return diff_dataframe(baseline, current, path, key_columns=key_columns)

    if isinstance(baseline, dict) or isinstance(current, dict):
        baseline = baseline if isinstance(baseline, dict) else {}
        current = current if isinstance(current, dict) else {}
        all_keys = list(dict.fromkeys(list(baseline.keys()) + list(current.keys())))

        frames = [
            diff_nested_results(
                baseline.get(key), current.get(key),
                path=f"{path}/{key}" if path else str(key),
                key_columns=key_columns,
            )
            for key in all_keys
        ]
        frames = [f for f in frames if not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DIFF_COLUMNS)

    raise TypeError(
        f"Unsupported leaf type at path '{path}': {type(baseline)!r} / {type(current)!r}. "
        "diff_nested_results only supports dict-of-...-of-DataFrame structures."
    )


def summarize_diff_magnitudes(diff_df):
    """Per-Field count/mean/median/max of AbsDelta, largest drift first.

    A quick way to check the plan's "expected magnitude" claims, e.g.
    "~0.00001 for LogLoss, <=0.0023 for AUROC" -- without eyeballing every row.
    """
    numeric = diff_df.dropna(subset=['AbsDelta'])
    if numeric.empty:
        return pd.DataFrame(columns=['count', 'mean', 'median', 'max'])
    return (
        numeric.groupby('Field')['AbsDelta']
        .agg(['count', 'mean', 'median', 'max'])
        .sort_values('max', ascending=False)
    )


def flag_exceeding_threshold(diff_df, thresholds):
    """Flag rows whose AbsDelta exceeds a per-metric tolerance.

    Parameters
    ----------
    diff_df : pd.DataFrame
        Output of :func:`diff_nested_results` / :func:`diff_dataframe`.
    thresholds : dict
        Maps a metric name (e.g. 'AUROC', 'LogLoss') to a maximum allowed
        AbsDelta. Keys are matched against both the 'Metric' identifier
        column, when present, and the 'Field' column. The latter supports
        wide tables with fields such as 'AUROC Mean'. The first matching
        key wins, so order matters if substrings overlap; rows matching no
        key are never flagged.

    Returns
    -------
    pd.DataFrame
        Copy of ``diff_df`` with added 'Threshold' and 'Exceeds' columns.
    """
    def _threshold_for(row):
        labels = [str(row['Field'])]
        if 'Metric' in row.index and pd.notna(row['Metric']):
            labels.insert(0, str(row['Metric']))
        for key, thr in thresholds.items():
            if any(key in label for label in labels):
                return thr
        return np.nan

    result = diff_df.copy()
    result['Threshold'] = result.apply(_threshold_for, axis=1)
    result['Exceeds'] = (
        result['Threshold'].notna()
        & result['AbsDelta'].notna()
        & (result['AbsDelta'] > result['Threshold'])
    )
    return result


def compare_loco_snapshots(baseline_path, current_path, thresholds=None, key_columns=DEFAULT_KEY_COLUMNS):
    """One-call load + diff + flag for two snapshots saved by :func:`snapshot_results`.

    Parameters
    ----------
    baseline_path, current_path : str or Path
        Snapshot files to compare, as saved by :func:`snapshot_results`.
    thresholds : dict, optional
        Forwarded to :func:`flag_exceeding_threshold`. If omitted, every
        row's 'Exceeds' is False (nothing flagged) -- still useful for
        ``magnitudes``.
    key_columns : sequence of str, default DEFAULT_KEY_COLUMNS
        Forwarded to :func:`diff_nested_results`.

    Returns
    -------
    diff : pd.DataFrame
        Full row-level diff report.
    magnitudes : pd.DataFrame
        Per-Field count/mean/median/max of AbsDelta, largest drift first.
    flagged : pd.DataFrame
        ``diff`` with 'Threshold'/'Exceeds' columns added.
    """
    baseline = load_snapshot(baseline_path)
    current = load_snapshot(current_path)
    diff = diff_nested_results(baseline, current, key_columns=key_columns)
    magnitudes = summarize_diff_magnitudes(diff)
    flagged = flag_exceeding_threshold(diff, thresholds or {})
    return diff, magnitudes, flagged
