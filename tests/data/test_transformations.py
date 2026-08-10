"""Tests for data transformations and glycomotif caching."""

import numpy as np
import pandas as pd
import pytest

from ibd_biom_glycoda.data import transformations as tf
from ibd_biom_glycoda.data.transformations import (
    CLRProcessor,
    GlyCompareProcessor,
    clear_glycompare_excel_cache,
    load_glycompare_data,
)


@pytest.fixture(autouse=True)
def _clear_glycompare_cache():
    clear_glycompare_excel_cache()
    yield
    clear_glycompare_excel_cache()


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.uniform(1, 10, size=(6, 4)),
        columns=['GP1', 'GP2', 'GP3', 'GP4'],
        index=[f's{i}' for i in range(6)],
    )


class TestCLRProcessorFit:
    def test_fit_does_not_run_a_clr_pass(self, monkeypatch, toy_df):
        def _boom(*args, **kwargs):
            raise AssertionError("fit() should not run a CLR pass")

        monkeypatch.setattr(tf, 'apply_clr_transformation', _boom)

        CLRProcessor(gamma=0.1, seed=42).fit(toy_df)  # must not raise

    def test_fit_sets_feature_names_out_from_input_columns(self, toy_df):
        processor = CLRProcessor(gamma=0.1, seed=42).fit(toy_df)
        assert list(processor.get_feature_names_out()) == list(toy_df.columns)

    def test_fit_accepts_plain_ndarray(self):
        X = np.random.default_rng(0).uniform(1, 10, size=(5, 3))
        processor = CLRProcessor(gamma=0.1, seed=42).fit(X)
        # get_feature_names_out() casts to dtype=str (pre-existing behaviour).
        assert list(processor.get_feature_names_out()) == ['0', '1', '2']

    def test_fit_does_not_mutate_global_numpy_rng_state(self, toy_df):
        np.random.seed(123)
        expected_next_draw = np.random.uniform()

        np.random.seed(123)
        CLRProcessor(gamma=0.1, seed=42).fit(toy_df)
        actual_next_draw = np.random.uniform()

        assert actual_next_draw == expected_next_draw

    def test_fit_does_not_touch_glwstats_rng(self, toy_df):
        rng_before = tf.glwstats.rng
        CLRProcessor(gamma=0.1, seed=42).fit(toy_df)
        assert tf.glwstats.rng is rng_before

    def test_transform_output_is_independent_of_whether_fit_was_called(self, toy_df):
        """Transform output is independent of prior fitting."""
        fitted = CLRProcessor(gamma=0.1, seed=42).fit(toy_df)
        out_after_fit = fitted.transform(toy_df)

        unfitted = CLRProcessor(gamma=0.1, seed=42)  # fit() never called
        out_without_fit = unfitted.transform(toy_df)

        np.testing.assert_array_equal(out_after_fit, out_without_fit)

    def test_transform_matches_direct_reference_computation(self, toy_df):
        """Transform matches the direct CLR computation."""
        seed, gamma = 42, 0.1
        tf.glwstats.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        expected = tf.apply_clr_transformation(
            toy_df, group1=toy_df.index.tolist(), group2=None,
            gamma=gamma, custom_scale=None, seed=seed,
        ).values

        result = CLRProcessor(gamma=gamma, seed=seed).fit(toy_df).transform(toy_df)

        np.testing.assert_array_equal(result, expected)


class TestGlyCompareExcelCache:
    @pytest.fixture
    def load_dataset_excel_calls(self, monkeypatch):
        calls = []

        def _fake_load(fname, data_dir=None):
            calls.append((str(fname), data_dir))
            if 'abd' in str(fname):
                return pd.DataFrame({'A': [1.0, 2.0], 'B': [3.0, 4.0]})
            return pd.DataFrame({'Name': ['A', 'B'], 'IUPAC': ['Glycan-A', 'Glycan-B']})

        import ibd_biom_glycoda.data.dataset as dataset_module
        monkeypatch.setattr(dataset_module, 'load_dataset_excel', _fake_load)
        return calls

    @pytest.fixture
    def orig_df(self):
        return pd.DataFrame({'GP1': [1.0, 2.0], 'other_col': ['x', 'y']}, index=['s0', 's1'])

    def test_repeated_calls_with_same_filenames_hit_cache(self, load_dataset_excel_calls, orig_df):
        for _ in range(3):
            load_glycompare_data(orig_df, ['GP1'], 'abd.xlsx', 'annot.xlsx', data_dir='data')

        # Two files, loaded once each, regardless of 3 repeated calls.
        assert len(load_dataset_excel_calls) == 2

    def test_different_filenames_are_not_cached_together(self, load_dataset_excel_calls, orig_df):
        load_glycompare_data(orig_df, ['GP1'], 'abd.xlsx', 'annot.xlsx', data_dir='data')
        load_glycompare_data(orig_df, ['GP1'], 'abd2.xlsx', 'annot2.xlsx', data_dir='data')

        assert len(load_dataset_excel_calls) == 4

    def test_clear_cache_forces_reload(self, load_dataset_excel_calls, orig_df):
        load_glycompare_data(orig_df, ['GP1'], 'abd.xlsx', 'annot.xlsx', data_dir='data')
        clear_glycompare_excel_cache()
        load_glycompare_data(orig_df, ['GP1'], 'abd.xlsx', 'annot.xlsx', data_dir='data')

        assert len(load_dataset_excel_calls) == 4

    def test_returned_frame_is_an_independent_copy(self, load_dataset_excel_calls, orig_df):
        _, X1 = load_glycompare_data(orig_df, ['GP1'], 'abd.xlsx', 'annot.xlsx', data_dir='data')
        X1.iloc[0, 0] = 999.0

        _, X2 = load_glycompare_data(orig_df, ['GP1'], 'abd.xlsx', 'annot.xlsx', data_dir='data')

        assert X2.iloc[0, 0] != 999.0
        assert len(load_dataset_excel_calls) == 2  # still a cache hit

    def test_fresh_processor_instance_reuses_cache(self, load_dataset_excel_calls, orig_df):
        """Simulates loco_cv._safe_clone: every LOCO task gets a fresh
        GlyCompareProcessor instance (X_glycomotif_full reset to None by
        __init__), so without the module-level cache each one's first
        fit() would re-read the Excel files from disk."""
        kwargs = dict(
            orig_df=orig_df, coda_cols=['GP1'],
            abd_fname='abd.xlsx', annot_fname='annot.xlsx', data_dir='data',
        )

        first = GlyCompareProcessor(**kwargs)
        first.fit(None)
        assert len(load_dataset_excel_calls) == 2

        second = GlyCompareProcessor(**kwargs)  # fresh instance
        assert second.X_glycomotif_full is None
        second.fit(None)

        assert len(load_dataset_excel_calls) == 2  # no new disk reads
