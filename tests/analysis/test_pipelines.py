"""Tests for preprocessing, fitting, and estimator construction."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from ibd_biom_glycoda.analysis.pipelines import (
    build_results_dict,
    fit_pipeline_experiment,
    make_estimators,
    preprocess_fold_data,
    run_pipeline_experiment,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _make_split(rng, n, n_features=5, n_cohorts=2):
    return {
        'X': rng.normal(size=(n, n_features)),
        'Z': rng.normal(size=(n, 2)),
        'Z_bin': np.column_stack([rng.integers(0, 2, size=n), rng.integers(0, 2, size=n)]),
        'y': rng.integers(0, 2, size=n),
        'V': np.eye(n_cohorts)[rng.integers(0, n_cohorts, size=n)],
    }


@pytest.fixture
def data_dict(rng):
    n_train, n_val, n_test_in, n_test_out = 60, 10, 10, 10
    splits = {
        'train': _make_split(rng, n_train),
        'val_in': _make_split(rng, n_val),
        'test_in': _make_split(rng, n_test_in),
        'test_out': _make_split(rng, n_test_out),
    }
    # LogisticRegression needs both classes in the training fold.
    splits['train']['y'][:2] = [0, 1]

    d = {}
    for split_name, split in splits.items():
        d[f'X_{split_name}'] = split['X']
        d[f'y_{split_name}'] = split['y']
        d[f'Z_{split_name}'] = split['Z']
        d[f'Z_{split_name}_bin'] = split['Z_bin']
        d[f'V_{split_name}'] = split['V']
    d['age_ranges'] = [0, 40, 100]
    d['cohort_locations'] = [['A', 'B']]
    return d


class _CountingScaler(BaseEstimator, TransformerMixin):
    """RobustScaler-like passthrough that counts fit() calls, to verify
    preprocessing runs once per fold rather than once per estimator."""

    def __init__(self):
        self.fit_calls = 0

    def fit(self, X, y=None):
        self.fit_calls += 1
        self._mean = np.mean(X, axis=0)
        return self

    def transform(self, X, y=None):
        return X - self._mean


class TestBuildResultsDict:
    def test_includes_test_in_and_test_out_bin_covariates(self, rng):
        data = {
            'train': _make_split(rng, 20),
            'val_in': _make_split(rng, 5),
            'test_in': _make_split(rng, 5),
            'test_out': _make_split(rng, 5),
        }
        predictions = {split: rng.uniform(size=(d['y'].shape[0], 2)) for split, d in data.items()}
        metadata = {'age_ranges': [0, 40, 100], 'cohort_locations': [['A', 'B']]}

        result = build_results_dict(
            estimator=None, preprocessor=None, data=data, predictions=predictions, metadata=metadata
        )

        assert 'Z_test_in_bin' in result
        assert 'Z_test_out_bin' in result
        np.testing.assert_array_equal(result['Z_test_in_bin'], data['test_in']['Z_bin'])
        np.testing.assert_array_equal(result['Z_test_out_bin'], data['test_out']['Z_bin'])

    def test_no_longer_carries_a_subgroup_metrics_placeholder(self, rng):
        data = {
            'train': _make_split(rng, 20),
            'val_in': _make_split(rng, 5),
            'test_in': _make_split(rng, 5),
            'test_out': _make_split(rng, 5),
        }
        predictions = {split: rng.uniform(size=(d['y'].shape[0], 2)) for split, d in data.items()}
        metadata = {'age_ranges': [0, 40, 100], 'cohort_locations': [['A', 'B']]}

        result = build_results_dict(
            estimator=None, preprocessor=None, data=data, predictions=predictions, metadata=metadata
        )

        assert 'subgroup_metrics' not in result

    def test_does_not_retain_unused_raw_matrices_or_scalars(self, rng):
        """Results omit unused raw matrices and derived scalars."""
        data = {
            'train': _make_split(rng, 20),
            'val_in': _make_split(rng, 5),
            'test_in': _make_split(rng, 5),
            'test_out': _make_split(rng, 5),
        }
        predictions = {split: rng.uniform(size=(d['y'].shape[0], 2)) for split, d in data.items()}
        metadata = {'age_ranges': [0, 40, 100], 'cohort_locations': [['A', 'B']]}

        result = build_results_dict(
            estimator=None, preprocessor=None, data=data, predictions=predictions, metadata=metadata
        )

        for split in ('train', 'val_in', 'test_in', 'test_out'):
            assert f'X_{split}' not in result
            assert f'Z_{split}' not in result  # continuous Z; Z_{split}_bin is kept
        assert 'num_locations' not in result
        assert 'num_age_bins' not in result

    def test_still_retains_model_processor_and_binned_covariates(self, rng):
        """Results retain fitted objects and subgroup covariates."""
        data = {
            'train': _make_split(rng, 20),
            'val_in': _make_split(rng, 5),
            'test_in': _make_split(rng, 5),
            'test_out': _make_split(rng, 5),
        }
        predictions = {split: rng.uniform(size=(d['y'].shape[0], 2)) for split, d in data.items()}
        metadata = {'age_ranges': [0, 40, 100], 'cohort_locations': [['A', 'B']]}
        sentinel_model, sentinel_proc = object(), object()

        result = build_results_dict(
            estimator=sentinel_model, preprocessor=sentinel_proc, data=data,
            predictions=predictions, metadata=metadata,
        )

        assert result['model'] is sentinel_model
        assert result['processor'] is sentinel_proc
        for split in ('train', 'val_in', 'test_in', 'test_out'):
            assert f'Z_{split}_bin' in result
            assert f'V_{split}' in result


class TestRunPipelineExperiment:
    def test_does_not_call_subgroup_computation_and_returns_bin_covariates(self, data_dict):
        result = run_pipeline_experiment(
            preprocessor=RobustScaler(),
            estimator=LogisticRegression(max_iter=1000),
            data_dict=data_dict,
            seed=0,
        )

        assert 'subgroup_metrics' not in result
        for split in ('train', 'val_in', 'test_in', 'test_out'):
            assert f'Z_{split}_bin' in result
            assert f'V_{split}' in result
            assert len(result[f'{split}_true']) == len(result[f'Z_{split}_bin'])


class TestPreprocessFoldDataSharedAcrossEstimators:
    """Tests reuse of preprocessed fold data across estimators."""

    def test_preprocessor_fits_once_across_two_estimators(self, data_dict):
        scaler = _CountingScaler()
        prepared = preprocess_fold_data(scaler, data_dict, seed=0)

        fit_pipeline_experiment(prepared, LogisticRegression(max_iter=1000, random_state=0), seed=0)
        fit_pipeline_experiment(prepared, LogisticRegression(max_iter=1000, random_state=1), seed=0)

        assert scaler.fit_calls == 1

    def test_split_calls_match_combined_run_pipeline_experiment(self, data_dict):
        """Split and combined pipeline APIs produce identical results."""
        seed = 0
        combined = run_pipeline_experiment(
            preprocessor=RobustScaler(),
            estimator=LogisticRegression(max_iter=1000, random_state=seed),
            data_dict=data_dict,
            seed=seed,
        )

        prepared = preprocess_fold_data(RobustScaler(), data_dict, seed=seed)
        split_result = fit_pipeline_experiment(
            prepared, LogisticRegression(max_iter=1000, random_state=seed), seed=seed
        )

        np.testing.assert_array_equal(combined['test_out_pred'], split_result['test_out_pred'])
        np.testing.assert_array_equal(combined['test_in_pred'], split_result['test_in_pred'])
        np.testing.assert_array_equal(combined['train_pred'], split_result['train_pred'])

    def test_fit_pipeline_experiment_reseeds_for_deterministic_fits(self, data_dict):
        """Repeated fits start from identical RNG state."""
        scaler = _CountingScaler()
        prepared = preprocess_fold_data(scaler, data_dict, seed=0)

        result_a = fit_pipeline_experiment(prepared, LogisticRegression(max_iter=1000, random_state=0), seed=0)
        result_b = fit_pipeline_experiment(prepared, LogisticRegression(max_iter=1000, random_state=0), seed=0)

        np.testing.assert_array_equal(result_a['test_out_pred'], result_b['test_out_pred'])

    def test_prepared_bundle_drops_unused_diagnostic_fields(self, data_dict):
        """Prepared data omits unused diagnostic fields."""
        prepared = preprocess_fold_data(RobustScaler(), data_dict, seed=0)

        assert 'feature_dimensions' not in prepared
        assert 'groups' not in prepared
        # sample_weights is still needed internally by fit_pipeline_experiment.
        assert 'sample_weights' in prepared

    def test_fit_result_drops_unused_diagnostic_fields(self, data_dict):
        result = run_pipeline_experiment(
            preprocessor=RobustScaler(),
            estimator=LogisticRegression(max_iter=1000),
            data_dict=data_dict,
            seed=0,
        )

        for key in ('feature_dimensions', 'groups', 'use_sample_weights', 'sample_weights'):
            assert key not in result


class TestMakeEstimatorsXGBoostThreading:
    """Tests XGBoost thread limits within parallel LOCO runs."""

    def test_xgboost_defaults_to_a_single_thread(self):
        parameters = {'LR': {'max_iter': 1000}, 'XB': {'n_estimators': 5}}
        estimators = make_estimators(parameters, seed=0)

        assert isinstance(estimators['XB'], XGBClassifier)
        assert estimators['XB'].n_jobs == 1

    def test_user_supplied_n_jobs_cannot_override_the_reproducibility_pin(self):
        """n_jobs=1 is a reproducibility pin, not a performance default, so it must
        win even when 'XB' params request more threads."""
        parameters = {'LR': {'max_iter': 1000}, 'XB': {'n_estimators': 5, 'n_jobs': 4}}
        estimators = make_estimators(parameters, seed=0)

        assert estimators['XB'].n_jobs == 1

    def test_logistic_regression_still_gets_configured_params_and_seed(self):
        parameters = {'LR': {'max_iter': 1000, 'C': 0.5}, 'XB': {'n_estimators': 5}}
        estimators = make_estimators(parameters, seed=7)

        assert estimators['LR'].C == 0.5
        assert estimators['LR'].random_state == 7
        assert estimators['XB'].random_state == 7


class TestMakeEstimatorsSelectiveConstruction:
    """An LR-only run must not require an unused 'XB' entry in parameters,
    and vice versa."""

    def test_model_keys_none_still_builds_every_recognised_model(self):
        parameters = {'LR': {'max_iter': 1000}, 'XB': {'n_estimators': 5}}
        estimators = make_estimators(parameters, seed=0)

        assert set(estimators) == {'LR', 'XB'}

    def test_selecting_lr_only_does_not_require_xb_parameters(self):
        parameters = {'LR': {'max_iter': 1000, 'C': 0.5}}
        estimators = make_estimators(parameters, seed=0, model_keys=['LR'])

        assert set(estimators) == {'LR'}
        assert estimators['LR'].C == 0.5

    def test_selecting_xb_only_does_not_require_lr_parameters(self):
        parameters = {'XB': {'n_estimators': 5}}
        estimators = make_estimators(parameters, seed=0, model_keys=['XB'])

        assert set(estimators) == {'XB'}

    def test_unknown_model_key_raises_a_clear_error(self):
        parameters = {'LR': {'max_iter': 1000}}
        with pytest.raises(ValueError, match="Unknown model key"):
            make_estimators(parameters, seed=0, model_keys=['LR', 'RF'])
