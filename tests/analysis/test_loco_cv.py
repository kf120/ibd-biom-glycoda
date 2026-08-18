"""Tests for fold aggregation, cloning, and LOCO execution."""

import ast
import contextlib
import inspect
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from ibd_biom_glycoda.analysis import loco_cv
from ibd_biom_glycoda.analysis.loco_cv import (
    _compute_worst_cohort,
    aggregate_loco_results_across_folds_per_seed,
    compute_fold_averaged_disease_metrics,
    compute_fold_averaged_subgroup_metrics,
    safe_clone,
    summarize_loco_results_across_folds_and_seeds,
    validate_loco_config,
)
from ibd_biom_glycoda.analysis.pipelines import fit_pipeline_experiment, preprocess_fold_data


@pytest.fixture
def rng():
    return np.random.default_rng(7)


class TestAggregateLocoResultsAcrossFoldsPerSeed:
    def test_per_sample_keys_concatenate_across_all_folds(self, rng):
        fold_results = []
        for fold_idx in (1, 2, 3):
            n = 4
            fold_results.append({
                'test_out_true': rng.integers(0, 2, size=n),
                'test_out_pred': rng.uniform(size=(n, 2)),
                'test_out_fold_id': np.full(n, fold_idx),
                'Z_test_out_bin': rng.integers(0, 2, size=(n, 2)),
                'V_test_out': np.eye(2)[rng.integers(0, 2, size=n)],
                'age_ranges': [0, 40, 100],  # constant metadata
            })

        aggregated = aggregate_loco_results_across_folds_per_seed(fold_results)

        assert len(aggregated['test_out_true']) == 12
        assert len(aggregated['test_out_fold_id']) == 12
        assert aggregated['Z_test_out_bin'].shape == (12, 2)
        assert aggregated['V_test_out'].shape == (12, 2)
        np.testing.assert_array_equal(
            aggregated['test_out_fold_id'], np.repeat([1, 2, 3], 4)
        )
        # Not just fold 1's copy: all three folds' Z values must be present.
        np.testing.assert_array_equal(
            aggregated['Z_test_out_bin'],
            np.concatenate([fr['Z_test_out_bin'] for fr in fold_results], axis=0),
        )

    def test_constant_metadata_keeps_fold_ones_value(self, rng):
        fold_results = [
            {'test_out_true': rng.integers(0, 2, size=3), 'test_out_pred': rng.uniform(size=(3, 2)),
             'test_out_fold_id': np.full(3, 1), 'Z_test_out_bin': rng.integers(0, 2, size=(3, 2)),
             'V_test_out': np.eye(2)[rng.integers(0, 2, size=3)], 'age_ranges': [0, 40, 100]},
            {'test_out_true': rng.integers(0, 2, size=3), 'test_out_pred': rng.uniform(size=(3, 2)),
             'test_out_fold_id': np.full(3, 2), 'Z_test_out_bin': rng.integers(0, 2, size=(3, 2)),
             'V_test_out': np.eye(2)[rng.integers(0, 2, size=3)], 'age_ranges': [0, 50, 100]},
        ]

        aggregated = aggregate_loco_results_across_folds_per_seed(fold_results)

        assert aggregated['age_ranges'] == [0, 40, 100]

    def test_model_and_processor_collect_one_entry_per_fold(self, rng):
        """Retain one fitted model and processor per fold."""
        fold_results = []
        sentinel_models = [object(), object(), object()]
        sentinel_procs = [object(), object(), object()]
        for i, fold_idx in enumerate((1, 2, 3)):
            fold_results.append({
                'test_out_true': rng.integers(0, 2, size=3),
                'test_out_pred': rng.uniform(size=(3, 2)),
                'test_out_fold_id': np.full(3, fold_idx),
                'Z_test_out_bin': rng.integers(0, 2, size=(3, 2)),
                'V_test_out': np.eye(2)[rng.integers(0, 2, size=3)],
                'age_ranges': [0, 40, 100],
                'model': sentinel_models[i],
                'processor': sentinel_procs[i],
            })

        aggregated = aggregate_loco_results_across_folds_per_seed(fold_results)

        assert aggregated['model'] == sentinel_models
        assert aggregated['processor'] == sentinel_procs


class TestSafeClone:
    def test_clones_a_standard_estimator(self):
        estimator = LogisticRegression(C=0.5, random_state=7)
        cloned = safe_clone(estimator)

        assert cloned is not estimator
        assert cloned.get_params() == estimator.get_params()

    def test_falls_back_to_original_object_on_clone_failure(self):
        class Unclonable(BaseEstimator):
            """get_params() raising means clone() can't reconstruct it --
            mirrors why _safe_clone existed for e.g. GlyCompareProcessor."""

            def get_params(self, deep=True):
                raise TypeError("cannot introspect params")

        obj = Unclonable()
        assert safe_clone(obj) is obj

    def test_none_passes_through(self):
        assert safe_clone(None) is None


class TestPerFoldCloningPreventsModelAliasing:
    """Tests independence of fitted objects across folds."""

    def _make_data_dict(self, rng, flip_labels):
        n = 40
        X = rng.normal(size=(n, 4))
        y = rng.integers(0, 2, size=n)
        y[:2] = [0, 1]
        if flip_labels:
            y = 1 - y
        Z = rng.normal(size=(n, 2))
        Z_bin = np.column_stack([rng.integers(0, 2, size=n), rng.integers(0, 2, size=n)])
        V = np.eye(2)[rng.integers(0, 2, size=n)]

        d = {}
        for split in ('train', 'val_in', 'test_in', 'test_out'):
            d[f'X_{split}'] = X
            d[f'y_{split}'] = y
            d[f'Z_{split}'] = Z
            d[f'Z_{split}_bin'] = Z_bin
            d[f'V_{split}'] = V
        d['age_ranges'] = [0, 40, 100]
        d['cohort_locations'] = [['A', 'B']]
        return d

    def test_two_folds_get_independent_fitted_models(self):
        rng = np.random.default_rng(0)
        base_estimator = LogisticRegression(max_iter=1000, random_state=0)

        fold_1_data = self._make_data_dict(rng, flip_labels=False)
        fold_2_data = self._make_data_dict(rng, flip_labels=True)  # opposite labels -> different fit

        prepared_1 = preprocess_fold_data(RobustScaler(), fold_1_data, seed=0)
        result_1 = fit_pipeline_experiment(prepared_1, safe_clone(base_estimator), seed=0)

        prepared_2 = preprocess_fold_data(RobustScaler(), fold_2_data, seed=0)
        result_2 = fit_pipeline_experiment(prepared_2, safe_clone(base_estimator), seed=0)

        assert result_1['model'] is not result_2['model']
        assert result_1['model'] is not base_estimator
        assert result_2['model'] is not base_estimator
        # Independent fits must retain distinct coefficients.
        assert not np.allclose(result_1['model'].coef_, result_2['model'].coef_)

    def test_without_cloning_folds_would_alias_the_same_object(self):
        """Documents the bug this fixes: reusing (not cloning) the
        estimator across folds means every fold's stored model is the same
        object, reflecting whichever fold happened to fit it last."""
        rng = np.random.default_rng(0)
        shared_estimator = LogisticRegression(max_iter=1000, random_state=0)

        fold_1_data = self._make_data_dict(rng, flip_labels=False)
        fold_2_data = self._make_data_dict(rng, flip_labels=True)

        prepared_1 = preprocess_fold_data(RobustScaler(), fold_1_data, seed=0)
        result_1 = fit_pipeline_experiment(prepared_1, shared_estimator, seed=0)  # no clone

        prepared_2 = preprocess_fold_data(RobustScaler(), fold_2_data, seed=0)
        result_2 = fit_pipeline_experiment(prepared_2, shared_estimator, seed=0)  # no clone

        assert result_1['model'] is result_2['model']
        np.testing.assert_array_equal(result_1['model'].coef_, result_2['model'].coef_)


class TestComputeFoldAveragedDiseaseMetrics:
    def test_matches_manual_unweighted_fold_average(self, rng):
        y_true = rng.integers(0, 2, size=40)
        y_pred = rng.uniform(size=40)
        fold_id = np.repeat([1, 2, 3, 4], 10)

        result = compute_fold_averaged_disease_metrics(y_true, y_pred, fold_id, metrics=['AUROC'])

        expected = np.mean([
            roc_auc_score(y_true[fold_id == f], y_pred[fold_id == f]) for f in (1, 2, 3, 4)
        ])
        assert result['AUROC'] == pytest.approx(expected, abs=1e-12)

    def test_differs_from_pooled_when_folds_are_imbalanced(self, rng):
        """Fold averaging differs from pooling for imbalanced folds."""
        y_true = rng.integers(0, 2, size=40)
        y_pred = rng.uniform(size=40)
        fold_id = np.repeat([1, 2, 3, 4], 10)

        fold_averaged = compute_fold_averaged_disease_metrics(y_true, y_pred, fold_id, metrics=['AUROC'])
        pooled = roc_auc_score(y_true, y_pred)

        # Not asserting a specific relationship, just that these are
        # genuinely different computations (extremely unlikely to tie).
        assert fold_averaged['AUROC'] != pytest.approx(pooled, abs=1e-9)


class TestComputeFoldAveragedSubgroupMetrics:
    @pytest.fixture
    def subgroup_names(self):
        return {'age': {0: '<40', 1: '>40'}, 'sex': {0: 'M', 1: 'F'}, 'location': {0: 'A', 1: 'B'}}

    def test_uses_every_fold_not_just_fold_one(self, subgroup_names):
        """Subgroup metrics include every valid fold."""
        # Fold 1: sex 0(M),0(M),1(F),1(F) — M perfectly separated.
        y_true_f1 = np.array([0, 1, 0, 1])
        y_pred_f1 = np.array([0.1, 0.9, 0.3, 0.7])
        sex_f1 = np.array([0, 0, 1, 1])
        age_f1 = np.zeros(4, dtype=int)

        # Fold 2: M predictions inverted -> AUROC 0.0 for M.
        y_true_f2 = np.array([0, 1, 0, 1])
        y_pred_f2 = np.array([0.9, 0.1, 0.3, 0.7])
        sex_f2 = np.array([0, 0, 1, 1])
        age_f2 = np.zeros(4, dtype=int)

        y_true = np.concatenate([y_true_f1, y_true_f2])
        y_pred = np.concatenate([y_pred_f1, y_pred_f2])
        fold_id = np.concatenate([np.full(4, 1), np.full(4, 2)])
        Z_bin = np.column_stack([
            np.concatenate([sex_f1, sex_f2]),
            np.concatenate([age_f1, age_f2]),
        ])
        V = np.ones((8, 1))

        result = compute_fold_averaged_subgroup_metrics(
            y_true, y_pred, fold_id, Z_bin, V, 'sex', ['AUROC'], subgroup_names
        )

        # Average the perfect and inverted fold scores.
        assert result['M']['AUROC'] == pytest.approx(0.5, abs=1e-9)

    def test_age_sex_returns_nested_structure(self, rng, subgroup_names):
        n = 40
        y_true = rng.integers(0, 2, size=n)
        y_pred = rng.uniform(size=n)
        fold_id = np.repeat([1, 2], 20)
        Z_bin = np.column_stack([rng.integers(0, 2, size=n), rng.integers(0, 2, size=n)])
        V = np.ones((n, 1))

        result = compute_fold_averaged_subgroup_metrics(
            y_true, y_pred, fold_id, Z_bin, V, 'age_sex', ['AUROC'], subgroup_names
        )

        assert set(result.keys()) <= {'<40', '>40'}
        for sex_dict in result.values():
            assert set(sex_dict.keys()) <= {'M', 'F'}
            for metrics_dict in sex_dict.values():
                assert 'AUROC' in metrics_dict

    def test_single_class_fold_slice_is_excluded_from_that_folds_average(self, subgroup_names):
        # Fold 1: M group is single-class (skipped entirely for M).
        y_true_f1 = np.array([0, 0, 0, 1])
        y_pred_f1 = np.array([0.1, 0.2, 0.3, 0.7])
        sex_f1 = np.array([0, 0, 1, 1])

        # Fold 2: M group has both classes, AUROC = 1.0.
        y_true_f2 = np.array([0, 1, 0, 1])
        y_pred_f2 = np.array([0.1, 0.9, 0.3, 0.7])
        sex_f2 = np.array([0, 0, 1, 1])

        y_true = np.concatenate([y_true_f1, y_true_f2])
        y_pred = np.concatenate([y_pred_f1, y_pred_f2])
        fold_id = np.concatenate([np.full(4, 1), np.full(4, 2)])
        Z_bin = np.column_stack([
            np.concatenate([sex_f1, sex_f2]),
            np.zeros(8, dtype=int),
        ])
        V = np.ones((8, 1))

        result = compute_fold_averaged_subgroup_metrics(
            y_true, y_pred, fold_id, Z_bin, V, 'sex', ['AUROC'], subgroup_names
        )

        # Only fold 2 contributed to M's average -> exactly 1.0, not diluted
        # by a NaN/None from fold 1's skipped single-class slice.
        assert result['M']['AUROC'] == pytest.approx(1.0, abs=1e-12)


class TestSummarizeLocoResultsAcrossFoldsAndSeeds:
    def _make_res(self, rng, n_per_fold=8, n_folds=2, n_features_ignored=None):
        n_total = n_per_fold * n_folds
        y_true = rng.integers(0, 2, size=n_total)
        y_pred = rng.uniform(size=(n_total,))
        fold_id = np.repeat(np.arange(1, n_folds + 1), n_per_fold)
        Z_bin = np.column_stack([
            rng.integers(0, 2, size=n_total),  # sex
            np.zeros(n_total, dtype=int),      # age (single bin, unused here)
        ])
        V = np.ones((n_total, 1))

        # compute_scoring_metrics expects 2D proba or 1D; give it 2-col proba
        # for consistency with real pipeline output.
        y_pred_2col = np.column_stack([1 - y_pred, y_pred])

        return {
            'test_out_true': y_true,
            'test_out_pred': y_pred_2col,
            'test_out_fold_id': fold_id,
            'Z_test_out_bin': Z_bin,
            'V_test_out': V,
            'age_ranges': [0, 40, 100],
            'cohort_locations': [['A', 'B']],
        }

    def test_disease_metrics_use_fold_averaging(self, rng):
        res = self._make_res(rng)
        results = {'LR': [res]}

        final_results, _ = summarize_loco_results_across_folds_and_seeds(
            results, variables=['disease'], all_metrics=['AUROC'], set_types=['test_out'], model_keys=['LR']
        )

        disease_df = final_results['test_out']['disease']
        assert list(disease_df['Model']) == ['LR']

        y_true, y_pred, fold_id = res['test_out_true'], res['test_out_pred'][:, 1], res['test_out_fold_id']
        expected = np.mean([
            roc_auc_score(y_true[fold_id == f], y_pred[fold_id == f]) for f in np.unique(fold_id)
        ])
        assert disease_df.loc[0, 'AUROC Mean'] == pytest.approx(round(expected, 3), abs=1e-9)

    def test_subgroup_output_contains_sex_groups(self, rng):
        res = self._make_res(rng)
        results = {'LR': [res]}

        _, subgroup_analysis = summarize_loco_results_across_folds_and_seeds(
            results, variables=['disease', 'sex'], all_metrics=['AUROC'], set_types=['test_out'], model_keys=['LR']
        )

        sex_groups = subgroup_analysis['test_out']['sex']
        assert set(sex_groups.keys()) <= {'M', 'F'}
        for group_result in sex_groups.values():
            assert 'LR' in group_result
            assert 'AUROC' in group_result['LR']

    def test_no_variables_besides_disease_skips_subgroup_work_without_error(self, rng):
        res = self._make_res(rng)
        results = {'LR': [res]}

        final_results, subgroup_analysis = summarize_loco_results_across_folds_and_seeds(
            results, variables=['disease'], all_metrics=['AUROC'], set_types=['test_out'], model_keys=['LR']
        )

        assert subgroup_analysis['test_out'] == {}


def _make_fake_process_one_loco_run(calls):
    """Stand-in for process_one_loco_run that avoids needing real LOCO
    data/dataset files: records the (cohort, seed) it was called with and
    returns a minimally-shaped aggregated dict per model_key, matching
    what collect_loco_results reads."""

    def _fake(df, feature_cols, cohort, seed, processor, model_keys,
              model_parameters, n_folds, **kwargs):
        calls.append((cohort, seed))
        aggregated = {
            model_key: {
                'train_true': np.array([0, 1]), 'train_pred': np.array([[0.5, 0.5], [0.4, 0.6]]),
                'val_in_true': np.array([0]), 'val_in_pred': np.array([[0.5, 0.5]]),
                'test_in_true': np.array([0]), 'test_in_pred': np.array([[0.5, 0.5]]),
                'test_out_true': np.array([0]), 'test_out_pred': np.array([[0.5, 0.5]]),
            }
            for model_key in model_keys
        }
        return cohort, seed, aggregated

    return _fake


class TestRunLocoCvFlattenedTaskGrid:
    """Tests the flattened preprocessor/cohort/seed task grid."""

    def test_covers_full_preprocessor_x_cohort_x_seed_grid(self, monkeypatch):
        calls = []
        monkeypatch.setattr(loco_cv, 'process_one_loco_run', _make_fake_process_one_loco_run(calls))

        scaled_procs = {'CLR': object(), 'GlyCmp': object()}
        loco_test_cohorts = ['UK', 'US']
        selected_seeds = [0, 1, 2]
        model_keys = ['LR', 'XB']
        pipeline_keys = [f'{m}+{p}' for m in model_keys for p in scaled_procs]

        loco_cv.run_loco_cv(
            df=None, feature_cols=[], loco_test_cohorts=loco_test_cohorts,
            selected_seeds=selected_seeds, scaled_procs=scaled_procs,
            model_keys=model_keys, model_parameters={'LR': {}, 'XB': {}},
            pipeline_keys=pipeline_keys, n_folds=3, n_jobs=1,
        )

        assert len(calls) == len(scaled_procs) * len(loco_test_cohorts) * len(selected_seeds)
        expected_pairs = {(c, s) for c in loco_test_cohorts for s in selected_seeds}
        counts = Counter(calls)
        assert set(counts.keys()) == expected_pairs
        # Each (cohort, seed) pair is processed once per preprocessor.
        assert all(count == len(scaled_procs) for count in counts.values())

    def test_result_structure_matches_pipeline_keys(self, monkeypatch):
        calls = []
        monkeypatch.setattr(loco_cv, 'process_one_loco_run', _make_fake_process_one_loco_run(calls))

        scaled_procs = {'CLR': object()}
        loco_test_cohorts = ['UK']
        selected_seeds = [0, 1]
        pipeline_keys = ['LR+CLR']

        result = loco_cv.run_loco_cv(
            df=None, feature_cols=[], loco_test_cohorts=loco_test_cohorts,
            selected_seeds=selected_seeds, scaled_procs=scaled_procs,
            model_keys=['LR'], model_parameters={'LR': {}, 'XB': {}},
            pipeline_keys=pipeline_keys, n_folds=3, n_jobs=1,
        )

        assert set(result.keys()) == {'raw', 'labels'}
        assert len(result['raw']['UK']['LR+CLR']) == len(selected_seeds)
        assert len(result['labels']['LR+CLR']['test_out_true']) == len(selected_seeds)

    def test_same_coverage_with_multiple_workers(self, monkeypatch):
        """n_jobs>1 must cover exactly the same grid as n_jobs==1 -- using
        the threading backend so the monkeypatched fake (which only exists
        in this test process) is actually visible to the workers, unlike
        loky's separate processes."""
        calls = []
        monkeypatch.setattr(loco_cv, 'process_one_loco_run', _make_fake_process_one_loco_run(calls))

        scaled_procs = {'CLR': object(), 'GlyCmp': object()}
        loco_test_cohorts = ['UK', 'US']
        selected_seeds = [0, 1]
        pipeline_keys = ['LR+CLR', 'LR+GlyCmp']

        loco_cv.run_loco_cv(
            df=None, feature_cols=[], loco_test_cohorts=loco_test_cohorts,
            selected_seeds=selected_seeds, scaled_procs=scaled_procs,
            model_keys=['LR'], model_parameters={'LR': {}, 'XB': {}},
            pipeline_keys=pipeline_keys, n_folds=3, n_jobs=2, backend='threading',
        )

        assert len(calls) == len(scaled_procs) * len(loco_test_cohorts) * len(selected_seeds)


class TestRunLocoCvThreadPinning:
    """Tests native thread limits within LOCO tasks."""

    def test_each_task_runs_inside_threadpool_limits_one(self, monkeypatch):
        recorded = []

        @contextlib.contextmanager
        def _fake_threadpool_limits(limits=None, **kwargs):
            recorded.append(limits)
            yield

        calls = []
        monkeypatch.setattr(loco_cv, 'threadpool_limits', _fake_threadpool_limits)
        monkeypatch.setattr(loco_cv, 'process_one_loco_run', _make_fake_process_one_loco_run(calls))

        scaled_procs = {'CLR': object()}
        loco_test_cohorts = ['UK', 'US']
        selected_seeds = [0, 1]
        pipeline_keys = ['LR+CLR']

        loco_cv.run_loco_cv(
            df=None, feature_cols=[], loco_test_cohorts=loco_test_cohorts,
            selected_seeds=selected_seeds, scaled_procs=scaled_procs,
            model_keys=['LR'], model_parameters={'LR': {}, 'XB': {}},
            pipeline_keys=pipeline_keys, n_folds=3, n_jobs=1,
        )

        n_tasks = len(scaled_procs) * len(loco_test_cohorts) * len(selected_seeds)
        assert recorded == [1] * n_tasks


class TestValidateLocoConfig:
    """Config drift that would otherwise surface as silently missing results."""

    @staticmethod
    def _valid_kwargs(**overrides):
        kwargs = dict(
            scaled_procs={'Raw': object(), 'CLR': object()},
            model_keys=['LR', 'XB'],
            model_parameters={'LR': {}, 'XB': {}},
            pipeline_keys=['LR+Raw', 'XB+Raw', 'LR+CLR', 'XB+CLR'],
            loco_test_cohorts=['UK', 'US'],
            selected_seeds=[0, 1],
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_config_passes_and_reports(self, capsys):
        validate_loco_config(**self._valid_kwargs())
        out = capsys.readouterr().out
        assert 'Config valid' in out
        assert '2 model(s)' in out
        assert '4 pipeline(s)' in out

    def test_verbose_false_prints_nothing(self, capsys):
        validate_loco_config(**self._valid_kwargs(), verbose=False)
        assert capsys.readouterr().out == ''

    def test_pipeline_key_order_does_not_matter(self):
        # Notebooks build preprocessor-outer; other callers build model-outer.
        validate_loco_config(**self._valid_kwargs(
            pipeline_keys=['LR+Raw', 'LR+CLR', 'XB+Raw', 'XB+CLR']
        ))

    def test_stale_pipeline_keys_after_shrinking_model_keys_raises(self):
        # MODEL_KEYS narrowed to LR but PIPELINE_KEYS still carries the XB entries.
        with pytest.raises(ValueError, match="unexpected"):
            validate_loco_config(**self._valid_kwargs(
                model_keys=['LR'], model_parameters={'LR': {}},
            ))

    def test_pipeline_keys_missing_a_pair_raises(self):
        with pytest.raises(ValueError, match="missing"):
            validate_loco_config(**self._valid_kwargs(
                pipeline_keys=['LR+Raw', 'XB+Raw', 'LR+CLR'],
            ))

    def test_model_without_parameters_raises(self):
        with pytest.raises(ValueError, match="model_parameters has no entry"):
            validate_loco_config(**self._valid_kwargs(model_parameters={'LR': {}}))

    def test_duplicate_pipeline_keys_raise(self):
        with pytest.raises(ValueError, match="duplicates"):
            validate_loco_config(**self._valid_kwargs(
                pipeline_keys=['LR+Raw', 'LR+Raw', 'XB+Raw', 'LR+CLR', 'XB+CLR'],
            ))

    @pytest.mark.parametrize('field, empty', [
        ('model_keys', []),
        ('scaled_procs', {}),
        ('loco_test_cohorts', []),
        ('selected_seeds', []),
    ])
    def test_empty_inputs_raise(self, field, empty):
        with pytest.raises(ValueError, match=f"{field} is empty"):
            validate_loco_config(**self._valid_kwargs(**{field: empty}))

    def test_every_problem_is_reported_together(self):
        with pytest.raises(ValueError) as excinfo:
            validate_loco_config(**self._valid_kwargs(
                model_keys=['LR', 'RF'], model_parameters={'LR': {}},
            ))
        message = str(excinfo.value)
        assert 'model_parameters has no entry' in message
        assert 'pipeline_keys does not match' in message

    def test_run_loco_cv_rejects_invalid_config_before_running(self, monkeypatch):
        calls = []
        monkeypatch.setattr(loco_cv, 'process_one_loco_run', _make_fake_process_one_loco_run(calls))

        with pytest.raises(ValueError, match="LOCO configuration is invalid"):
            loco_cv.run_loco_cv(
                df=None, feature_cols=[], loco_test_cohorts=['UK'],
                selected_seeds=[0], scaled_procs={'CLR': object()},
                model_keys=['LR'], model_parameters={'LR': {}},
                pipeline_keys=['LR+CLR', 'XB+CLR'], n_folds=3, n_jobs=1,
            )
        assert calls == []

    def test_run_loco_cv_validate_false_skips_the_check(self, monkeypatch):
        calls = []
        monkeypatch.setattr(loco_cv, 'process_one_loco_run', _make_fake_process_one_loco_run(calls))

        loco_cv.run_loco_cv(
            df=None, feature_cols=[], loco_test_cohorts=['UK'],
            selected_seeds=[0], scaled_procs={'CLR': object()},
            model_keys=['LR'], model_parameters={'LR': {}},
            pipeline_keys=['LR+CLR'], n_folds=3, n_jobs=1, validate=False,
        )
        assert len(calls) == 1


class TestComputeWorstCohort:
    """Direction must generalize to any metric, not a fixed whitelist, or newly
    requested metrics report the best cohort as the worst."""

    def test_higher_is_better_metric_reports_lowest_mean_as_worst(self):
        cohort_means = [('UK', 0.90), ('US', 0.70), ('IT', 0.85)]
        assert _compute_worst_cohort('AUROC', cohort_means) == 'US'

    def test_lower_is_better_metric_reports_highest_mean_as_worst(self):
        cohort_means = [('UK', 0.30), ('US', 0.70), ('IT', 0.50)]
        assert _compute_worst_cohort('LogLoss', cohort_means) == 'US'

    def test_higher_is_better_metrics_beyond_auroc_report_correctly(self):
        cohort_means = [('UK', 0.90), ('US', 0.70), ('IT', 0.85)]
        for metric in ('AUPRC', 'BAcc', 'MCC', 'Precision'):
            assert _compute_worst_cohort(metric, cohort_means) == 'US', metric

    def test_empty_cohort_means_returns_none(self):
        assert _compute_worst_cohort('AUROC', []) is None

    def test_target_valued_metric_reports_the_cohort_furthest_from_its_target(self):
        """A calibration slope has no direction. Ranking it by raw value would
        name the best-calibrated cohort as the worst whenever the miscalibrated
        one over-shrinks."""
        cohort_means = [('UK', 1.00), ('US', 1.60), ('IT', 0.95)]
        assert _compute_worst_cohort('CalibrationSlope', cohort_means) == 'US'

    def test_target_valued_metric_also_catches_under_shrinkage(self):
        cohort_means = [('UK', 1.00), ('US', 1.10), ('IT', 0.40)]
        assert _compute_worst_cohort('CalibrationSlope', cohort_means) == 'IT'

    def test_calibration_intercept_targets_zero_in_both_directions(self):
        cohort_means = [('UK', 0.05), ('US', -0.90), ('IT', 0.30)]
        assert _compute_worst_cohort('CalibrationIntercept', cohort_means) == 'US'

    def test_cohorts_with_nan_values_are_not_named_worst(self):
        """A missing estimate is not a poor one; naming it worst would turn a
        non-estimable subgroup into a reported failure."""
        cohort_means = [('UK', 0.90), ('US', float('nan')), ('IT', 0.75)]
        assert _compute_worst_cohort('AUROC', cohort_means) == 'IT'

    def test_all_nan_cohort_means_returns_none(self):
        cohort_means = [('UK', float('nan')), ('US', float('nan'))]
        assert _compute_worst_cohort('AUROC', cohort_means) is None


class TestWorkerOutputIsAsciiSafe:
    """Tests compatibility with ASCII-only worker output streams."""

    @staticmethod
    def _string_literals_in_call(node):
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    yield sub.value

    def test_no_non_ascii_characters_in_print_or_warn_calls(self):
        source_path = Path(inspect.getfile(loco_cv))
        tree = ast.parse(source_path.read_text(encoding='utf-8'))

        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_print = isinstance(func, ast.Name) and func.id == 'print'
            is_warn = isinstance(func, ast.Attribute) and func.attr == 'warn'
            if not (is_print or is_warn):
                continue
            for text in self._string_literals_in_call(node):
                if not text.isascii():
                    offenders.append((node.lineno, text))

        assert offenders == [], f"Non-ASCII text in print()/warn() calls: {offenders}"
