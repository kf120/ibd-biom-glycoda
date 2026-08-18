# -*- coding: utf-8 -*-
"""
@author: Kostis Flevaris
"""

import warnings
import numpy as np
import pandas as pd

from collections import defaultdict, Counter
from joblib import Parallel, delayed
from scipy.stats import t
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import Bunch
from threadpoolctl import threadpool_limits

from ibd_biom_glycoda.utils import seed_everything, get_safe_index
from ibd_biom_glycoda.data.dataset import (
    prepare_loco_data,
    cohort_index_from_onehot,
    UNKNOWN_COHORT_INDEX,
)
from ibd_biom_glycoda.analysis.pipelines import (
    make_estimators,
    preprocess_fold_data,
    fit_pipeline_experiment,
    create_demographic_subgroup_names,
)
from ibd_biom_glycoda.evaluation.metrics import (
    compute_scoring_metrics,
    aggregate_all_metrics,
    summarize_scoring_metrics,
    metric_distance_from_ideal,
    SUPPORT_KEYS,
    ESTIMABLE_KEY,
)
from ibd_biom_glycoda.evaluation.subgroup_analysis import (
    calculate_subgroup_performance,
    calculate_intersection_performance,
)

# Splits retained for fold-level and subgroup metrics.
LOCO_SPLITS = ('train', 'val_in', 'test_in', 'test_out')


def safe_clone(estimator):
    """Clone an estimator, or return it unchanged if cloning is unsupported.

    The fallback reuses the same instance across folds, weakening their
    independence, so it warns rather than failing silently.
    """
    if estimator is None:
        return None
    try:
        return clone(estimator)
    except Exception as exc:
        # Not repr(estimator): BaseEstimator.__repr__ calls the same get_params()
        # that clone() just failed on, which would mask this exception with a second.
        warnings.warn(
            f"safe_clone: clone() failed for a {type(estimator).__name__} instance "
            f"({exc!r}); reusing the same instance instead of an independent clone."
        )
        return estimator

def create_group_identifiers(V, Z_bin, age_ranges):
    """
    Create group identifiers from cohort and demographic information.
    
    Groups are defined as the cross-product: Cohort x Age_bin x Sex
    
    Parameters
    ----------
    V : array-like, shape (n_samples, n_cohorts)
        One-hot encoded cohort indicators
    Z_bin : array-like, shape (n_samples, 2)
        Binary demographics [Sex, Age_bin]
    age_ranges : list
        Age bin edges (used to determine number of age bins)
    
    Returns
    -------
    groups : ndarray
        Integer group identifiers for each sample
    n_groups : int
        Total number of unique groups
    
    Examples
    --------
    With 3 cohorts, 2 age bins, 2 sexes:
    - Total groups: 3 x 2 x 2 = 12
    - Group 0: Cohort_0, Age_0, Sex_0
    - Group 1: Cohort_0, Age_0, Sex_1
    - Group 2: Cohort_0, Age_1, Sex_0
    - ...
    """
    if hasattr(V, 'values'):
        V = V.values
    else:
        V = np.asarray(V)

    if hasattr(Z_bin, 'values'):
        Z_bin = Z_bin.values
    else:
        Z_bin = np.asarray(Z_bin)

    cohort_idx = cohort_index_from_onehot(V)
    if np.any(cohort_idx == UNKNOWN_COHORT_INDEX):
        raise ValueError(
            "create_group_identifiers received rows with an all-zero cohort one-hot. "
            "These are samples from a cohort the encoder never saw (typically the "
            "held-out cohort); grouping them would silently merge them into the "
            "first training cohort."
        )
    n_cohorts = V.shape[1]

    # Z_bin columns are [Sex, Age_bin].
    sex = Z_bin[:, 0].astype(int)
    age_bin = Z_bin[:, 1].astype(int)

    n_age_bins = len(age_ranges) - 1
    n_sex = 2

    # group_id = cohort_idx * (n_age_bins * n_sex) + age_bin * n_sex + sex
    groups = cohort_idx * (n_age_bins * n_sex) + age_bin * n_sex + sex

    n_groups = n_cohorts * n_age_bins * n_sex
    
    return groups, n_groups

def create_stratification_labels(V, y, Z_bin, age_ranges):
    """
    Create joint stratification labels combining cohort, demographics, and disease.
    
    Ensures stratified splits preserve cohort, age, sex, and disease distributions.
    Uses the same grouping logic as create_group_identifiers() for consistency.
    
    Parameters
    ----------
    V : array-like, shape (n_samples, n_cohorts)
        One-hot encoded cohort indicators
    y : array-like, shape (n_samples,)
        Disease labels (0 to C-1)
    Z_bin : array-like, shape (n_samples, 2)
        Binary demographics [Sex, Age_bin]
    age_ranges : list
        Age bin edges (used to determine number of age bins)
        
    Returns
    -------
    ndarray
        Joint labels: group_id * n_classes + disease_idx
    """
    demographic_groups, _ = create_group_identifiers(V, Z_bin, age_ranges)
    y = np.asarray(y).astype(int).ravel()
    num_classes = len(np.unique(y))
    stratification_labels = demographic_groups * num_classes + y
    return stratification_labels

def print_stratification_profile(stratification_labels, V, y, age_ranges, ohe_cohort, dataset_name="Dataset", verbose=True):
    """Print sample counts per stratum to diagnose sparse splits."""
    strata_counts = Counter(stratification_labels)
    unique_strata = sorted(strata_counts.keys())

    cohort_names = [ohe_cohort.categories_[0][i] for i in range(ohe_cohort.categories_[0].shape[0])]
    n_cohorts = V.shape[1]
    n_age_bins = len(age_ranges) - 1
    n_sexes = 2
    n_classes = len(np.unique(y))
    total_possible_strata = n_cohorts * n_age_bins * n_sexes * n_classes

    print(f"\n{'='*70}")
    print(f"STRATIFICATION PROFILE: {dataset_name}")
    print(f"{'='*70}")
    summary_line = (
        f"Total samples: {len(stratification_labels)} | "
        f"Unique strata: {len(unique_strata)} / {total_possible_strata} possible | "
        f"Min count per stratum: {min(strata_counts.values()) if strata_counts else 0} | "
        f"Max count per stratum: {max(strata_counts.values()) if strata_counts else 0}"
    )
    print(summary_line)
    print(f"{'-'*70}")

    if verbose:
        for stratum_id in unique_strata:
            count = strata_counts[stratum_id]
            disease_idx = stratum_id % n_classes
            tmp = stratum_id // n_classes
            sex_idx = tmp % n_sexes
            tmp //= n_sexes
            age_idx = tmp % n_age_bins
            cohort_idx = tmp // n_age_bins

            print(
                f"Stratum {stratum_id:3d} | Cohort={cohort_names[cohort_idx]:>10} | "
                f"AgeBin={age_ranges[age_idx]}-{age_ranges[age_idx+1]} | Sex={sex_idx} | "
                f"Disease={disease_idx} | Count={count:4d}"
            )

    print(f"{'='*70}\n")

    # Minimum folds equals minimum count across strata (avoid zero counts per fold)
    min_stratum_count = min(strata_counts.values()) if strata_counts else 0
    return strata_counts, min_stratum_count

def split_train_validation(X, y, Z, Z_bin, V, test_size=0.1, stratify_by=None, random_state=42):
    """
    Split fold data into train and validation partitions.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target labels.
    Z : array-like
        Continuous covariates.
    Z_bin : array-like
        Binary covariates.
    V : array-like
        Cohort indicators.
    test_size : float, default=0.1
        Validation split proportion.
    stratify_by : array-like, optional
        Labels used for stratification.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping with ``train`` and ``val`` entries.
    """
    stratify = stratify_by if stratify_by is not None else y

    X_train, X_val, y_train, y_val, Z_train, Z_val, Z_bin_train, Z_bin_val, V_train, V_val = train_test_split(
        X, y, Z, Z_bin, V,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    return {
        'train': {
            'X': X_train,
            'y': y_train,
            'Z': Z_train,
            'Z_bin': Z_bin_train,
            'V': V_train
        },
        'val': {
            'X': X_val,
            'y': y_val,
            'Z': Z_val,
            'Z_bin': Z_bin_val,
            'V': V_val
        }
    }

def extract_age_sex_from_Z(Z_bin):
    """Return age-bin and sex arrays from compact demographic encoding."""
    if hasattr(Z_bin, 'values'):
        Z_bin = Z_bin.values
    else:
        Z_bin = np.asarray(Z_bin)
    sex = Z_bin[:, 0].astype(int)
    age_bin = Z_bin[:, 1].astype(int)
    return age_bin, sex


def assign_age_sex_strata(age_bin, sex):
    """Map (age_bin, sex) pairs to unique stratum identifiers."""
    return age_bin * 2 + sex


def assign_inverse_frequency_weights(strata):
    """Compute inverse-frequency weights so each stratum contributes equally."""
    strata = np.asarray(strata)
    unique, counts = np.unique(strata, return_counts=True)
    N = len(strata)
    k = len(unique)
    weight_per_group = {g: (N / k) / n_g for g, n_g in zip(unique, counts)}
    return np.array([weight_per_group[g] for g in strata])


def build_weight_strata(Z_bin, y=None, components=None):
    """Construct modular stratum IDs for weighting strategies.

    Parameters
    ----------
    Z_bin : array-like, shape (n_samples, 2)
        Encoded demographic covariates [Sex, Age_bin].
    y : array-like, optional
        Label vector. Required when using the "label" component.
    components : sequence of str, optional
        Ordered components to include when building strata. Supported values:
        - "age_sex": use age-sex strata (default)
        - "label": include class labels to capture prevalence

    Returns
    -------
    numpy.ndarray
        Integer stratum IDs spanning the Cartesian product of all components.
    """
    if components is None:
        components = ("age_sex",)
    if isinstance(components, str):
        components = (components,)

    component_arrays = []
    component_bases = []

    def _encode(values):
        values = np.asarray(values)
        unique_vals, inverse = np.unique(values, return_inverse=True)
        return inverse, len(unique_vals)

    for component in components:
        if component == "age_sex":
            age_bin, sex = extract_age_sex_from_Z(Z_bin)
            values, base = _encode(assign_age_sex_strata(age_bin, sex))
        elif component == "label":
            if y is None:
                raise ValueError("y must be provided when components include 'label'.")
            values, base = _encode(np.asarray(y).ravel())
        else:
            raise ValueError(f"Unsupported component '{component}'.")

        component_arrays.append(values)
        component_bases.append(base)

    strata = component_arrays[0]
    for values, base in zip(component_arrays[1:], component_bases[1:]):
        if base == 0:
            continue
        strata = strata * base + values

    return strata


def compute_sample_weights(Z_bin, y=None, components=None):
    """
    Compute sample weights based on age-sex strata.
    
    Works with compact Z_bin format (n_samples, 2) where:
    - Column 0: Sex (0 or 1)
    - Column 1: Age bin (0, 1, 2, ...)
    
    Parameters
    ----------
    Z_bin : array-like, shape (n_samples, 2)
        Binary encoded covariates [Sex, Age_bin]
    y : array-like, optional
        Label vector. Required when including the "label" component.
    components : sequence of str, optional
        Weighting components to include (default: ("age_sex",)).
    
    Returns
    -------
    weights : numpy array
        Sample weights for each sample
    """
    strata = build_weight_strata(Z_bin, y=y, components=components)
    return assign_inverse_frequency_weights(strata)

def prepare_loco_data_outer(df_sub, coda_cols, loco_cohort, seed):
    """Prepare outer-loop datasets for a LOCO run.

    Parameters
    ----------
    df_sub : DataFrame
        Cohort-filtered data.
    coda_cols : list
        Names of compositional features.
    loco_cohort : str
        Cohort held out for testing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Bunch
        Container with train and test splits plus metadata.
    """
    seed_everything(seed)
    
    train_data, test_data, metadata = prepare_loco_data(
        df=df_sub,
        coda_cols=coda_cols,
        loco_test_cohort=loco_cohort,
        random_state=seed,
        print_set_sizes=False
    )
    
    (X_train, Z_train, Z_train_bin, y_train, V_train) = train_data
    (X_test_out, Z_test_out, Z_test_out_bin, y_test_out, V_test_out) = test_data

    joint_labels_train = create_stratification_labels(V_train, y_train, Z_train_bin, metadata['age_ranges'])

    _, min_nfolds = print_stratification_profile(
        stratification_labels=joint_labels_train,
        V=V_train,
        y=y_train,
        age_ranges=metadata['age_ranges'],
        ohe_cohort=metadata['ohe_cohort'],
        dataset_name=f"Training Set (LOCO={loco_cohort}, seed={seed})",
        verbose=False
    )
    
    base = Bunch(
        seed=seed,
        loco_cohort=loco_cohort,
        X_train=X_train, y_train=y_train, Z_train=Z_train,
        Z_train_bin=Z_train_bin, V_train=V_train,
        X_test_out=X_test_out, y_test_out=y_test_out,
        Z_test_out=Z_test_out, Z_test_out_bin=Z_test_out_bin,
        V_test_out=V_test_out,
        age_ranges=metadata['age_ranges'],
        cohort_locations=metadata['cohort_locations'],
        ohe_cohort=metadata['ohe_cohort'],
        scaler=metadata['scaler_Z'],
        gp_cols=metadata['coda_cols'],
        joint_labels_train=joint_labels_train,
        min_nfolds=min_nfolds
    )
    
    return base


def prepare_loco_data_inner(
    outer,
    train_idx,
    val_idx,
    fold_idx,
    use_full_stratification_inner_cv=True,
    use_sample_weights=True,
    val_size_inner=0.10
):
    """Assemble fold-specific data for the inner cross-validation loop.

    Parameters
    ----------
    outer : Bunch
        Output of ``prepare_loco_data_outer``.
    train_idx : array-like
        Indices for the fold training portion.
    val_idx : array-like
        Indices for the fold validation portion.
    fold_idx : int
        Fold counter used for seeding.
    use_sample_weights : bool, default=True
        Whether to compute age-sex sample weights.
    val_size_inner : float, default=0.10
        Validation split proportion for the inner CV fold.

    Returns
    -------
    dict
        Dictionary containing all fold splits and metadata.
    """
    
    X_all = outer.X_train
    y_all = outer.y_train
    Z_all = outer.Z_train
    Z_bin_all = outer.Z_train_bin
    V_all = outer.V_train
    joint_labels_all = outer.joint_labels_train

    X_fold_train = get_safe_index(X_all, train_idx)
    X_fold_test_in   = get_safe_index(X_all, val_idx)
    y_fold_train = get_safe_index(y_all, train_idx)
    y_fold_test_in   = get_safe_index(y_all, val_idx)
    Z_fold_train = get_safe_index(Z_all, train_idx)
    Z_fold_test_in   = get_safe_index(Z_all, val_idx)
    Z_bin_fold_train = get_safe_index(Z_bin_all, train_idx)
    Z_bin_fold_test_in   = get_safe_index(Z_bin_all, val_idx)
    V_fold_train = get_safe_index(V_all, train_idx)
    V_fold_test_in   = get_safe_index(V_all, val_idx)
    joint_labels_fold_train = get_safe_index(joint_labels_all, train_idx)

    if use_full_stratification_inner_cv:
        stratification_labels = joint_labels_fold_train
    else:
        stratification_labels = y_fold_train

    splits = split_train_validation(
        X=X_fold_train,
        y=y_fold_train,
        Z=Z_fold_train,
        Z_bin=Z_bin_fold_train,
        V=V_fold_train,
        test_size=val_size_inner,
        stratify_by=stratification_labels,
        random_state=outer.seed + fold_idx
    )

    X_train_final = splits['train']['X']
    y_train_final = splits['train']['y']
    Z_train_final = splits['train']['Z']
    Z_bin_train_final = splits['train']['Z_bin']
    V_train_final = splits['train']['V']

    X_val_in = splits['val']['X']
    y_val_in = splits['val']['y']
    Z_val_in = splits['val']['Z']
    Z_bin_val_in = splits['val']['Z_bin']
    V_val_in = splits['val']['V']

    if use_sample_weights:
        sample_weights_train = compute_sample_weights(Z_bin_train_final, y=y_train_final, components=("age_sex", "label"))
    else:
        sample_weights_train = None

    groups_train, n_groups_train = create_group_identifiers(
        V_train_final, Z_bin_train_final, outer.age_ranges
    )

    # No trailing newline: the next print continues this line.
    print(
        f" Fold {fold_idx} sizes | "
        f"Train: {len(X_train_final)}, "
        f"Val: {len(X_val_in) if X_val_in is not None else 0}, "
        f"Test In: {len(X_fold_test_in)}, "
        f"Test Out: {len(outer.X_test_out)}",
        end=" ",
        flush=True
    )


    fold_data = {
        'X_train': X_train_final,
        'y_train': y_train_final,
        'Z_train': Z_train_final,
        'Z_train_bin': Z_bin_train_final,
        'V_train': V_train_final,
        'sample_weights_train': sample_weights_train,
        'groups_train': groups_train,

        'X_val_in': X_val_in,
        'y_val_in': y_val_in,
        'Z_val_in': Z_val_in,
        'Z_val_in_bin': Z_bin_val_in,
        'V_val_in': V_val_in,

        'X_test_in': X_fold_test_in,
        'y_test_in': y_fold_test_in,
        'Z_test_in': Z_fold_test_in,
        'Z_test_in_bin': Z_bin_fold_test_in,
        'V_test_in': V_fold_test_in,

        'X_test_out': outer.X_test_out,
        'y_test_out': outer.y_test_out,
        'Z_test_out': outer.Z_test_out,
        'Z_test_out_bin': outer.Z_test_out_bin,
        'V_test_out': outer.V_test_out,

        'age_ranges': outer.age_ranges,
        'cohort_locations': outer.cohort_locations,
        'ohe_cohort': outer.ohe_cohort,
        'scaler': outer.scaler,
        'GP_cols': outer.gp_cols,

        'n_groups': n_groups_train,
        'use_sample_weights': use_sample_weights,
    }

    return fold_data


def aggregate_loco_results_across_folds_per_seed(fold_results):
    """Stack per-fold outputs into aggregate arrays.

    Parameters
    ----------
    fold_results : list of dict
        Results returned by each fold.

    Returns
    -------
    dict
        Aggregated outputs keyed as in the fold results.
    """
    aggregated = {}

    # Concatenate predictions, fold labels, and subgroup covariates.
    per_sample_keys = {
        'train_true', 'train_pred', 'train_fold_id',
        'val_in_true', 'val_in_pred', 'val_in_fold_id',
        'test_in_true', 'test_in_pred', 'test_in_fold_id',
        'test_out_true', 'test_out_pred', 'test_out_fold_id',
        'Z_train_bin', 'V_train',
        'Z_val_in_bin', 'V_val_in',
        'Z_test_in_bin', 'V_test_in',
        'Z_test_out_bin', 'V_test_out',
    }

    per_fold_object_keys = {'model', 'processor'}

    for key in fold_results[0].keys():
        if key in per_sample_keys:
            arrays = [np.asarray(fold_res[key]) for fold_res in fold_results
                     if fold_res.get(key) is not None]

            if arrays:
                try:
                    aggregated[key] = np.concatenate(arrays, axis=0)
                except Exception as e:
                    print(f"Error concatenating key '{key}': {e}")
                    raise e
            else:
                aggregated[key] = None
        elif key in per_fold_object_keys:
            aggregated[key] = [fold_res[key] for fold_res in fold_results]
        else:
            # Constant across folds: any fold's value is representative.
            aggregated[key] = fold_results[0][key]

    return aggregated


def process_one_loco_run(
    df,
    feature_cols,
    cohort,
    seed,
    processor,
    model_keys,
    model_parameters,
    n_folds,
    use_full_stratification_inner_cv=True,
    use_sample_weights=True,
    include_covariates=True,
    val_size_inner=0.10
):
    """Execute one LOCO experiment for a cohort and seed.

    Parameters
    ----------
    df : DataFrame
        Source data for the experiment.
    feature_cols : list
        Names of feature columns.
    cohort : str
        Cohort held out for testing.
    seed : int
        Random seed for reproducibility.
    processor : transformer
        Preprocessing object applied before modelling.
    model_keys : list
        Identifiers for the estimators to train.
    model_parameters : dict
        Parameter grids keyed by model identifier.
    n_folds : int
        Number of inner cross-validation folds.
    use_sample_weights : bool, default=True
        Whether to use age-sex sample weights.
    include_covariates : bool, default=True
        Whether to pass covariates to pipelines.
    val_size_inner : float, default=0.10
        Validation split proportion for the inner CV fold.

    Returns
    -------
    tuple
        Cohort name, seed, and aggregated results by model.
    """
    print(f"  Starting cohort={cohort}, seed={seed}")

    if use_full_stratification_inner_cv:
        print(f"     Stratified K-Fold ENABLED for inner CV - stratifying by cohort-age-sex-disease")
    
    if use_sample_weights:
        print(f"     Sample weighting ENABLED - balancing age-sex-disease strata")
    
    outer = prepare_loco_data_outer(
        df_sub=df,
        coda_cols=feature_cols,
        loco_cohort=cohort,
        seed=seed
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    estimators = make_estimators(model_parameters, seed, model_keys=model_keys)
    fold_results = {model_key: [] for model_key in estimators}

    for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(outer.X_train, outer.y_train), start=1):

        print(f"    - Fold {fold_idx}/{n_folds}")

        fold_data = prepare_loco_data_inner(
            outer, train_idx, val_idx, fold_idx, 
            use_full_stratification_inner_cv=use_full_stratification_inner_cv,
            use_sample_weights=use_sample_weights,
            val_size_inner=val_size_inner
        )
        
        # Share one independently fitted preprocessor across fold estimators.
        prepared = preprocess_fold_data(
            safe_clone(processor), fold_data, seed, include_covariates=include_covariates
        )

        # Fit an independent estimator for each fold.
        for model_key, estimator in estimators.items():
            result = fit_pipeline_experiment(prepared, safe_clone(estimator), seed)
            # Retain fold membership for fold-averaged metrics.
            for split in LOCO_SPLITS:
                n_split = len(result[f'{split}_true'])
                result[f'{split}_fold_id'] = np.full(n_split, fold_idx, dtype=int)
            fold_results[model_key].append(result)
    
    aggregated = {
        model_key: aggregate_loco_results_across_folds_per_seed(
            fold_results[model_key]
        )
        for model_key in fold_results
    }

    for model_key in aggregated:
        aggregated[model_key]['seed'] = seed
        aggregated[model_key]['loco_cohort'] = cohort
    
    return cohort, seed, aggregated

def validate_loco_config(
    scaled_procs,
    model_keys,
    model_parameters,
    pipeline_keys,
    loco_test_cohorts,
    selected_seeds,
    verbose=True,
):
    """Check the estimator/preprocessor/pipeline configuration before running.

    Catches config drift that would otherwise surface as silently missing
    results downstream: a selected model with no hyperparameters, or
    ``pipeline_keys`` left stale after ``model_keys``/``scaled_procs`` changed.
    Pipeline keys are compared as a set, since ordering carries no meaning.

    Raises
    ------
    ValueError
        If any check fails, listing every problem found.
    """
    model_keys = list(model_keys)
    processor_names = list(scaled_procs)
    pipeline_keys = list(pipeline_keys)
    problems = []

    for name, values in (
        ('model_keys', model_keys),
        ('scaled_procs', processor_names),
        ('pipeline_keys', pipeline_keys),
        ('loco_test_cohorts', list(loco_test_cohorts)),
        ('selected_seeds', list(selected_seeds)),
    ):
        if not values:
            problems.append(f"{name} is empty.")
        duplicates = sorted({v for v in values if values.count(v) > 1})
        if duplicates:
            problems.append(f"{name} contains duplicates: {duplicates}.")

    missing_params = sorted(k for k in model_keys if k not in model_parameters)
    if missing_params:
        problems.append(f"model_parameters has no entry for model(s) {missing_params}.")

    if model_keys and processor_names:
        expected = {f"{m}+{p}" for p in processor_names for m in model_keys}
        actual = set(pipeline_keys)
        if expected != actual:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            detail = []
            if missing:
                detail.append(f"missing {missing}")
            if unexpected:
                detail.append(f"unexpected {unexpected}")
            problems.append(
                "pipeline_keys does not match model_keys x scaled_procs ("
                + "; ".join(detail)
                + "). Rebuild it as [f'{m}+{p}' for p in PROCESSOR_KEYS for m in MODEL_KEYS]."
            )

    if problems:
        raise ValueError(
            "LOCO configuration is invalid:\n  - " + "\n  - ".join(problems)
        )

    if verbose:
        print(
            f" Config valid: {len(model_keys)} model(s) x {len(processor_names)} "
            f"preprocessor(s) = {len(pipeline_keys)} pipeline(s); "
            f"{len(list(loco_test_cohorts))} cohort(s) x {len(list(selected_seeds))} seed(s)"
        )


def run_loco_cv(
    df,
    feature_cols,
    loco_test_cohorts,
    selected_seeds,
    scaled_procs,
    model_keys,
    model_parameters,
    pipeline_keys,
    n_folds=5,
    n_jobs=1,
    backend="loky",
    use_full_stratification_inner_cv=True,
    use_sample_weights=True,
    include_covariates=True,
    val_size_inner=0.10,
    validate=True
):
    """
    Run LOCO cross-validation across cohorts, seeds, and preprocessors.

    The full (preprocessor, cohort, seed) task grid is flattened into a
    single ``joblib.Parallel`` call, so it scales the same way whether
    ``n_jobs`` is 1 (a laptop, running sequentially in-process) or however
    many cores a cluster node/allocation provides -- ``n_jobs`` is the only
    knob that controls parallelism.

    Each task pins its own BLAS/XGBoost threading to 1 (see
    ``threadpoolctl`` below and ``make_estimators``'s XGBoost default).
    This is a reproducibility requirement, not just a performance one:
    multi-threaded BLAS reductions are not guaranteed bit-identical across
    thread counts, so without pinning, the same seed could produce subtly
    different floating-point results depending on ``n_jobs``, machine core
    count, or unrelated system load -- pinning to 1 thread per task makes
    each (preprocessor, cohort, seed) result depend only on that seed,
    regardless of how many tasks run concurrently or on what hardware. To
    use more cores, raise ``n_jobs`` (one task per core is the natural
    ceiling); do not rely on per-task multi-threading.

    Parameters
    ----------
    df : DataFrame
        Full dataset with features and targets.
    feature_cols : list
        Feature column names to include.
    loco_test_cohorts : list
        Cohorts to hold out during LOCO evaluation.
    selected_seeds : list
        Seeds applied for reproducibility.
    scaled_procs : dict
        Mapping from preprocessor name to transformer.
    model_keys : list
        Identifiers of estimators to evaluate.
    model_parameters : dict
        Parameters for each estimator.
    pipeline_keys : list
        Flattened keys combining model and preprocessor.
    n_folds : int, default=5
        Number of inner cross-validation folds.
    n_jobs : int, default=1
        Number of parallel tasks (preprocessor x cohort x seed) to run.
        1 runs sequentially in-process; set it to the number of cores
        available, whether that's a handful on a laptop or many on a
        cluster node.
    backend : str, default="loky"
        Joblib backend used when running in parallel. Any joblib-compatible
        backend works (e.g. a cluster/distributed backend registered via
        ``joblib.register_parallel_backend``), since only ``n_jobs`` and
        ``backend`` are threaded through to ``Parallel``.
    use_sample_weights : bool, default=True
        Whether to use age-sex sample weights.
    include_covariates : bool, default=True
        Whether to pass covariates to the pipelines.
    val_size_inner : float, default=0.10
        Validation split proportion for the inner CV fold.
    validate : bool, default=True
        Check the model/preprocessor/pipeline configuration up front via
        ``validate_loco_config`` and report the result.

    Returns
    -------
    dict
        Aggregated raw outputs and label collections by pipeline key.
    """
    if validate:
        validate_loco_config(
            scaled_procs=scaled_procs,
            model_keys=model_keys,
            model_parameters=model_parameters,
            pipeline_keys=pipeline_keys,
            loco_test_cohorts=loco_test_cohorts,
            selected_seeds=selected_seeds,
        )

    raw = {
        cohort: {key: [] for key in pipeline_keys}
        for cohort in loco_test_cohorts
    }

    labels = {
        key: {"train_true": [], "train_pred": [], "val_in_true": [], "val_in_pred": [], "test_in_true": [], "test_in_pred": [], "test_out_true": [], "test_out_pred": []}
        for key in pipeline_keys
    }

    def _run_one(proc_name, processor, cohort, seed):
        # Limit native thread pools within each parallel task.
        with threadpool_limits(limits=1):
            processor_local = safe_clone(processor)
            _, _, aggregated = process_one_loco_run(
                df=df,
                feature_cols=feature_cols,
                cohort=cohort,
                seed=seed,
                processor=processor_local,
                model_keys=model_keys,
                model_parameters=model_parameters,
                n_folds=n_folds,
                use_full_stratification_inner_cv=use_full_stratification_inner_cv,
                use_sample_weights=use_sample_weights,
                include_covariates=include_covariates,
                val_size_inner=val_size_inner
            )
        return proc_name, cohort, seed, aggregated

    tasks = [
        (proc_name, cohort, seed)
        for proc_name in scaled_procs
        for cohort in loco_test_cohorts
        for seed in selected_seeds
    ]
    mode = "sequentially" if n_jobs == 1 else f"in parallel (n_jobs={n_jobs})"
    print(
        f" Processing {len(tasks)} tasks {mode} "
        f"({len(scaled_procs)} preprocessors x {len(loco_test_cohorts)} cohorts "
        f"x {len(selected_seeds)} seeds)..."
    )

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_run_one)(proc_name, scaled_procs[proc_name], cohort, seed)
        for proc_name, cohort, seed in tasks
    )

    for proc_name, cohort, seed, aggregated in results:
        for model_key in model_keys:
            flat_key = f"{model_key}+{proc_name}"
            collect_loco_results(
                aggregated[model_key],
                raw,
                cohort,
                flat_key,
                labels[flat_key]["train_true"],
                labels[flat_key]["train_pred"],
                labels[flat_key]["val_in_true"],
                labels[flat_key]["val_in_pred"],
                labels[flat_key]["test_in_true"],
                labels[flat_key]["test_in_pred"],
                labels[flat_key]["test_out_true"],
                labels[flat_key]["test_out_pred"],
            )
        print(f"  Done preprocessor={proc_name}, cohort={cohort}, seed={seed}")

    return {'raw': raw, 'labels': labels}

def collect_loco_results(exp_results, results_raw, test_cohort, model_key, train_true_list, train_pred_list, val_in_true_list, val_in_pred_list, test_in_true_list, test_in_pred_list, test_out_true_list, test_out_pred_list):
    '''Collect loco results at the seed & fold level for a given model and cohort'''

    train_true_list.append(exp_results['train_true'])
    train_pred_list.append(exp_results['train_pred'])
    val_in_true_list.append(exp_results['val_in_true'])
    val_in_pred_list.append(exp_results['val_in_pred'])
    test_in_true_list.append(exp_results['test_in_true'])
    test_in_pred_list.append(exp_results['test_in_pred'])
    test_out_true_list.append(exp_results['test_out_true'])
    test_out_pred_list.append(exp_results['test_out_pred'])
    results_raw[test_cohort][model_key].append(exp_results)

def _fold_average_metric_dicts(per_fold_dicts):
    """Combine per-fold results: sum support counts, average scoring metrics.

    Metrics are averaged only over the folds where they could be estimated. That
    denominator varies between subgroups, so it is reported alongside the values
    as ``n_folds`` and ``n_folds_estimable`` rather than left implicit: a
    subgroup estimable in 2 of 5 folds has a mean built from 2 numbers, and
    those 2 folds are not a random sample of the 5.
    """
    if not per_fold_dicts:
        return {}

    keys = set()
    for fold_dict in per_fold_dicts:
        keys.update(fold_dict.keys())

    averaged = {}
    for key in keys:
        values = [
            fold_dict[key] for fold_dict in per_fold_dicts
            if fold_dict.get(key) is not None and not pd.isna(fold_dict[key])
        ]
        if key in SUPPORT_KEYS:
            averaged[key] = float(np.sum(values)) if values else 0.0
        elif key == ESTIMABLE_KEY:
            continue  # replaced by the explicit fold counters below
        elif values:
            averaged[key] = float(np.mean(values))

    averaged['n_folds'] = float(len(per_fold_dicts))
    if ESTIMABLE_KEY in keys:
        averaged['n_folds_estimable'] = float(sum(
            1 for fold_dict in per_fold_dicts if fold_dict.get(ESTIMABLE_KEY)
        ))
    return averaged


def compute_fold_averaged_disease_metrics(y_true, y_pred, fold_id, metrics):
    """Compute disease-level metrics per fold, then average unweighted across folds."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fold_id = np.asarray(fold_id)

    per_fold = []
    for fold in np.unique(fold_id):
        mask = fold_id == fold
        per_fold.append(compute_scoring_metrics(y_true[mask], y_pred[mask], metrics=metrics))
    return _fold_average_metric_dicts(per_fold)


def _subgroup_labels_for_var(subgroup_var, Z_bin, V):
    """Return the per-sample group-label array for a non-intersection subgroup variable."""
    if subgroup_var == 'age':
        return Z_bin[:, 1]
    if subgroup_var == 'sex':
        return Z_bin[:, 0]
    if subgroup_var == 'location':
        return cohort_index_from_onehot(V)
    raise ValueError(f"Unsupported subgroup_var '{subgroup_var}' for label extraction.")


def compute_fold_averaged_subgroup_metrics(
    y_true, y_pred, fold_id, Z_bin, V, subgroup_var, metrics, subgroup_names
):
    """Average subgroup metrics across valid fold slices."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fold_id = np.asarray(fold_id)
    Z_bin = np.asarray(Z_bin)
    V = np.asarray(V)
    folds = np.unique(fold_id)

    if subgroup_var == 'age_sex':
        per_fold_by_group = defaultdict(lambda: defaultdict(list))
        for fold in folds:
            mask = fold_id == fold
            fold_perf = calculate_intersection_performance(
                y_true[mask], y_pred[mask],
                Z_bin[mask, 1], Z_bin[mask, 0],
                age_subgroup_names=subgroup_names['age'],
                sex_subgroup_names=subgroup_names['sex'],
                metrics=metrics,
            )
            for age_group, sex_dict in fold_perf.items():
                for sex_group, group_metrics in sex_dict.items():
                    per_fold_by_group[age_group][sex_group].append(group_metrics)

        return {
            age_group: {
                sex_group: _fold_average_metric_dicts(fold_dicts)
                for sex_group, fold_dicts in sex_dict.items()
            }
            for age_group, sex_dict in per_fold_by_group.items()
        }

    names_by_var = {
        'age': subgroup_names['age'],
        'sex': subgroup_names['sex'],
        'location': subgroup_names['location'],
    }
    per_fold_by_group = defaultdict(list)
    for fold in folds:
        mask = fold_id == fold
        labels = _subgroup_labels_for_var(subgroup_var, Z_bin[mask], V[mask])
        fold_perf = calculate_subgroup_performance(
            y_true[mask], y_pred[mask], labels,
            group_name=subgroup_var,
            subgroup_names=names_by_var.get(subgroup_var),
            metrics=metrics,
        )
        for group, group_metrics in fold_perf.items():
            per_fold_by_group[group].append(group_metrics)

    return {
        group: _fold_average_metric_dicts(fold_dicts)
        for group, fold_dicts in per_fold_by_group.items()
    }


def summarize_loco_results_across_folds_and_seeds(results, variables, all_metrics, set_types, model_keys):
    """Aggregate per-cohort experimental outputs across seeds.

    Parameters
    ----------
    results : dict
        Raw results grouped by model key.
    variables : list
        Variables to analyze (e.g., disease, age_sex).
    all_metrics : list
        Names of scoring metrics to compute.
    set_types : list
        Dataset partitions to summarise.
    model_keys : list
        Model identifiers to aggregate.

    Returns
    -------
    tuple
        DataFrames with overall metrics and subgroup analyses.
    """
    
    # defaultdict avoids explicit initialization at each nesting level.
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    subgroup_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    subgroup_vars = [v for v in variables if v != 'disease']

    for model_key in model_keys:
        for run_index, res in enumerate(results[model_key]):
            subgroup_names = (
                create_demographic_subgroup_names(res['age_ranges'], res['cohort_locations'])
                if subgroup_vars else None
            )

            for set_type in set_types:
                y_true = res[f'{set_type}_true']
                y_pred = res[f'{set_type}_pred']
                fold_id = res[f'{set_type}_fold_id']

                disease_metrics = compute_fold_averaged_disease_metrics(
                    y_true, y_pred, fold_id, all_metrics
                )
                metrics[model_key][set_type]['disease'].append(disease_metrics)

                if subgroup_vars:
                    Z_bin = res[f'Z_{set_type}_bin']
                    V = res[f'V_{set_type}']

                for subgroup_var in subgroup_vars:
                    subgroup_results = compute_fold_averaged_subgroup_metrics(
                        y_true, y_pred, fold_id, Z_bin, V, subgroup_var, all_metrics, subgroup_names
                    )
                    if subgroup_var == 'age_sex':
                        for age_group, age_dict in subgroup_results.items():
                            if age_group not in subgroup_metrics[model_key][set_type][subgroup_var]:
                                subgroup_metrics[model_key][set_type][subgroup_var][age_group] = {}
                            for sex_group, group_metrics in age_dict.items():
                                if sex_group not in subgroup_metrics[model_key][set_type][subgroup_var][age_group]:
                                    subgroup_metrics[model_key][set_type][subgroup_var][age_group][sex_group] = []
                                subgroup_metrics[model_key][set_type][subgroup_var][age_group][sex_group].append(group_metrics)
                    else:
                        for group, group_metrics in subgroup_results.items():
                            subgroup_metrics[model_key][set_type][subgroup_var][group].append(group_metrics)


    final_results = {}
    subgroup_analysis_final = {}

    for set_type in set_types:
        final_results[set_type] = {}
        subgroup_analysis_final[set_type] = {}

        for var in variables:
            if var == 'disease':
                aggregated_disease_results = aggregate_all_metrics(
                    metrics, model_keys, set_type, var, all_metrics
                )
                rows = []
                for model_key, agg in aggregated_disease_results.items():
                    row = {'Model': model_key}
                    for metric, values in agg.items():
                        row[f"{metric} Mean"] = values['Mean']
                        row[f"{metric} SEM"] = values['SEM']
                        row[f"{metric} CI"] = values['CI']
                    rows.append(row)
                final_results[set_type][var] = pd.DataFrame(rows)

        for subgroup_var in [v for v in variables if v != 'disease']:
            subgroup_analysis_final[set_type][subgroup_var] = {}
            # Groups are keyed off the first model; all models share the same
            # subgroup structure, so any model's keys would do.
            ref_model = model_keys[0]
            if subgroup_var == 'age_sex':
                for age_group, sex_dict in subgroup_metrics[ref_model][set_type][subgroup_var].items():
                    subgroup_analysis_final[set_type][subgroup_var][age_group] = {}
                    for sex_group in sex_dict.keys():
                        subgroup_analysis_final[set_type][subgroup_var][age_group][sex_group] = {}
                        for model_key in model_keys:
                            if age_group in subgroup_metrics[model_key][set_type][subgroup_var]:
                                if sex_group in subgroup_metrics[model_key][set_type][subgroup_var][age_group]:
                                    model_group_agg = summarize_scoring_metrics(
                                        subgroup_metrics[model_key][set_type][subgroup_var][age_group][sex_group]
                                    )
                                    if model_key not in subgroup_analysis_final[set_type][subgroup_var][age_group][sex_group]:
                                        subgroup_analysis_final[set_type][subgroup_var][age_group][sex_group][model_key] = {}
                                    for metric in all_metrics:
                                        if metric in model_group_agg:
                                            subgroup_analysis_final[set_type][subgroup_var][age_group][sex_group][model_key][metric] = {
                                                'Mean': model_group_agg[metric]['Mean'],
                                                'SEM': model_group_agg[metric]['SEM'],
                                                'CI': model_group_agg[metric]['CI']
                                            }
                            else:
                                warnings.warn(
                                    f"Group '{age_group}-{sex_group}' not found in {model_key} "
                                    f"for subgroup '{subgroup_var}' in set '{set_type}'."
                                )
            else:
                # Single-level subgroups: age, sex, location.
                for group in subgroup_metrics[ref_model][set_type][subgroup_var].keys():
                    subgroup_analysis_final[set_type][subgroup_var][group] = {}
                    for model_key in model_keys:
                        if group in subgroup_metrics[model_key][set_type][subgroup_var]:
                            model_group_agg = summarize_scoring_metrics(
                                subgroup_metrics[model_key][set_type][subgroup_var][group]
                            )
                            subgroup_analysis_final[set_type][subgroup_var][group][model_key] = {}
                            for metric in all_metrics:
                                if metric in model_group_agg:
                                    subgroup_analysis_final[set_type][subgroup_var][group][model_key][metric] = {
                                        'Mean': model_group_agg[metric]['Mean'],
                                        'SEM': model_group_agg[metric]['SEM'],
                                        'CI': model_group_agg[metric]['CI']
                                    }
                        else:
                            warnings.warn(
                                f"Group '{group}' not found in {model_key} for subgroup "
                                f"'{subgroup_var}' in set '{set_type}'."
                            )
            
    return final_results, subgroup_analysis_final

def aggregate_loco_results_per_test_cohort(
    results_loco_raw,
    loco_test_cohorts,
    pipeline_keys,
    vars=None,
    metrics=None,
    sets=None,
):
    """Summarise LOCO outputs for each test cohort.

    Parameters
    ----------
    results_loco_raw : dict
        Raw LOCO outputs grouped by cohort.
    loco_test_cohorts : list
        Cohorts used as held-out test sets.
    pipeline_keys : list
        Flattened pipeline identifiers.
    vars : list, optional
        Variables to analyse, by default ``['disease', 'age_sex']``.
    metrics : list, optional
        Metrics to compute, by default ``['AUROC', 'LogLoss']``.
    sets : list, optional
        Dataset partitions to include, by default ``['test_in', 'test_out']``.

    Returns
    -------
    tuple
        Summary tables per cohort and subgroup analyses.
    """
    vars = vars or ['disease', 'age_sex']
    metrics = metrics or ['AUROC', 'LogLoss']
    sets = sets or ['test_in', 'test_out']

    results_loco_test_cohorts = {cohort: {} for cohort in loco_test_cohorts}
    subgroup_analyses_loco = {cohort: {} for cohort in loco_test_cohorts}

    for cohort in loco_test_cohorts:
        print(f"Results for Test Cohort: {cohort}")
        
        results_test_cohort, subgroup_analysis = summarize_loco_results_across_folds_and_seeds(
            results_loco_raw[cohort],
            vars,
            metrics,
            sets,
            pipeline_keys
        )

        results_loco_test_cohorts[cohort] = results_test_cohort
        subgroup_analyses_loco[cohort] = subgroup_analysis

    return results_loco_test_cohorts, subgroup_analyses_loco

def meta_analyze_fixed_effects(weighted_means, weighted_sems):
    """Compute fixed-effects pooled mean and heterogeneity statistics."""
    weights_fixed = 1.0 / (weighted_sems ** 2)
    pooled_mean_fixed = float(np.average(weighted_means, weights=weights_fixed))
    q = float(np.sum(weights_fixed * ((weighted_means - pooled_mean_fixed) ** 2)))
    df_q = len(weighted_means) - 1

    if len(weighted_means) >= 2 and q > 0 and df_q > 0:
        i2_value = max(0.0, ((q - df_q) / q) * 100)
    elif len(weighted_means) == 1:
        i2_value = 0.0
    else:
        i2_value = 0.0

    return weights_fixed, pooled_mean_fixed, q, df_q, i2_value

def meta_analyze_random_effects(weighted_means, weighted_sems, weights_fixed, q, df_q):
    """Compute random-effects pooled mean, weights, and naive SEM."""
    sum_weights = np.sum(weights_fixed)
    c_denom = sum_weights - (np.sum(weights_fixed ** 2) / sum_weights) if sum_weights > 0 else 0.0
    if c_denom > 0:
        tau_squared = max(0.0, (q - df_q) / c_denom)
    else:
        tau_squared = 0.0

    weights_random = 1.0 / (weighted_sems ** 2 + tau_squared)
    pooled_mean = float(np.average(weighted_means, weights=weights_random))
    sum_random = np.sum(weights_random)
    pooled_sem_naive = float(np.sqrt(1.0 / sum_random)) if sum_random > 0 else np.nan

    return tau_squared, weights_random, pooled_mean, pooled_sem_naive

def represent_ci(ci_value):
    """Convert CI representations into numeric tuples."""
    if isinstance(ci_value, (list, tuple)) and len(ci_value) == 2:
        try:
            return float(ci_value[0]), float(ci_value[1])
        except (TypeError, ValueError):
            return (np.nan, np.nan)
    if isinstance(ci_value, str):
        stripped = ci_value.strip().strip("[]()")
        parts = [p.strip() for p in stripped.split(',')]
        if len(parts) == 2:
            try:
                return float(parts[0]), float(parts[1])
            except ValueError:
                return (np.nan, np.nan)
    return (np.nan, np.nan)


def infer_sem_from_ci(sem_value, ci_value):
    """Infer SEM from CI if not provided."""
    if sem_value is not None and not pd.isna(sem_value):
        return float(sem_value)
    low, high = represent_ci(ci_value)
    if not pd.isna(low) and not pd.isna(high):
        half_width = abs(high - low) / 2
        if half_width > 0:
            return half_width / 1.96
    return np.nan


def classify_heterogeneity(i2_value):
    """Classify heterogeneity level."""
    if i2_value is None or pd.isna(i2_value):
        return "Not available"
    if i2_value >= 75:
        return "High"
    if i2_value >= 50:
        return "Moderate"
    if i2_value >= 25:
        return "Low"
    return "Negligible"


def compute_pooled_stats(
    means,
    sems,
    cis,
    ci_lvl,
    use_random_effects_threshold=50.0,
):
    """Compute pooled statistics using fixed/random-effects meta-analysis."""
    cleaned = []
    for mean_val, sem_val, ci_val in zip(means, sems, cis):
        if mean_val is None or pd.isna(mean_val):
            continue
        inferred_sem = infer_sem_from_ci(sem_val, ci_val)
        cleaned.append((float(mean_val), inferred_sem))

    if not cleaned:
        return None

    means_arr = np.array([m for m, _ in cleaned], dtype=float)
    sems_arr = np.array([s for _, s in cleaned], dtype=float)
    n = len(means_arr)

    between_sd = float(np.std(means_arr, ddof=1)) if n > 1 else 0.0
    range_min = float(np.min(means_arr))
    range_max = float(np.max(means_arr))

    valid_weights_mask = (~np.isnan(sems_arr)) & (sems_arr > 0)

    if valid_weights_mask.sum() < 1:
        pooled_mean = float(np.mean(means_arr))
        pooled_sem = float(np.std(means_arr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan

        df = max(n - 1, 1)
        crit = float(t.ppf((1 + ci_lvl) / 2, df)) if n > 1 else 1.96
        if not pd.isna(pooled_sem):
            pooled_ci = (
                pooled_mean - crit * pooled_sem,
                pooled_mean + crit * pooled_sem
            )
        else:
            pooled_ci = (np.nan, np.nan)

        return {
            'mean': pooled_mean,
            'sem': pooled_sem,
            'ci': pooled_ci,
            'between_sd': between_sd,
            'range_min': range_min,
            'range_max': range_max,
            'i2': 0.0 if n == 1 else np.nan,
            'n': n,
            'label': "Not available",
            'pooling_method': 'Unweighted',
            'tau_squared': np.nan,
            'prediction_interval': (np.nan, np.nan)
        }

    weighted_means = means_arr[valid_weights_mask]
    weighted_sems = sems_arr[valid_weights_mask]
    n_valid = valid_weights_mask.sum()

    weights_fixed, pooled_mean_fixed, q, df_q, i2_value = meta_analyze_fixed_effects(
        weighted_means,
        weighted_sems
    )

    use_random_effects = (i2_value >= use_random_effects_threshold and n_valid >= 3)
    tau_squared = 0.0
    pooling_method = 'Fixed-effects'
    sum_weights_fixed = np.sum(weights_fixed)
    pooled_sem = float(np.sqrt(1.0 / sum_weights_fixed)) if sum_weights_fixed > 0 else np.nan
    pooled_mean = float(pooled_mean_fixed)

    if use_random_effects:
        tau_squared, _, pooled_mean, pooled_sem = meta_analyze_random_effects(
            weighted_means,
            weighted_sems,
            weights_fixed,
            q,
            df_q
        )
        pooling_method = 'Random-effects'

    df = max(n_valid - 1, 1)
    crit = float(t.ppf((1 + ci_lvl) / 2, df)) if n_valid > 1 else 1.96
    if not pd.isna(pooled_sem):
        pooled_ci = (
            pooled_mean - crit * pooled_sem,
            pooled_mean + crit * pooled_sem
        )
    else:
        pooled_ci = (np.nan, np.nan)

    prediction_interval = (np.nan, np.nan)

    if i2_value >= 75:
        warnings.warn(
            f"High heterogeneity detected (I^2={i2_value:.1f}%). "
            f"Using {pooling_method}. "
            "Interpret pooled estimates with caution - results vary substantially across cohorts."
        )

    return {
        'mean': pooled_mean,
        'sem': pooled_sem,
        'ci': pooled_ci,
        'between_sd': between_sd,
        'range_min': range_min,
        'range_max': range_max,
        'i2': i2_value,
        'n': n,
        'label': classify_heterogeneity(i2_value),
        'pooling_method': pooling_method,
        'ci_method': 'Standard',
        'tau_squared': tau_squared,
        'prediction_interval': prediction_interval
    }


def _filter_and_validate_loco_inputs(
    results_loco_test_cohorts,
    subgroup_analyses_loco,
    test_cohorts,
):
    """Filter selected cohorts and validate minimum cohort count."""
    if not results_loco_test_cohorts:
        raise ValueError("results_loco_test_cohorts is empty; nothing to summarize.")

    if test_cohorts is not None:
        if isinstance(test_cohorts, str):
            requested_cohorts = [test_cohorts]
        else:
            requested_cohorts = list(test_cohorts)
        if not requested_cohorts:
            raise ValueError("test_cohorts is empty; provide at least one cohort name.")
        missing = [cohort for cohort in requested_cohorts if cohort not in results_loco_test_cohorts]
        if missing:
            raise ValueError(
                "Requested test_cohorts not found in results: " + ", ".join(missing)
            )
        selected_cohorts = requested_cohorts
    else:
        selected_cohorts = list(results_loco_test_cohorts.keys())

    results_loco_test_cohorts = {
        cohort: results_loco_test_cohorts[cohort]
        for cohort in selected_cohorts
    }

    if subgroup_analyses_loco:
        subgroup_analyses_loco = {
            cohort: subgroup_analyses_loco.get(cohort, {})
            for cohort in selected_cohorts
            if cohort in subgroup_analyses_loco
        }

    n_cohorts = len(results_loco_test_cohorts)
    if n_cohorts < 2:
        raise ValueError(f"Need at least 2 LOCO rounds for meta-analysis, got {n_cohorts}")

    if n_cohorts < 3:
        warnings.warn(
            f"Only {n_cohorts} LOCO rounds available. "
            "Statistical estimates will be very imprecise."
        )

    return results_loco_test_cohorts, subgroup_analyses_loco


def _infer_loco_summary_defaults(results_loco_test_cohorts, sets, metrics, model_keys):
    """Infer missing sets, metrics, and model keys from cohort results."""
    if sets is None:
        first_cohort = next(iter(results_loco_test_cohorts.values()))
        sets = list(first_cohort.keys())

    if metrics is None:
        metrics = []
        for cohort_data in results_loco_test_cohorts.values():
            for set_dict in cohort_data.values():
                disease_df = set_dict.get('disease')
                if isinstance(disease_df, pd.DataFrame) and not disease_df.empty:
                    metrics = [col.replace(' Mean', '') for col in disease_df.columns if col.endswith(' Mean')]
                    break
            if metrics:
                break
        if not metrics:
            raise ValueError("Unable to infer metrics from results_loco_test_cohorts.")

    if model_keys is None:
        for cohort_data in results_loco_test_cohorts.values():
            for set_dict in cohort_data.values():
                disease_df = set_dict.get('disease')
                if isinstance(disease_df, pd.DataFrame) and not disease_df.empty:
                    inferred = disease_df['Model'].tolist()
                    model_keys = list(dict.fromkeys(inferred))
                    break
            if model_keys:
                break
        if not model_keys:
            raise ValueError("Unable to infer model keys from results_loco_test_cohorts.")

    return sets, metrics, model_keys


def _compute_worst_cohort(metric, cohort_means):
    """Return worst-performing cohort name for a metric.

    Ranks by distance from the metric's ideal rather than by raw value, so that
    a target-valued metric such as the calibration slope is not scored as if
    smaller were better. Cohorts whose value is NaN are not eligible to be named
    worst, since a missing estimate is not a poor one.
    """
    if not cohort_means:
        return None
    scored = [
        (name, metric_distance_from_ideal(metric, value))
        for name, value in cohort_means
    ]
    scored = [(name, loss) for name, loss in scored if not pd.isna(loss)]
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])[0]


def _round_summary_frame(df):
    """Round summary numeric columns using existing display rules."""
    if df.empty:
        return df

    for col in [
        'Pooled Mean', 'Lower CI', 'Upper CI', 'Pooled SEM',
        'Between SD', 'Range Min', 'Range Max', 'Tau²',
        'Prediction Interval Low', 'Prediction Interval High'
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(3)

    if 'I2 (%)' in df:
        df['I2 (%)'] = pd.to_numeric(df['I2 (%)'], errors='coerce').round(1)

    return df


def _build_disease_summary(
    results_loco_test_cohorts,
    sets,
    model_keys,
    metrics,
    ci_level,
    use_random_effects_threshold,
):
    """Build pooled disease-level summaries across cohorts."""
    disease_summary = {}

    for set_type in sets:
        rows = []
        for model_key in model_keys:
            for metric in metrics:
                mean_vals, sem_vals, ci_vals = [], [], []
                cohort_means = []

                for cohort_name, cohort_data in results_loco_test_cohorts.items():
                    set_dict = cohort_data.get(set_type, {})
                    disease_df = set_dict.get('disease')
                    if not isinstance(disease_df, pd.DataFrame) or disease_df.empty:
                        continue

                    row = disease_df[disease_df['Model'] == model_key]
                    if row.empty:
                        continue

                    row_data = row.iloc[0]
                    mean_val = row_data.get(f"{metric} Mean")
                    mean_vals.append(mean_val)
                    sem_vals.append(row_data.get(f"{metric} SEM"))
                    ci_vals.append(row_data.get(f"{metric} CI"))

                    if mean_val is not None and not pd.isna(mean_val):
                        cohort_means.append((cohort_name, mean_val))

                stats = compute_pooled_stats(
                    mean_vals,
                    sem_vals,
                    ci_vals,
                    ci_level,
                    use_random_effects_threshold=use_random_effects_threshold,
                )
                if stats is None:
                    continue

                rows.append({
                    'Model': model_key,
                    'Metric': metric,
                    'Pooled Mean': stats['mean'],
                    'Lower CI': stats['ci'][0],
                    'Upper CI': stats['ci'][1],
                    'Pooled SEM': stats['sem'],
                    'Between SD': stats['between_sd'],
                    'Range Min': stats['range_min'],
                    'Range Max': stats['range_max'],
                    'I2 (%)': stats['i2'],
                    'Heterogeneity': stats['label'],
                    'Cohorts': stats['n'],
                    'Pooling Method': stats['pooling_method'],
                    'Tau²': stats['tau_squared'],
                    'Prediction Interval Low': stats['prediction_interval'][0],
                    'Prediction Interval High': stats['prediction_interval'][1],
                    'Worst Cohort': _compute_worst_cohort(metric, cohort_means)
                })

        disease_df_summary = _round_summary_frame(pd.DataFrame(rows))
        disease_summary[set_type] = disease_df_summary

    return disease_summary


def _infer_subgroup_vars(subgroup_analyses_loco, subgroup_vars):
    """Infer subgroup variable names, excluding disease."""
    if subgroup_vars is None:
        try:
            first_cohort = next(iter(subgroup_analyses_loco.values()))
        except StopIteration:
            subgroup_vars = []
        else:
            first_set = next(iter(first_cohort.keys())) if first_cohort else None
            subgroup_vars = list(first_cohort[first_set].keys()) if first_set else []

    return [var for var in subgroup_vars if var != 'disease']


def _collect_group_keys_for_subgroup(subgroup_analyses_loco, set_type, subgroup_var):
    """Collect all subgroup keys present across cohorts for one subgroup var."""
    group_keys = set()
    for cohort_data in subgroup_analyses_loco.values():
        subgroup_dict = cohort_data.get(set_type, {}).get(subgroup_var)
        if not subgroup_dict:
            continue
        if subgroup_var == 'age_sex':
            for age_group, sex_dict in subgroup_dict.items():
                for sex_group in sex_dict:
                    group_keys.add((age_group, sex_group))
        else:
            group_keys.update(subgroup_dict.keys())
    return sorted(group_keys)


def _build_subgroup_summary(
    subgroup_analyses_loco,
    subgroup_vars,
    sets,
    model_keys,
    metrics,
    ci_level,
    use_random_effects_threshold,
):
    """Build pooled subgroup summaries across cohorts."""
    subgroup_summary = {}

    for set_type in sets:
        subgroup_summary[set_type] = {}
        for subgroup_var in subgroup_vars:
            group_rows = {}
            group_keys = _collect_group_keys_for_subgroup(
                subgroup_analyses_loco,
                set_type,
                subgroup_var,
            )

            for group_key in group_keys:
                group_label = (
                    f"{group_key[0]} | {group_key[1]}" if isinstance(group_key, tuple) else str(group_key)
                )

                rows = []
                for model_key in model_keys:
                    for metric in metrics:
                        mean_vals, sem_vals, ci_vals = [], [], []

                        for cohort_data in subgroup_analyses_loco.values():
                            subgroup_dict = cohort_data.get(set_type, {}).get(subgroup_var)
                            if not subgroup_dict:
                                continue

                            if isinstance(group_key, tuple):
                                age_group, sex_group = group_key
                                model_metrics = subgroup_dict.get(age_group, {}).get(sex_group, {}).get(model_key, {})
                            else:
                                model_metrics = subgroup_dict.get(group_key, {}).get(model_key, {})

                            metric_stats = model_metrics.get(metric)
                            if not metric_stats:
                                continue

                            mean_vals.append(metric_stats.get('Mean'))
                            sem_vals.append(metric_stats.get('SEM'))
                            ci_vals.append(metric_stats.get('CI'))

                        stats = compute_pooled_stats(
                            mean_vals,
                            sem_vals,
                            ci_vals,
                            ci_level,
                            use_random_effects_threshold=use_random_effects_threshold,
                        )
                        if stats is None:
                            continue

                        rows.append({
                            'Group': group_label,
                            'Model': model_key,
                            'Metric': metric,
                            'Pooled Mean': stats['mean'],
                            'Lower CI': stats['ci'][0],
                            'Upper CI': stats['ci'][1],
                            'Pooled SEM': stats['sem'],
                            'Between SD': stats['between_sd'],
                            'Range Min': stats['range_min'],
                            'Range Max': stats['range_max'],
                            'I2 (%)': stats['i2'],
                            'Heterogeneity': stats['label'],
                            'Cohorts': stats['n'],
                            'Pooling Method': stats['pooling_method'],
                            'Tau²': stats['tau_squared'],
                            'Prediction Interval Low': stats['prediction_interval'][0],
                            'Prediction Interval High': stats['prediction_interval'][1]
                        })

                subgroup_df = _round_summary_frame(pd.DataFrame(rows))
                group_rows[group_label] = subgroup_df

            subgroup_summary[set_type][subgroup_var] = group_rows

    return subgroup_summary

def summarize_loco_results_all(
    results_loco_test_cohorts,
    subgroup_analyses_loco=None,
    model_keys=None,
    sets=None,
    metrics=None,
    test_cohorts=None,
    subgroup_vars=None,
    ci_level=0.95,
    use_random_effects_threshold=50.0,
):
    """
    Aggregate LOCO summaries across all cohorts and report heterogeneity.
    
    Assumes results_loco_test_cohorts contains one entry per LOCO round,
    where each entry's Mean/SEM have been computed by averaging over
    multiple random seeds (technical replicates).
    
    Parameters
    ----------
    results_loco_test_cohorts : dict
        Results from each LOCO round (one entry per held-out cohort)
    subgroup_analyses_loco : dict, optional
        Subgroup-specific results from each LOCO round
    model_keys : list, optional
        List of model identifiers to analyze
    sets : list, optional
        List of evaluation sets (e.g., ['test_in', 'test_oos'])
    metrics : list, optional
        List of metrics to analyze (e.g., ['AUROC', 'AUPRC'])
    test_cohorts : list, optional
        Subset of cohort names to include in the meta-analysis
    subgroup_vars : list, optional
        List of subgroup variables to analyze
    ci_level : float, default=0.95
        Confidence level for intervals
    use_random_effects_threshold : float, default=50.0
        I^2 threshold above which to use random-effects pooling

    Returns
    -------
    dict
        Dictionary with 'disease' and 'subgroups' summaries
    """

    results_loco_test_cohorts, subgroup_analyses_loco = _filter_and_validate_loco_inputs(
        results_loco_test_cohorts,
        subgroup_analyses_loco,
        test_cohorts,
    )

    sets, metrics, model_keys = _infer_loco_summary_defaults(
        results_loco_test_cohorts,
        sets,
        metrics,
        model_keys,
    )

    disease_summary = _build_disease_summary(
        results_loco_test_cohorts,
        sets,
        model_keys,
        metrics,
        ci_level,
        use_random_effects_threshold,
    )

    subgroup_summary = {}
    if subgroup_analyses_loco:
        subgroup_vars = _infer_subgroup_vars(subgroup_analyses_loco, subgroup_vars)
        subgroup_summary = _build_subgroup_summary(
            subgroup_analyses_loco,
            subgroup_vars,
            sets,
            model_keys,
            metrics,
            ci_level,
            use_random_effects_threshold,
        )

    return {
        'disease': disease_summary,
        'subgroups': subgroup_summary
    }
