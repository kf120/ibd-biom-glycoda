# -*- coding: utf-8 -*-
"""
@author: Kostis Flevaris
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from ibd_biom_glycoda.utils.helpers import seed_everything, safe_column_access
from ibd_biom_glycoda.evaluation.subgroup_analysis import create_subgroup_metrics_dict


def combine_features_with_covariates(X, Z):
    """
    Concatenate primary features with covariates.
    
    Parameters
    ----------
    X : array-like
        Primary features
    Z : array-like
        Covariates
        
    Returns
    -------
    ndarray
        Concatenated features
    """
    return np.concatenate([X, Z], axis=1)


def create_demographic_subgroup_names(age_ranges, cohort_locations):
    """
    Create human-readable names for demographic subgroups.
    
    Parameters
    ----------
    age_ranges : array-like
        Age bin boundaries
    cohort_locations : array-like
        Cohort location identifiers
        
    Returns
    -------
    dict
        Dictionary with 'age', 'sex', and 'location' keys mapping to name dictionaries
    """
    # Age group names
    age_names = {}
    for i in range(len(age_ranges) - 1):
        if i == 0:
            age_names[i] = f"<{int(age_ranges[i+1])}"
        elif i < len(age_ranges) - 2:
            age_names[i] = f"{int(age_ranges[i])}-{int(age_ranges[i+1])}"
        else:
            age_names[i] = f">{int(age_ranges[i])}"
    
    # Sex group names
    sex_names = {0: 'M', 1: 'F'}
    
    # Location names
    location_names = {i: loc for i, loc in enumerate(cohort_locations[0])}
    
    return {
        'age': age_names,
        'sex': sex_names,
        'location': location_names
    }


def build_results_dict(estimator, preprocessor, data, predictions, metadata):
    """
    Collect model artefacts, predictions, and metadata in one dict.

    Parameters
    ----------
    estimator : estimator
        Trained model instance.
    preprocessor : transformer
        Fitted preprocessing pipeline.
    data : dict
        Original data splits used during training and evaluation.
    predictions : dict
        Uncalibrated probability outputs.
    metadata : dict
        Supporting metadata such as age ranges and locations.

    Returns
    -------
    dict
        Aggregated experiment results.
    """
    return {
        'model': estimator,
        'processor': preprocessor,
        'train_true': np.array(data['train']['y']),

        # Training predictions
        'train_true': np.array(data['train']['y']),
        'train_pred': np.array(predictions['train']),
        
        # Validation predictions
        'val_in_true': np.array(data['val_in']['y']),
        'val_in_pred': np.array(predictions['val_in']),
        
        # Test predictions
        'test_in_true': np.array(data['test_in']['y']),
        'test_in_pred': np.array(predictions['test_in']),
        'test_out_true': np.array(data['test_out']['y']),
        'test_out_pred': np.array(predictions['test_out']),
        
        # Training data
        'X_train': data['train']['X'],
        'Z_train': data['train']['Z'],
        'Z_train_bin': data['train']['Z_bin'],
        'V_train': data['train']['V'],
        
        # Validation data
        'X_val_in': data['val_in']['X'],
        'Z_val_in': data['val_in']['Z'],
        'Z_val_in_bin': data['val_in']['Z_bin'],
        'V_val_in': data['val_in']['V'],
        
        # Test data
        'X_test_in': data['test_in']['X'],
        'Z_test_in': data['test_in']['Z'],
        'V_test_in': data['test_in']['V'],
        'X_test_out': data['test_out']['X'],
        'Z_test_out': data['test_out']['Z'],
        'V_test_out': data['test_out']['V'],
        
        # Metadata
        'age_ranges': metadata['age_ranges'],
        'cohort_locations': metadata['cohort_locations'],
        'num_locations': data['train']['V'].shape[1],
        'num_age_bins': len(np.unique(safe_column_access(data['train']['Z'], 1))),
        'subgroup_metrics': None  # Will be populated later
    }

def unpack_train_data(data_dict):
    """Return training split from the raw data dictionary."""
    return {
        'X': data_dict['X_train'],
        'Z': data_dict['Z_train'],
        'Z_bin': data_dict['Z_train_bin'],
        'y': data_dict['y_train'],
        'V': data_dict['V_train']
    }


def unpack_validation_data(data_dict):
    """Return validation split if present in the raw data dictionary."""    
    return {
        'X': data_dict['X_val_in'],
        'Z': data_dict['Z_val_in'],
        'Z_bin': data_dict['Z_val_in_bin'],
        'y': data_dict['y_val_in'],
        'V': data_dict['V_val_in']
    }


def unpack_test_in_data(data_dict):
    """Return in-sample test split from the raw data dictionary."""
    return {
        'X': data_dict['X_test_in'],
        'Z': data_dict['Z_test_in'],
        'Z_bin': data_dict['Z_test_in_bin'],
        'y': data_dict['y_test_in'],
        'V': data_dict['V_test_in']
    }


def unpack_test_out_data(data_dict):
    """Return out-of-sample test split from the raw data dictionary."""
    return {
        'X': data_dict['X_test_out'],
        'Z': data_dict['Z_test_out'],
        'Z_bin': data_dict['Z_test_out_bin'],
        'y': data_dict['y_test_out'],
        'V': data_dict['V_test_out']
    }


def unpack_metadata(data_dict):
    """Return metadata entries from the raw data dictionary."""
    return {
        'age_ranges': data_dict['age_ranges'],
        'cohort_locations': data_dict['cohort_locations'],
        'ohe_cohort': data_dict.get('ohe_cohort'),
        'scaler': data_dict.get('scaler'),
        'GP_cols': data_dict.get('GP_cols')
    }


def preprocess_all_splits(preprocessor, train_X, val_in_X, test_in_X, test_out_X):
    """
    Fit the preprocessor on training data and transform every split.

    Parameters
    ----------
    preprocessor : transformer
        Preprocessing pipeline with ``fit`` and ``transform``.
    train_X : array-like
        Training features.
    val_in_X : array-like or None
        Validation features, if available.
    test_in_X : array-like
        In-sample test features.
    test_out_X : array-like
        Out-of-sample test features.

    Returns
    -------
    dict
        Preprocessed feature arrays keyed by split name.
    """
    preprocessor.fit(train_X)
    
    result = {
        'train': preprocessor.transform(train_X),
        'val_in': preprocessor.transform(val_in_X),
        'test_in': preprocessor.transform(test_in_X),
        'test_out': preprocessor.transform(test_out_X)
    }
    
    return result


def augment_with_covariates(X_preprocessed, train_Z, val_in_Z, test_in_Z, test_out_Z):
    """
    Append covariates to preprocessed features for each split.

    Parameters
    ----------
    X_preprocessed : dict
        Transformed features keyed by split name.
    train_Z : array-like
        Training covariates.
    val_in_Z : array-like or None
        Validation covariates, if available.
    test_in_Z : array-like
        In-sample test covariates.
    test_out_Z : array-like
        Out-of-sample test covariates.

    Returns
    -------
    dict
        Feature matrices augmented with covariates.
    """
    result = {
        'train': np.concatenate([X_preprocessed['train'], train_Z], axis=1),
        'val_in': np.concatenate([X_preprocessed['val_in'], val_in_Z], axis=1),
        'test_in': np.concatenate([X_preprocessed['test_in'], test_in_Z], axis=1),
        'test_out': np.concatenate([X_preprocessed['test_out'], test_out_Z], axis=1)
    }
    
    return result


def fit_and_predict(estimator, X_train, y_train, X_val_in, y_val_in, X_test_in, X_test_out, sample_weight=None):
    """
    Train the estimator and return probability predictions per split.

    Parameters
    ----------
    estimator : estimator
        Model supporting ``fit`` and ``predict_proba``.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    X_val_in : array-like or None
        Validation features, if available.
    y_val_in : array-like or None
        Validation labels, if available.
    X_test_in : array-like
        In-sample test features.
    X_test_out : array-like
        Out-of-sample test features.
    sample_weight : array-like, optional
        Sample weights for the training data.

    Returns
    -------
    dict
        Probability predictions keyed by split name.
    """
    # Check estimator type
    is_xgboost = isinstance(estimator, XGBClassifier)

    if is_xgboost:
        # XGBoost with eval_set for early stopping and monitoring
        fit_params = {}
        
        # Add sample weights if provided
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
        
        # Add eval_set if validation data is provided
        if X_val_in is not None and y_val_in is not None:
            fit_params['eval_set'] = [(X_train, y_train), (X_val_in, y_val_in)]
            fit_params['verbose'] = False
        
        estimator.fit(X_train, y_train, **fit_params)
    
    else:
        # Standard estimator
        if sample_weight is not None:
            estimator.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            estimator.fit(X_train, y_train)
    
    # Standard predictions
    predictions = {
        'train': estimator.predict_proba(X_train),
        'val_in': estimator.predict_proba(X_val_in),
        'test_in': estimator.predict_proba(X_test_in),
        'test_out': estimator.predict_proba(X_test_out)
    }
    
    return predictions


def run_pipeline_experiment(preprocessor, estimator, data_dict, seed, include_covariates=True):
    """
    Execute preprocessing, model fitting, validation, and evaluation.

    Parameters
    ----------
    preprocessor : transformer
        Preprocessing pipeline applied prior to modelling.
    estimator : estimator
        Model trained within the experiment.
    data_dict : dict
        Raw data splits and metadata.
    seed : int
        Random seed for reproducibility.
    include_covariates : bool, default=True
        Whether to append covariates to feature matrices.

    Returns
    -------
    dict
        Complete experiment results including predictions and metadata.
    """
    seed_everything(seed)
    
    # Check flags
    use_sample_weights = data_dict.get('use_sample_weights', True)
    
    # Unpack data into organized structure
    data = {
        'train': unpack_train_data(data_dict),
        'val_in': unpack_validation_data(data_dict),
        'test_in': unpack_test_in_data(data_dict),
        'test_out': unpack_test_out_data(data_dict)
    }
    metadata = unpack_metadata(data_dict)
    
    # Extract sample weights
    sample_weights = {
        'train': data_dict.get('sample_weights_train') if use_sample_weights else None,
        'val_in': data_dict.get('sample_weights_val_in') if use_sample_weights else None,
        'test_in': data_dict.get('sample_weights_test_in') if use_sample_weights else None,
        'test_out': data_dict.get('sample_weights_test_out') if use_sample_weights else None
    }

    # Extract groups
    groups = {
        'train': data_dict.get('groups_train'),
        'val_in': data_dict.get('groups_val_in'),
        'test_in': data_dict.get('groups_test_in'),
        'test_out': data_dict.get('groups_test_out')
    }
    
    # Preprocess all splits
    X_preprocessed = preprocess_all_splits(
        preprocessor,
        train_X=data['train']['X'],
        val_in_X=data['val_in']['X'],
        test_in_X=data['test_in']['X'],
        test_out_X=data['test_out']['X']
    )

    # Ensure downstream models always receive numpy arrays (prevents bad feature names on pandas objects)
    for split_name, split_matrix in X_preprocessed.items():
        if hasattr(split_matrix, "to_numpy"):
            X_preprocessed[split_name] = split_matrix.to_numpy()
        else:
            X_preprocessed[split_name] = np.asarray(split_matrix)
    
    # Optionally augment with covariates
    if include_covariates:
        X_final = augment_with_covariates(
            X_preprocessed,
            train_Z=data['train']['Z'],
            val_in_Z=data['val_in']['Z'],
            test_in_Z=data['test_in']['Z'],
            test_out_Z=data['test_out']['Z']
        )
    else:
        X_final = X_preprocessed
    
    # Log feature dimensionality for transparency/debugging
    glycan_feat_dim = X_preprocessed['train'].shape[1]
    covariate_dim = data['train']['Z'].shape[1] if include_covariates else 0
    total_feat_dim = X_final['train'].shape[1]
    feature_dims = {
        'glycan': glycan_feat_dim,
        'covariate': covariate_dim,
        'total': total_feat_dim,
    }

    # Fit estimator with sample weights and groups, generate predictions
    predictions = fit_and_predict(
        estimator,
        X_train=X_final['train'],
        y_train=data['train']['y'],
        X_val_in=X_final['val_in'],
        y_val_in=data['val_in']['y'],
        X_test_in=X_final['test_in'],
        X_test_out=X_final['test_out'],
        sample_weight=sample_weights['train']
    )
    
    # Collect test predictions
    predictions = {
        'train': predictions['train'],
        'val_in': predictions['val_in'],
        'test_in': predictions['test_in'],
        'test_out': predictions['test_out']
    }
    
    # Create subgroup names
    subgroup_names = create_demographic_subgroup_names(
        metadata['age_ranges'],
        metadata['cohort_locations']
    )
    
    # Perform subgroup analysis
    subgroup_metrics = create_subgroup_metrics_dict(
        y_train_true=data['train']['y'],
        y_train_pred=predictions['train'],
        Z_train_bin=data['train']['Z_bin'],
        V_train=data['train']['V'],
        y_val_in_true=data['val_in']['y'],
        y_val_in_pred=predictions['val_in'],
        Z_val_in_bin=data['val_in']['Z_bin'],
        V_val_in=data['val_in']['V'],
        y_test_in_true=data['test_in']['y'],
        y_test_in_pred=predictions['test_in'],
        Z_test_in_bin=data['test_in']['Z_bin'],
        V_test_in=data['test_in']['V'],
        y_test_out_true=data['test_out']['y'],
        y_test_out_pred=predictions['test_out'],
        Z_test_out_bin=data['test_out']['Z_bin'],
        V_test_out=data['test_out']['V'],
        age_group_names=subgroup_names['age'],
        sex_group_names=subgroup_names['sex'],
        location_names=subgroup_names['location'],
    )
    
    # Build and return complete results
    results = build_results_dict(
        estimator=estimator,
        preprocessor=preprocessor,
        data=data,
        predictions=predictions,
        metadata=metadata,
    )

    results['feature_dimensions'] = feature_dims
    results['groups'] = groups
    results['subgroup_metrics'] = subgroup_metrics
    results['use_sample_weights'] = use_sample_weights
    results['sample_weights'] = sample_weights
    
    return results


def make_preprocessors(processors, selected_keys=None, with_scaler=True):
    """
    Builds a dict of preprocessors, chaining each with:
      1. The provided processor,
      2. Optional RobustScaler.
    """

    selected = selected_keys
    preprocessors = {}
    for name in selected:
        steps = [
            (name.lower(), processors[name]),
        ]
        if with_scaler:
            steps.append(('scaler', RobustScaler()))
        prep = Pipeline(steps)
        preprocessors[name] = prep

    return preprocessors

def make_estimators(parameters, seed):
    """Instantiate estimators for the configured experiment suite.

    Parameters
    ----------
    parameters : dict
        Hyperparameters keyed by model identifier.
    seed : int
        Random seed forwarded to estimators.

    Returns
    -------
    dict
        Estimators keyed by identifier.
    """
    return {
        'LR': LogisticRegression(**parameters['LR'], random_state=seed),
        'XB': XGBClassifier(**parameters['XB'], random_state=seed),
    }