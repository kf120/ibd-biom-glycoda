# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ibd_biom_glycoda.evaluation.metrics import compute_scoring_metrics, is_lower_better
from ibd_biom_glycoda.evaluation.domain_generalization import parse_pipeline, _resolve_metric_axis_config, plot_between_test_cohort_metrics_from_summary

def create_subgroup_metrics_dict(
                y_train_true, y_train_pred, Z_train_bin, V_train, 
                y_val_in_true, y_val_in_pred, Z_val_in_bin, V_val_in, 
                y_test_in_true, y_test_in_pred, Z_test_in_bin, V_test_in, 
                y_test_out_true, y_test_out_pred, Z_test_out_bin, V_test_out, 
                age_group_names=None, sex_group_names=None, location_names=None):
    '''Create a dictionary of subgroup metrics for the validation and test sets,
    including performance metrics for the intersection of age and sex.'''
    
    subgroup_metrics = {
        'age': {
            'train': calculate_subgroup_performance(np.array(y_train_true), np.array(y_train_pred), 
                                                   Z_train_bin[:, 1], group_name='age', subgroup_names=age_group_names),
            'val_in': calculate_subgroup_performance(np.array(y_val_in_true), np.array(y_val_in_pred), 
                                                   Z_val_in_bin[:, 1], group_name='age', subgroup_names=age_group_names),
            'test_in': calculate_subgroup_performance(np.array(y_test_in_true), np.array(y_test_in_pred), 
                                                   Z_test_in_bin[:, 1], group_name='age', subgroup_names=age_group_names),
            'test_out': calculate_subgroup_performance(np.array(y_test_out_true), np.array(y_test_out_pred), 
                                                    Z_test_out_bin[:, 1], group_name='age', subgroup_names=age_group_names)
        },
        'sex': {
            'train': calculate_subgroup_performance(np.array(y_train_true), np.array(y_train_pred), 
                                                   Z_train_bin[:, 0], group_name='sex', subgroup_names=sex_group_names),
            'val_in': calculate_subgroup_performance(np.array(y_val_in_true), np.array(y_val_in_pred), 
                                                   Z_val_in_bin[:, 0], group_name='sex', subgroup_names=sex_group_names),
            'test_in': calculate_subgroup_performance(np.array(y_test_in_true), np.array(y_test_in_pred), 
                                                   Z_test_in_bin[:, 0], group_name='sex', subgroup_names=sex_group_names),
            'test_out': calculate_subgroup_performance(np.array(y_test_out_true), np.array(y_test_out_pred), 
                                                    Z_test_out_bin[:, 0], group_name='sex', subgroup_names=sex_group_names)
        },
        'location': {
            'train': calculate_subgroup_performance(np.array(y_train_true), np.array(y_train_pred), 
                                                   np.argmax(V_train, axis=1), group_name='location', subgroup_names=location_names),
            'val_in': calculate_subgroup_performance(np.array(y_val_in_true), np.array(y_val_in_pred), 
                                                   np.argmax(V_val_in, axis=1), group_name='location', subgroup_names=location_names),
            'test_in': calculate_subgroup_performance(np.array(y_test_in_true), np.array(y_test_in_pred), 
                                                   np.argmax(V_test_in, axis=1), group_name='location', subgroup_names=location_names),
            'test_out': calculate_subgroup_performance(np.array(y_test_out_true), np.array(y_test_out_pred), 
                                                    np.argmax(V_test_out, axis=1), group_name='location', subgroup_names=location_names)
        },
        'age_sex': {
            'train': calculate_intersection_performance(np.array(y_train_true), np.array(y_train_pred), 
                                                       Z_train_bin[:, 1], Z_train_bin[:, 0], 
                                                       age_subgroup_names=age_group_names,
                                                       sex_subgroup_names=sex_group_names),
            'val_in': calculate_intersection_performance(np.array(y_val_in_true), np.array(y_val_in_pred),
                                                         Z_val_in_bin[:, 1], Z_val_in_bin[:, 0], 
                                                         age_subgroup_names=age_group_names,
                                                         sex_subgroup_names=sex_group_names),
            'test_in': calculate_intersection_performance(np.array(y_test_in_true), np.array(y_test_in_pred), 
                                                      Z_test_in_bin[:, 1], Z_test_in_bin[:, 0], 
                                                      age_subgroup_names=age_group_names, 
                                                      sex_subgroup_names=sex_group_names),
            'test_out': calculate_intersection_performance(np.array(y_test_out_true), np.array(y_test_out_pred), 
                                                       Z_test_out_bin[:, 1], Z_test_out_bin[:, 0], 
                                                       age_subgroup_names=age_group_names, 
                                                       sex_subgroup_names=sex_group_names)
        }
    }
    return subgroup_metrics



def calculate_subgroup_performance(true_labels, pred_proba_labels, subgroup_labels, group_name=None, subgroup_names=None, metrics=['AUROC', 'ECE', 'LogLoss', 'Brier', 'Resolution Ratio', 'Reliability', 'Sensitivity', 'Specificity']):
    """
    Calculate and return metrics for each subgroup with interpretable subgroup names.
    
    Parameters:
    -----------
    true_labels : np.array
        The ground truth labels.
    pred_proba_labels : np.array
        The predicted probabilities or predictions.
    subgroup_labels : np.array
        The labels indicating the subgroup (e.g., age group, sex).
    group_name : str
        Name of the group being analyzed (e.g., 'age', 'sex', 'location').
    subgroup_names : dict or list, optional
        A dictionary or list mapping subgroup labels to human-readable names (e.g., {0: '<20', 1: '20-40'} for age).

    Returns:
    --------
    dict
        A dictionary with subgroup-wise performance metrics.
    """
    performance_dict = {}
    unique_groups = np.unique(subgroup_labels)
    
    for group in unique_groups:
        mask = subgroup_labels == group
        
        # Check if true_labels for this group contains more than one class
        if len(np.unique(true_labels[mask])) == 1:
            # Skip this subgroup if only one class is present
            continue
        
        # Compute the metrics for subgroups with more than one class
        subgroup_metrics = compute_scoring_metrics(true_labels[mask], pred_proba_labels[mask], metrics=metrics)
        
        # Ensure 'group' is converted to a scalar type, not numpy array
        group_value = group.item() if isinstance(group, np.ndarray) else group

        # Use subgroup_names if provided, otherwise use the group number
        group_name_label = subgroup_names.get(group_value, group_value) if subgroup_names else group_value
        
        performance_dict[group_name_label] = subgroup_metrics
    
    return performance_dict

def calculate_intersection_performance(true_labels, pred_proba_labels, age_labels, sex_labels,
                                       age_subgroup_names=None, sex_subgroup_names=None, metrics=['AUROC', 'ECE', 'LogLoss', 'Brier', 'Resolution Ratio', 'Reliability', 'Sensitivity', 'Specificity']):
    """
    Compute performance metrics for the intersection of age and sex subgroups.
    
    Returns a nested dictionary, e.g.:
    {
      age_group1_label: {
          sex_group1_label: {metric: value, ...},
          sex_group2_label: {metric: value, ...},
          ...
      },
      age_group2_label: {
          sex_group1_label: {metric: value, ...},
          sex_group2_label: {metric: value, ...},
          ...
      },
      ...
    }
    """
    performance_dict = {}
    unique_age = np.unique(age_labels)
    unique_sex = np.unique(sex_labels)
    
    for age in unique_age:
        # Convert age value to a scalar if needed.
        age_value = age.item() if isinstance(age, np.ndarray) else age
        # Use age subgroup names if provided.
        age_label = age_subgroup_names.get(age_value, age_value) if age_subgroup_names else age_value
        performance_dict[age_label] = {}
        
        for sex in unique_sex:
            sex_value = sex.item() if isinstance(sex, np.ndarray) else sex
            sex_label = sex_subgroup_names.get(sex_value, sex_value) if sex_subgroup_names else sex_value
            
            mask = (age_labels == age) & (sex_labels == sex)
            if len(np.unique(true_labels[mask])) < 2:
                # Skip if only one class is present.
                continue
                
            subgroup_metrics = compute_scoring_metrics(true_labels[mask], pred_proba_labels[mask], metrics=metrics)
            performance_dict[age_label][sex_label] = subgroup_metrics
    return performance_dict


def prepare_loco_subgroup_data(
    subgroup_analyses_loco,
    metric='AUROC',
    sets=('test_in', 'test_out')
):
    """
    Extract subgroup performance from LOCO nested structure.
    
    Converts nested dictionary:
        subgroup_analyses_loco[cohort][split]['age_sex'][age][sex][pipeline][metric]
    
    Into two DataFrames:
        1. Performance DataFrame: cohort, split, pipeline, age, sex, metric_mean, metric_sem
        2. Summary DataFrame: cohort, pipeline, overall_mean (aggregated across subgroups)
    
    Parameters
    ----------
    subgroup_analyses_loco : dict
        Nested dictionary with structure:
        {
            'cohort': {
                'test_in': {
                    'age_sex': {
                        '<40': {
                            'Male': {
                                'Pipeline': {
                                    'AUROC': {'Mean': 0.85, 'SEM': 0.02}
                                }
                            },
                            'Female': {...}
                        },
                        '>40': {...}
                    }
                },
                'test_out': {...}
            }
        }
    metric : str
        Metric to extract (e.g., 'AUROC', 'AUPRC')
    sets : tuple
        Splits to extract (e.g., ('test_in', 'test_out'))
    
    Returns
    -------
    detail_df : pd.DataFrame
        Per-subgroup performance with columns:
        ['Cohort', 'Split', 'Pipeline', 'Age', 'Sex', 'Mean', 'SEM']
    summary_df : pd.DataFrame
        Per-pipeline summary across subgroups:
        ['Cohort', 'Split', 'Pipeline', 'Overall_Mean', 'Overall_SEM', 
         'N_Subgroups', 'Worst_Subgroup', 'Worst_Performance']
    
    Examples
    --------
    >>> detail, summary = prepare_loco_subgroup_data(
    ...     subgroup_analyses_loco,
    ...     metric='AUROC',
    ...     sets=('test_in', 'test_out')
    ... )
    >>> print(detail.head())
    >>> print(summary.head())
    """
    records = []
    
    # Extract all cohorts
    cohorts = list(subgroup_analyses_loco.keys())
    
    for cohort in cohorts:
        for split in sets:
            if split not in subgroup_analyses_loco[cohort]:
                continue
            
            # Get age groups from the first cohort/split
            age_sex_data = subgroup_analyses_loco[cohort][split].get('age_sex', {})
            
            for age_group in age_sex_data.keys():
                for sex in age_sex_data[age_group].keys():
                    # Each pipeline
                    for pipeline, metrics in age_sex_data[age_group][sex].items():
                        if metric not in metrics:
                            continue
                        
                        perf_data = metrics[metric]
                        
                        records.append({
                            'Cohort': cohort,
                            'Split': split,
                            'Pipeline': pipeline,
                            'Age': age_group,
                            'Sex': sex,
                            'Mean': perf_data.get('Mean', np.nan),
                            'SEM': perf_data.get('SEM', np.nan)
                        })
    
    detail_df = pd.DataFrame(records)
    
    # Create subgroup label
    detail_df['Subgroup'] = detail_df['Age'].astype(str) + ' | ' + detail_df['Sex'].astype(str)
    
    # Compute summary per pipeline
    summary_records = []
    
    for (cohort, split, pipeline), grp in detail_df.groupby(['Cohort', 'Split', 'Pipeline']):
        performances = grp['Mean'].values
        sems = grp['SEM'].values
        subgroups = grp['Subgroup'].values
        
        # Overall mean (simple average across subgroups)
        overall_mean = np.nanmean(performances)
        
        # Propagate uncertainty (sum of variances)
        overall_sem = np.sqrt(np.nansum(sems**2)) / len(sems)
        
        # Worst subgroup
        if len(performances) > 0 and not np.all(np.isnan(performances)):
            worst_idx = np.nanargmin(performances)
            worst_subgroup = subgroups[worst_idx]
            worst_performance = performances[worst_idx]
        else:
            worst_subgroup = 'N/A'
            worst_performance = np.nan
        
        summary_records.append({
            'Cohort': cohort,
            'Split': split,
            'Pipeline': pipeline,
            'Overall_Mean': overall_mean,
            'Overall_SEM': overall_sem,
            'N_Subgroups': len(performances),
            'Worst_Subgroup': worst_subgroup,
            'Worst_Performance': worst_performance
        })
    
    summary_df = pd.DataFrame(summary_records)
    
    return detail_df, summary_df

def prepare_generalization_data_all_subgroups(
    df_metric1_detail,
    df_metric2_detail,
    group_by='Preprocessor',
    filter_value=None,
    subgroups=None,
    metric_names=('metric1', 'metric2'),
    sets=('test_in', 'test_out')
):
    """
    Convert LOCO subgroup performance DataFrames into format for plot_generalization_gaps_subgroups.
    This function collects data for ALL subgroups at once.
    
    Parameters
    ----------
    df_auroc_detail : pd.DataFrame
        AUROC detail data from prepare_loco_subgroup_data
        Columns: ['Cohort', 'Split', 'Pipeline', 'Age', 'Sex', 'Mean', 'SEM', 'Subgroup']
    df_ece_detail : pd.DataFrame
        ECE detail data from prepare_loco_subgroup_data
    group_by : str, default='Preprocessor'
        Column to group by: 'Preprocessor' or 'Estimator'
    filter_value : str, optional
        Value to filter by (e.g., 'LR' when group_by='Preprocessor')
    subgroups : list of str, optional
        Specific subgroups to analyze (e.g., ['<40_Male', '<40_Female'])
        If None, uses all available subgroups
    metric_names : tuple/list of str, default=('metric1', 'metric2')
        Human-readable names for the two metrics being compared, used to label
        output keys (e.g., ('AUROC', 'ECE')).
    sets : sequence of str, default ('test_in', 'test_out')
        Exactly two evaluation splits to extract (e.g., ('val_in', 'test_in')).
        The first entry acts as the reference split and the second as the
        comparison split when computing gaps downstream.
    
    Returns
    -------
    dict
        Nested dictionary: {method: {subgroup: {cohort: {metric_split: value}}}}
        Keys follow the pattern ``{metric}_{set}``, where ``set`` originates
        from the provided ``sets`` argument.
    str
        Mode for plotting: 'preprocessing' or 'model'
    
    Examples
    --------
    >>> # Get detail dataframes
    >>> df_auroc_detail, _ = prepare_loco_subgroup_data(
    ...     subgroup_analyses_loco, metric='AUROC'
    ... )
    >>> df_ece_detail, _ = prepare_loco_subgroup_data(
    ...     subgroup_analyses_loco, metric='ECE'
    ... )
    >>> 
    >>> # Prepare data for all subgroups, LR model, compare preprocessing
    >>> results_dict, mode = prepare_generalization_data_all_subgroups(
    ...     df_auroc_detail, df_ece_detail,
    ...     group_by='Preprocessor',
    ...     filter_value='LR',
    ...     subgroups=['<40_Male', '<40_Female', '40-60_Male', '40-60_Female'],
    ...     metric_names=('AUROC', 'ECE')
    ... )
    >>> 
    >>> # Then plot
    >>> fig, ax = plot_generalization_gaps_subgroups(
    ...     results_dict, mode='preprocessing', as_fold_change=True
    ... )
    """
    if not isinstance(sets, (list, tuple)) or len(sets) != 2:
        raise ValueError("sets must be a sequence with exactly two entries.")

    set_names = tuple(str(s) for s in sets)
    if any(not name for name in set_names):
        raise ValueError("sets entries must be non-empty strings.")
    
    # Add Preprocessor and Estimator columns if not present
    if 'Preprocessor' not in df_metric1_detail.columns:
        df_metric1_detail[['Preprocessor', 'Estimator']] = df_metric1_detail['Pipeline'].apply(
            lambda s: pd.Series(parse_pipeline(s))
        )
    if 'Preprocessor' not in df_metric2_detail.columns:
        df_metric2_detail[['Preprocessor', 'Estimator']] = df_metric2_detail['Pipeline'].apply(
            lambda s: pd.Series(parse_pipeline(s))
        )
    
    mode = 'preprocessing' if group_by == 'Preprocessor' else 'model'
    
    # Get all available subgroups if not specified
    if subgroups is None:
        subgroups = df_metric1_detail['Subgroup'].unique().tolist()
    
    if metric_names is None or len(metric_names) != 2:
        raise ValueError("metric_names must be an iterable with exactly two entries.")

    metric1_name, metric2_name = metric_names

    # Merge AUROC and ECE data
    df_merged = df_metric1_detail.merge(
        df_metric2_detail,
        on=['Cohort', 'Split', 'Pipeline', 'Age', 'Sex', 'Subgroup', 'Preprocessor', 'Estimator'],
        suffixes=('_metric1', '_metric2')
    )
    
    # Rename Mean columns to match expected format
    df_merged = df_merged.rename(columns={
        'Mean_metric1': f'{metric1_name} Mean',
        'Mean_metric2': f'{metric2_name} Mean'
    })
    
    # Apply filter if specified
    if filter_value is not None:
        filter_col = 'Estimator' if group_by == 'Preprocessor' else 'Preprocessor'
        df_merged = df_merged[df_merged[filter_col] == filter_value]
    
    # Filter to specified subgroups
    df_merged = df_merged[df_merged['Subgroup'].isin(subgroups)]
    
    # Build nested dictionary: {method: {subgroup: {cohort: {metrics}}}}
    results_dict = {}
    
    # Group by the specified column (method/preprocessing)
    for method in df_merged[group_by].unique():
        df_method = df_merged[df_merged[group_by] == method]
        results_dict[method] = {}
        
        # Group by subgroup
        for subgroup in subgroups:
            df_subgroup = df_method[df_method['Subgroup'] == subgroup]
            
            if len(df_subgroup) == 0:
                continue
                
            results_dict[method][subgroup] = {}
            
            # Group by cohort
            for cohort in df_subgroup['Cohort'].unique():
                df_cohort = df_subgroup[df_subgroup['Cohort'] == cohort]
                
                set_rows = {}
                missing_set = False
                for set_name in set_names:
                    df_set = df_cohort[df_cohort['Split'] == set_name]
                    if df_set.empty:
                        missing_set = True
                        break
                    set_rows[set_name] = df_set.iloc[0]

                if missing_set:
                    continue

                entry = {}
                for metric_name in (metric1_name, metric2_name):
                    mean_col = f'{metric_name} Mean'
                    for set_name in set_names:
                        entry[f'{metric_name}_{set_name}'] = set_rows[set_name][mean_col]

                results_dict[method][subgroup][cohort] = entry
    
    return results_dict, mode

def run_full_generalization_gap_analysis_subgroups(
    subgroup_analyses_loco_test_cohorts,
    metrics,
    model_keys,
    sets,
    group_by="Preprocessor",
    subgroups=None,
):
    """
    Run the full generalization gap analysis for all subgroups and multiple models.

    Parameters
    ----------
    subgroup_analyses_loco_test_cohorts : Any
        Subgroup LOCO performance results as expected by `prepare_loco_subgroup_data`.
    metrics : list of str
        List of primary metrics to analyze, e.g. metrics = ['Discrimination', 'Calibration'].
        The first two entries (metrics[0], metrics[1]) are used.
    model_keys : list of str
        Model identifiers to filter on (e.g. ['LR', 'XB']). Each entry is passed as
        `filter_value` to `prepare_generalization_data_all_subgroups`.
    generalization_sets : iterable of str
        Sets passed to `prepare_loco_subgroup_data` and `prepare_generalization_data_all_subgroups`,
        e.g. ('test_in', 'test_out').
    group_by : str, default 'Preprocessor'
        Column used to group models in `prepare_generalization_data_all_subgroups`.
    subgroups : iterable of str or None, default None
        If provided, these subgroups will be used in `prepare_generalization_data_all_subgroups`.
        If None, all available subgroups are used (internally handled by that function).

    Returns
    -------
    results_dicts_subgroup : dict
        Dict mapping model_key -> subgroup generalization results dict,
        as returned by `prepare_generalization_data_all_subgroups`.
    mode : Any
        The `mode` object returned by `prepare_generalization_data_all_subgroups`
        (same for all models, so the last one is returned).
    df_metric1_detail : Any
        Detailed subgroup DataFrame for metrics[0], as returned by `prepare_loco_subgroup_data`.
    df_metric2_detail : Any
        Detailed subgroup DataFrame for metrics[1], as returned by `prepare_loco_subgroup_data`.
    subgroups_available : Any
        Array/sequence of unique subgroups from `df_metric1_detail['Subgroup'].unique()`.
    """

    # Prepare detailed subgroup data for both metrics
    df_metric1_detail, _ = prepare_loco_subgroup_data(
        subgroup_analyses_loco_test_cohorts,
        metric=metrics[0],
        sets=sets,
    )

    df_metric2_detail, _ = prepare_loco_subgroup_data(
        subgroup_analyses_loco_test_cohorts,
        metric=metrics[1],
        sets=sets,
    )

    # Get unique subgroups (for information/inspection)
    subgroups_available = df_metric1_detail["Subgroup"].unique()
    subgroups_to_use = subgroups if subgroups is not None else None

    # Run subgroup generalization analysis per model
    results_dicts_subgroup = {}
    mode = None

    for model_key in model_keys:
        results_dict_subgroup, mode = prepare_generalization_data_all_subgroups(
            df_metric1_detail,
            df_metric2_detail,
            group_by=group_by,
            filter_value=model_key,
            subgroups=subgroups_to_use,
            metric_names=(metrics[0], metrics[1]),
            sets=sets,
        )
        results_dicts_subgroup[model_key] = results_dict_subgroup

    return results_dicts_subgroup, mode, df_metric1_detail, df_metric2_detail, subgroups_available


def plot_generalization_gaps_subgroups(
    results_dict,
    mode='preprocessing',
    as_fold_change=False,
    subgroup_palette=None,
    figsize=(10, 7),
    ax=None,
    axis_config=None,
    sets=('test_in', 'test_out')
):
    """
    Plot subgroup-level generalization gaps between two metrics (e.g., AUROC vs ECE).
    Each point represents the average gap across cohorts for a specific method/model and subgroup.

    Parameters
    ----------
    results_dict : dict
        Nested dictionary with structure:
        {
            'method_or_model': {
                'subgroup_name': {
                    'cohort_name': {
                        '<metric>_<set_a>': float,
                        '<metric>_<set_b>': float
                    }
                }
            }
        }
        Exactly two unique metrics with suffixes matching the provided ``sets``
        argument (e.g., ``_test_in``/``_test_out``) are required.
    name : str
        Name for the plot title
    mode : str, default='preprocessing'
        Either 'preprocessing' to compare preprocessing methods,
        or 'model' to compare models
    as_fold_change : bool, default=False
        If True, compute gaps as fold changes; if False, compute as differences
    subgroup_palette : sequence or dict, optional
        Colors for subgroups. Provide a sequence aligned with subgroup order or
        a dict mapping subgroup name to color. If None, uses a tab10 colormap.
    figsize : tuple, default=(10, 7)
        Figure size. Only used if ax is None
    ax : matplotlib.axes.Axes, optional
        If provided, plot into this axes. If None, create new figure and axes
    axis_config : mapping or callable, optional
        Axis overrides keyed by metric name, matching ``plot_generalization_gaps``.
        Accepts ``xlim``, ``ylim``, ``xticks``, ``yticks``, ``xticklabels`` and
        ``yticklabels`` either directly or nested under ``'primary'`` / ``'secondary'``.
    sets : sequence of str, default ('test_in', 'test_out')
        Exactly two evaluation splits to compare. The first entry is treated as
        the reference split, the second as the comparison split when computing
        gaps.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        Figure object if ax was None, otherwise None
    ax : matplotlib.axes.Axes
        The axes containing the plot
    """
    if mode not in ['preprocessing', 'model']:
        raise ValueError("mode must be either 'preprocessing' or 'model'")

    if not isinstance(sets, (list, tuple)) or len(sets) != 2:
        raise ValueError("sets must be a sequence with exactly two entries.")

    set_names = tuple(str(s) for s in sets)
    if any(not name for name in set_names):
        raise ValueError("sets entries must be non-empty strings.")
    
    # Markers for different methods/models
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Get methods/models and subgroups
    methods = list(results_dict.keys())
    n_methods = len(methods)
    
    if n_methods > len(marker_styles):
        raise ValueError(f"Too many methods/models ({n_methods}). Maximum supported: {len(marker_styles)}")

    if n_methods == 0:
        raise ValueError("results_dict is empty; nothing to plot.")
    
    # Get all subgroups (from first method)
    subgroups = list(results_dict[methods[0]].keys())
    n_subgroups = len(subgroups)

    if n_subgroups == 0:
        raise ValueError("results_dict does not contain any subgroups to visualize.")

    # Ensure at least one cohort entry exists
    first_cohort_keys = list(results_dict[methods[0]][subgroups[0]].keys())
    if len(first_cohort_keys) == 0:
        raise ValueError("results_dict does not contain any cohort entries to visualize.")

    # Determine metric names from the first cohort entry
    sample_results = results_dict[methods[0]][subgroups[0]][first_cohort_keys[0]]
    metric_order = []
    for key in sample_results.keys():
        for set_name in set_names:
            suffix = f'_{set_name}'
            if key.endswith(suffix):
                metric_order.append(key[: -len(suffix)])
                break

    metric_names = list(dict.fromkeys(metric_order))
    if len(metric_names) != 2:
        raise ValueError(
            "plot_generalization_gaps_subgroups expects exactly two metrics per cohort with suffixes matching the provided sets."
        )

    for metric in metric_names:
        missing = [set_name for set_name in set_names if f'{metric}_{set_name}' not in sample_results]
        if missing:
            raise ValueError(
                "Each metric entry must contain values for both requested sets. "
                f"Metric '{metric}' is missing: {', '.join(missing)}"
            )

    metric_x, metric_y = metric_names

    def _compute_gap(reference, comparison, low_is_better, fold_change):
        if reference is None or comparison is None:
            return np.nan
        reference = float(reference)
        comparison = float(comparison)
        if np.isnan(reference) or np.isnan(comparison):
            return np.nan
        if fold_change:
            if reference == 0:
                return np.nan
            ratio = comparison / reference
            return (ratio - 1.0) if low_is_better else 1.0 - ratio
        return (comparison - reference) if low_is_better else (reference - comparison)
    
    # Color scheme for subgroups
    if subgroup_palette is None:
        subgroup_colors = list(plt.cm.get_cmap('tab10', n_subgroups).colors)
    elif isinstance(subgroup_palette, dict):
        missing = [subgroup for subgroup in subgroups if subgroup not in subgroup_palette]
        if missing:
            raise ValueError(
                "subgroup_palette dict is missing colors for subgroups: " + ", ".join(missing)
            )
        subgroup_colors = [subgroup_palette[subgroup] for subgroup in subgroups]
    else:
        if len(subgroup_palette) < n_subgroups:
            raise ValueError(
                f"subgroup_palette must provide at least {n_subgroups} colors; got {len(subgroup_palette)}"
            )
        subgroup_colors = list(subgroup_palette[:n_subgroups])

    subgroup_color_map = dict(zip(subgroups, subgroup_colors))
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    
    # Store all gaps for axis limits
    all_x_gaps = []
    all_y_gaps = []

    x_lower_better = is_lower_better(metric_x)
    y_lower_better = is_lower_better(metric_y)
    
    # Process each method/model
    for method_idx, method in enumerate(methods):
        marker = marker_styles[method_idx]
        
        # Process each subgroup
        for subgroup_idx, subgroup in enumerate(subgroups):
            cohort_data = results_dict[method][subgroup]
            
            # Compute gaps for each cohort, then average
            x_gaps = []
            y_gaps = []

            for cohort_name, results in cohort_data.items():
                try:
                    ref_x = results[f'{metric_x}_{set_names[0]}']
                    comp_x = results[f'{metric_x}_{set_names[1]}']
                    ref_y = results[f'{metric_y}_{set_names[0]}']
                    comp_y = results[f'{metric_y}_{set_names[1]}']
                except KeyError as exc:
                    raise KeyError(
                        "Results dictionary missing expected metric/set combination. "
                        f"Available keys: {list(results.keys())}"
                    ) from exc

                gap_x = _compute_gap(ref_x, comp_x, x_lower_better, as_fold_change)
                gap_y = _compute_gap(ref_y, comp_y, y_lower_better, as_fold_change)

                x_gaps.append(gap_x)
                y_gaps.append(gap_y)
            
            # Compute average across cohorts for this method+subgroup
            avg_x_gap = np.nanmean(x_gaps)
            avg_y_gap = np.nanmean(y_gaps)

            all_x_gaps.append(avg_x_gap)
            all_y_gaps.append(avg_y_gap)

            if np.isnan(avg_x_gap) or np.isnan(avg_y_gap):
                continue

            # Plot subgroup average point
            ax.scatter(avg_x_gap, avg_y_gap,
                      s=60, alpha=0.85,
                      color=subgroup_color_map[subgroup],
                      marker=marker,
                      edgecolor='black', linewidth=1.5,
                      zorder=2)
    
    # Determine axis limits with padding
    all_x_gaps = np.array(all_x_gaps, dtype=float)
    all_y_gaps = np.array(all_y_gaps, dtype=float)

    valid_x = all_x_gaps[~np.isnan(all_x_gaps)]
    valid_y = all_y_gaps[~np.isnan(all_y_gaps)]

    if valid_x.size == 0 or valid_y.size == 0:
        raise ValueError("Gaps contain only NaN values; cannot determine axis limits.")

    max_x = max(valid_x.max(), 0) * 1.2 + 0.01
    max_y = max(valid_y.max(), 0) * 1.2 + 0.01
    min_x = min(valid_x.min(), 0) * 1.2 - 0.01
    min_y = min(valid_y.min(), 0) * 1.2 - 0.01
    
    # Add reference lines
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    
    # Styling
    if as_fold_change:
        ax.set_xlabel(
            f"{metric_x} Gap (fold change {set_names[1]}/{set_names[0]})", fontweight='bold'
        )
        ax.set_ylabel(
            f"{metric_y} Gap (fold change {set_names[1]}/{set_names[0]})", fontweight='bold'
        )
    else:
        ax.set_xlabel(
            f"{metric_x} Gap", fontweight='bold'
        )
        ax.set_ylabel(
            f"{metric_y} Gap", fontweight='bold'
        )
    
    # Set axis limits
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_y, max_y])

    axis_primary_override = _resolve_metric_axis_config(metric_x, 'primary', axis_config)
    if axis_primary_override is not None:
        if 'xlim' in axis_primary_override and axis_primary_override['xlim'] is not None:
            ax.set_xlim(axis_primary_override['xlim'])
        if 'xticks' in axis_primary_override and axis_primary_override['xticks'] is not None:
            ax.set_xticks(list(axis_primary_override['xticks']))
        if 'xticklabels' in axis_primary_override and axis_primary_override['xticklabels'] is not None:
            ax.set_xticklabels(list(axis_primary_override['xticklabels']))

    axis_secondary_override = _resolve_metric_axis_config(metric_y, 'secondary', axis_config)
    if axis_secondary_override is not None:
        if 'ylim' in axis_secondary_override and axis_secondary_override['ylim'] is not None:
            ax.set_ylim(axis_secondary_override['ylim'])
        if 'yticks' in axis_secondary_override and axis_secondary_override['yticks'] is not None:
            ax.set_yticks(list(axis_secondary_override['yticks']))
        if 'yticklabels' in axis_secondary_override and axis_secondary_override['yticklabels'] is not None:
            ax.set_yticklabels(list(axis_secondary_override['yticklabels']))
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    
    # Create legend organized in two rows: [Subgroups] on first row, [Methods] on second row
    legend_elements_subgroups = []
    legend_elements_methods = []
    
    # First row: Add subgroup colors
    for subgroup_idx, subgroup in enumerate(subgroups):
        legend_elements_subgroups.append(
            Patch(facecolor=subgroup_color_map[subgroup], edgecolor='black',
                  label=subgroup)
        )
    
    # Second row: Add methods with their markers
    for method_idx, method in enumerate(methods):
        legend_elements_methods.append(
            Line2D([0], [0], marker=marker_styles[method_idx], color='w',
                   markerfacecolor='gray', markersize=8,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=method)
        )
    
    # Combine legend elements
    legend_elements = legend_elements_subgroups + legend_elements_methods
    
    # Create legend with proper row organization
    # ncol should be max of n_subgroups and n_methods for proper alignment
    ncol = max(n_subgroups, n_methods)
    ax.legend(handles=legend_elements, 
             loc='upper left', 
             frameon=True, 
             shadow=True,
             fontsize=9,
             ncol=ncol,
             columnspacing=1.0,
             handletextpad=0.5)
    
    # Only call tight_layout if we created the figure
    if fig is not None:
        plt.tight_layout()
    
    return fig, ax

def plot_between_test_cohort_metrics_subgroup(
    meta_summary,
    model_names,
    subgroup_var,
    metrics=("AUROC", "ECE"),
    set_name="test_out",
    groups=None,
    preprocessor_names=None,
    palette="okabe_ito",
    metric_labels=None,
    figsize=(7.0, 5.5),
    bar_width=0.28,
    show_sem=True,
    hatch="///",
    include_legend=True,
    ax=None,
    axis_config=None,
    preprocessor_parser=None,
):
    """Visualize subgroup meta-analysis results for one or two models.

    Parameters
    ----------
    meta_summary : dict
        Output from ``summarize_loco_results_all``.
    model_name : str or sequence of str
        One or two model identifiers to display. When multiple models are
        provided, the plot overlays their subgroup performance using
        different colors.
    subgroup_var : str
        Subgroup variable key within ``meta_summary['subgroups'][set_name]``.
    metrics : sequence of str, default ('AUROC', 'ECE')
        Two metrics to plot; left axis uses the first metric.
    set_name : str, default 'test_out'
        Evaluation split to visualize.
    groups : sequence of str, optional
        Sequence containing the subgroup label to visualise. If not provided,
        defaults to the first available subgroup. Only a single subgroup is
        supported per plot.
    preprocessor_names : sequence of str, optional
        Desired ordering of preprocessing methods. Defaults to the observed
        order within the subgroup data.
    palette : str or mapping, default 'okabe_ito'
        Palette name or explicit color mapping for model identifiers.
    metric_labels : sequence of str, optional
        Display labels for metrics; falls back to ``metrics``.
    figsize : tuple, default (7.0, 5.5)
        Figure size when creating axes.
    bar_width : float, default 0.28
        Width of each metric bar per subgroup.
    show_sem : bool, default True
        Whether to draw SEM-based error bars when available.
    hatch : str, default '///'
        Hatch pattern for the secondary metric bars.
    include_legend : bool, default True
        Display legend describing models and metrics.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot into.
    axis_config : mapping or callable, optional
        Optional axis overrides forwarded to ``plot_between_test_cohort_metrics_from_summary``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure.
    ax : matplotlib.axes.Axes
        Primary axis for the first metric.
    ax_secondary : matplotlib.axes.Axes
        Twin axis for the second metric.
    """

    if isinstance(model_names, str):
        model_names = (model_names,)
    elif isinstance(model_names, (list, tuple)):
        model_names = tuple(
            dict.fromkeys(
                [name for name in model_names if isinstance(name, str) and name.strip()]
            )
        )
    else:
        raise TypeError("model_names must be a string or sequence of strings.")
    if not model_names:
        raise ValueError("model_name must provide at least one non-empty identifier.")
    if len(model_names) > 2:
        raise ValueError("model_name can contain at most two entries for comparison.")

    if len(metrics) != 2:
        raise ValueError("metrics must contain exactly two entries.")
    if metric_labels is not None and len(metric_labels) != 2:
        raise ValueError("metric_labels must contain exactly two entries when provided.")

    metric_primary, metric_secondary = metrics
    label_primary, label_secondary = (
        metric_labels if metric_labels is not None else metrics
    )

    subgroup_summary = meta_summary.get('subgroups')
    if subgroup_summary is None:
        raise ValueError("meta_summary does not contain a 'subgroups' section.")
    if set_name not in subgroup_summary:
        available_sets = ", ".join(sorted(subgroup_summary.keys()))
        raise ValueError(
            f"set_name='{set_name}' not found. Available sets: {available_sets}."
        )

    set_groups = subgroup_summary[set_name]
    if subgroup_var not in set_groups:
        available_vars = ", ".join(sorted(set_groups.keys()))
        raise ValueError(
            f"subgroup_var='{subgroup_var}' not found. Available subgroup vars: {available_vars}."
        )

    subgroup_dict = set_groups[subgroup_var]
    if not isinstance(subgroup_dict, dict) or not subgroup_dict:
        raise ValueError(
            f"No subgroup data found for '{subgroup_var}' in set '{set_name}'."
        )

    available_groups = [grp for grp, df in subgroup_dict.items() if isinstance(df, pd.DataFrame) and not df.empty]
    if not available_groups:
        raise ValueError(
            f"No subgroup tables available for '{subgroup_var}' in set '{set_name}'."
        )

    if groups is None or len(groups) == 0:
        target_group = available_groups[0]
    else:
        if len(groups) != 1:
            raise ValueError("Only a single subgroup can be visualised per plot.")
        target_group = groups[0]
        if target_group not in available_groups:
            raise ValueError(
                f"Subgroup '{target_group}' not available for '{subgroup_var}' in set '{set_name}'."
            )

    df_group = subgroup_dict[target_group]
    if not isinstance(df_group, pd.DataFrame) or df_group.empty:
        raise ValueError(
            f"No data available for subgroup '{target_group}' in set '{set_name}'."
        )

    df_group = df_group.copy()
    df_group['Model'] = df_group['Model'].astype(str)
    df_group['Metric'] = df_group['Metric'].astype(str)

    if preprocessor_parser is None:
        def _preprocessor_parser(model_label, _model_key):
            preprocessor, _ = parse_pipeline(model_label)
            return preprocessor
    else:
        _preprocessor_parser = preprocessor_parser

    summary_rows = []

    # Match the parsed estimator exactly; substring matching on the raw label
    # would make 'LR' match 'XB+CLR'.
    estimator_series = df_group['Model'].apply(lambda s: parse_pipeline(s)[1])

    for model_key in model_names:
        mask = estimator_series == model_key
        df_model = df_group.loc[mask].copy()
        if df_model.empty:
            raise ValueError(
                f"Model '{model_key}' not found in subgroup '{target_group}' for set '{set_name}'."
            )

        df_model = df_model[df_model['Metric'].isin(metrics)]
        if df_model.empty:
            raise ValueError(
                f"Requested metrics {metrics} not available for model '{model_key}' in subgroup '{target_group}'."
            )

        df_model['Preprocessor'] = df_model.apply(
            lambda row: _preprocessor_parser(row['Model'], model_key), axis=1
        )
        df_model['Preprocessor'] = df_model['Preprocessor'].replace('', 'Unknown').fillna('Unknown')

        for (preproc, metric_name), grp_metric in df_model.groupby(['Preprocessor', 'Metric'], dropna=False):
            row = grp_metric.iloc[0]
            summary_rows.append({
                'ModelKey': model_key,
                'Preprocessor': preproc,
                'Metric': metric_name,
                'Pooled Mean': row.get('Pooled Mean', np.nan),
                'Pooled SEM': row.get('Pooled SEM', np.nan),
                'Lower CI': row.get('Lower CI', np.nan),
                'Upper CI': row.get('Upper CI', np.nan)
            })

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise ValueError("Unable to build summary table for subgroup plotting.")

    if preprocessor_names is not None:
        preprocessor_order = tuple(
            dict.fromkeys(
                [proc for proc in preprocessor_names if isinstance(proc, str) and proc.strip()]
            )
        )
    else:
        observed_preprocessors = summary_df['Preprocessor'].tolist()
        preprocessor_order = tuple(dict.fromkeys([proc for proc in observed_preprocessors if proc]))

    if not preprocessor_order:
        raise ValueError("No preprocessors detected for the selected subgroup and metrics.")

    fig, ax_primary, ax_secondary = plot_between_test_cohort_metrics_from_summary(
        summary_df=summary_df,
        model_names=model_names,
        preprocessor_names=preprocessor_order,
        metrics=metrics,
        palette=palette,
        metric_labels=(label_primary, label_secondary),
        figsize=figsize,
        bar_width=bar_width,
        show_sem=show_sem,
        hatch=hatch,
        include_legend=include_legend,
        ax=ax,
        axis_config=axis_config,
    )

    created_new_axes = ax is None
    ax_primary.set_xlabel('Feature Representation', fontweight='bold')

    # Add title
    title_text = f"{target_group}"
    if created_new_axes:
        fig.suptitle(title_text, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    else:
        ax_primary.set_title(title_text, fontweight='bold')

    return fig, ax_primary, ax_secondary

def generate_strata_summary(df, disease_col='DISEASE', age_col='Age', sex_col='Sex', 
                            cohort_col='Cohort', test_cohort=None, age_ranges=None, plot=True):
    """
    Create stacked bar plots of age-sex strata by cohort and disease.
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    disease_col : str
        Name of disease column
    age_col : str
        Name of age column
    sex_col : str
        Name of sex column (should be binary: M/F)
    cohort_col : str
        Name of cohort column
    test_cohort : str or None
        If None: show all cohorts separately
        If provided: pool all cohorts except test cohort, show side-by-side
    age_ranges : list of tuples or None
        Age ranges as [(min1, max1, 'label1'), (min2, max2, 'label2'), ...]
        If None: use pd.qcut to create 3 equal groups
    
    Returns:
    --------
    tuple : (counts_table, ratio_table)
        counts_table: Absolute counts of strata
        ratio_table: DataFrame with proportions for each domain-disease-strata combination
    """
    
    df_plot = df.copy()
    
    # Create age categories
    if age_ranges is None:
        df_plot['Age_Category'] = pd.qcut(df_plot[age_col], q=3, 
                                           labels=['Young', 'Middle', 'Old'],
                                           duplicates='drop')
    else:
        bins = [r[0] for r in age_ranges] + [age_ranges[-1][1]]
        labels = [r[2] for r in age_ranges]
        df_plot['Age_Category'] = pd.cut(df_plot[age_col], bins=bins, labels=labels)
    
    # Create age-sex strata
    df_plot['Strata'] = df_plot['Age_Category'].astype(str) + '_' + df_plot[sex_col]
    
    # Handle domain pooling
    if test_cohort is not None:
        df_plot['Domain'] = df_plot[cohort_col].apply(
            lambda x: test_cohort if x == test_cohort else 'Pooled'
        )
        group_col = 'Domain'
    else:
        df_plot['Domain'] = df_plot[cohort_col]
        group_col = 'Domain'
    
    # Get counts
    counts = df_plot.groupby([group_col, disease_col, 'Strata']).size().unstack(fill_value=0)
    
    # Calculate ratios (proportions within each domain-disease combination)
    ratios = np.round(100 * counts.div(counts.sum(axis=1), axis=0), 1)
    
    # Create readable tables
    ratio_table = ratios.reset_index()
    ratio_table.columns.name = None
    
    counts_table = counts.reset_index()
    counts_table.columns.name = None
    
    # Plot setup
    domains_list = df_plot[group_col].unique()
    diseases = df_plot[disease_col].unique()
    strata = counts.columns
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    n_diseases = len(diseases)
    bar_width = 0.8 / n_diseases
    x = np.arange(len(domains_list))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strata)))
    
    # Plot stacked bars with proportion labels
    for i, disease in enumerate(diseases):
        bottoms = np.zeros(len(domains_list))
        
        for j, stratum in enumerate(strata):
            heights = []
            proportions = []
            
            for domain in domains_list:
                try:
                    heights.append(counts.loc[(domain, disease), stratum])
                    proportions.append(ratios.loc[(domain, disease), stratum])
                except KeyError:
                    heights.append(0)
                    proportions.append(0)
            
            bars = ax.bar(x + i * bar_width, heights, bar_width, bottom=bottoms,
                         label=stratum if i == 0 else "", color=colors[j])
            
            # Add proportion labels on bars
            for k, (bar, height, prop) in enumerate(zip(bars, heights, proportions)):
                if height > 0 and prop >= 5:  # Only show if proportion >= 5%
                    label_y = bottoms[k] + height / 2
                    ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                           f'{prop:.0f}%', ha='center', va='center',
                           fontsize=8, fontweight='bold', color='white')
            
            bottoms += np.array(heights)
    
    # Customize
    ax.set_xlabel(group_col, fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Age-Sex Strata Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(x + bar_width * (n_diseases - 1) / 2)
    ax.set_xticklabels(domains_list)
    ax.legend(title='Age-Sex Strata', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add disease labels
    for i, disease in enumerate(diseases):
        for j, domain_x in enumerate(x):
            ax.text(domain_x + i * bar_width, -ax.get_ylim()[1] * 0.05, 
                   disease, ha='center', va='top', rotation=90, fontsize=8)
    
    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print("\n=== COUNTS ===")
    print(counts_table.to_string(index=False))
    print("\n=== PROPORTIONS ===")
    print(ratio_table.to_string(index=False))
    
    return counts_table, ratio_table