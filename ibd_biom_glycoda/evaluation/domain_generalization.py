# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from ibd_biom_glycoda.evaluation.metrics import is_lower_better
from ibd_biom_glycoda.plotting.style import create_model_colors


def parse_pipeline(model_label):
    """Parse a pipeline string into (preprocessor, estimator) components."""
    if model_label is None or (isinstance(model_label, float) and pd.isna(model_label)):
        return "Unknown", "Unknown"

    label = str(model_label).strip()
    if not label:
        return "Unknown", "Unknown"

    def _clean(value):
        if value is None:
            return "Unknown"
        value = str(value).strip()
        return value if value else "Unknown"

    if '+' in label:
        est, prep = label.split('+', 1)
        return _clean(prep), _clean(est)
    if '|' in label:
        prep, est = label.split('|', 1)
        return _clean(prep), _clean(est)
    if '__' in label:
        prep, est = label.split('__', 1)
        return _clean(prep), _clean(est)

    tokens = label.split()
    if len(tokens) >= 2:
        est = tokens[-1]
        prep = ' '.join(tokens[:-1])
    elif tokens:
        est = tokens[0]
        prep = 'Unknown'
    else:
        return "Unknown", "Unknown"

    return _clean(prep), _clean(est)

def prepare_loco_performance_data(
    results_test_cohorts,
    sets=('test_in', 'test_out'),
    metric='AUROC',
    parse_model_name=None,
    estimators_order=None,
    preprocessor_order=None
):
    """
    Prepare LOCO performance data for plotting.
    
    Parameters
    ----------
    results_test_cohorts : dict
        Results dictionary with structure:
        {cohort: {set_name: {'disease': DataFrame}}}
    sets : tuple
        Test set names to include (e.g., ('test_in', 'test_out'))
    metric : str
        Metric name (e.g., 'AUROC', 'AUPRC')
    parse_model_name : callable, optional
        Function to parse model name into (preprocessor, estimator).
        If None, uses default parser.
    estimators_order : list, optional
        Order of estimators. If provided, filters and orders data.
    preprocessor_order : list, optional
        Order of preprocessors. If provided, filters and orders data.
    
    Returns
    -------
    pd.DataFrame
        Prepared data with columns:
        - LOCO: cohort name
        - Set: test set name
        - Model: full model name
        - Preprocessor: preprocessor name
        - Estimator: estimator name
        - {metric} Mean: mean metric value
        - SEM: standard error
    dict
        Metadata with keys:
        - metric: metric name
        - mean_col: column name for mean
        - sem_col: column name for SEM
        - estimators: list of unique estimators
        - preprocessors: list of unique preprocessors
        - cohorts: list of unique cohorts
        - sets: list of test sets
    
    Examples
    --------
    >>> df, meta = prepare_loco_performance_data(
    ...     results_test_cohorts,
    ...     sets=('test_in', 'test_out'),
    ...     metric='AUROC',
    ...     estimators_order=['LR', 'RF', 'XGB'],
    ...     preprocessor_order=['Raw', 'CLR', 'ILR']
    ... )
    """
    parser = parse_model_name or parse_pipeline
    
    # Melt results into DataFrame
    rows = []
    mean_col = f"{metric} Mean"
    sem_col = f"{metric} SEM"
    
    for cohort, res in results_test_cohorts.items():
        for set_name in sets:
            df_part = res[set_name]['disease']
            for _, r in df_part.iterrows():
                rows.append({
                    'LOCO': cohort,
                    'Set': set_name,
                    'Model': r['Model'],
                    mean_col: r[mean_col],
                    'SEM': r[sem_col]
                })
    
    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        raise ValueError("No data found—check dict structure or metric name.")
    
    # Parse model names
    df_plot[['Preprocessor', 'Estimator']] = df_plot['Model'].apply(
        lambda s: pd.Series(parser(s))
    )
    
    # Filter and order estimators
    if estimators_order is not None:
        df_plot = df_plot[df_plot['Estimator'].isin(estimators_order)]
        df_plot['Estimator'] = pd.Categorical(
            df_plot['Estimator'], 
            categories=estimators_order, 
            ordered=True
        )
    
    # Filter and order preprocessors
    if preprocessor_order is not None:
        df_plot = df_plot[df_plot['Preprocessor'].isin(preprocessor_order)]
        df_plot['Preprocessor'] = pd.Categorical(
            df_plot['Preprocessor'], 
            categories=preprocessor_order, 
            ordered=True
        )
    
    # Create metadata
    metadata = {
        'metric': metric,
        'mean_col': mean_col,
        'sem_col': sem_col,
        'estimators': (df_plot['Estimator'].cat.categories.tolist() 
                      if hasattr(df_plot['Estimator'], 'cat')
                      else df_plot['Estimator'].unique().tolist()),
        'preprocessors': (df_plot['Preprocessor'].cat.categories.tolist()
                         if hasattr(df_plot['Preprocessor'], 'cat')
                         else df_plot['Preprocessor'].unique().tolist()),
        'cohorts': df_plot['LOCO'].unique().tolist(),
        'sets': list(sets)
    }
    
    return df_plot, metadata

def prepare_generalization_data_overall(
    metric_dfs,
    group_by='Preprocessor',
    filter_value=None,
    sets=('test_in', 'test_out'),
):
    """
    Convert LOCO performance DataFrames into format for plot_generalization_gaps.
    
    Parameters
    ----------
    metric_dfs : dict
        Dictionary of metric DataFrames: {metric_name: df}
        Each df should be from prepare_loco_performance_data with columns:
        ['LOCO', 'Set', 'Model', 'Preprocessor', 'Estimator', '{Metric} Mean']
        Example: {'auroc': df_auroc, 'brier': df_brier, 'sensitivity': df_sens, 'specificity': df_spec}
    group_by : str, default='Preprocessor'
        Column to group by: 'Preprocessor' or 'Estimator'
        - 'Preprocessor': Compare preprocessing methods (for a given model)
        - 'Estimator': Compare models (for a given preprocessing)
    filter_value : str, optional
        Value to filter by. If group_by='Preprocessor', filters Estimator column.
        If group_by='Estimator', filters Preprocessor column.
    
    sets : sequence of str, default=('test_in', 'test_out')
        Exactly two evaluation splits. The first entry acts as the reference
        split and the second as the comparison split when computing gaps.

    Returns
    -------
    dict
        Nested dictionary: {method: {cohort: {metric_set: value}}}
        Keys like 'auroc_<set_a>', 'brier_<set_b>', 'sensitivity_<set_a>', etc.
    str
        Mode for plotting: 'preprocessing' or 'model'
    
    Examples
    --------
    >>> # Compare preprocessing methods for Logistic Regression
    >>> metric_dfs = {
    ...     'auroc': df_auroc,
    ...     'brier': df_brier,
    ...     'sensitivity': df_sens,
    ...     'specificity': df_spec
    ... }
    >>> data, mode = prepare_generalization_data(
    ...     metric_dfs,
    ...     group_by='Preprocessor',
    ...     filter_value='LR'
    ... )
    
    >>> # Compare models for Raw preprocessing
    >>> data, mode = prepare_generalization_data(
    ...     metric_dfs,
    ...     group_by='Estimator',
    ...     filter_value='Raw'
    ... )
    """
    mode = 'preprocessing' if group_by == 'Preprocessor' else 'model'
    
    def _find_mean_column(df, metric):
        """Locate the mean column for a metric, allowing flexible capitalization."""
        metric_norm = metric.replace(' ', '').lower()
        for col in df.columns:
            col_norm = col.replace(' ', '').lower()
            if 'mean' in col_norm and metric_norm in col_norm:
                return col
        # As a fallback, accept exact "{metric} Mean" case-sensitive
        fallback = f'{metric} Mean'
        if fallback in df.columns:
            return fallback
        raise ValueError(f"Could not find mean column for metric '{metric}'")

    if len(sets) != 2:
        raise ValueError(f"Exactly 2 sets must be specified, got {len(sets)}")

    set_a, set_b = sets

    # Start with the first metric as base
    metric_names = list(metric_dfs.keys())
    if len(metric_names) == 0:
        raise ValueError("metric_dfs is empty; nothing to prepare.")

    first_metric = metric_names[0]
    df_merged = metric_dfs[first_metric].copy()
    mean_col = _find_mean_column(df_merged, first_metric)
    df_merged = df_merged.rename(columns={mean_col: f'{first_metric}_mean'})

    # Merge all other metrics
    for metric_name in metric_names[1:]:
        df_metric = metric_dfs[metric_name].copy()
        mean_col = _find_mean_column(df_metric, metric_name)
        df_metric = df_metric.rename(columns={mean_col: f'{metric_name}_mean'})

        df_merged = df_merged.merge(
            df_metric[['LOCO', 'Set', 'Model', 'Preprocessor', 'Estimator', f'{metric_name}_mean']],
            on=['LOCO', 'Set', 'Model', 'Preprocessor', 'Estimator'],
            how='inner'
        )
    
    # Apply filter if specified
    if filter_value is not None:
        filter_col = 'Estimator' if group_by == 'Preprocessor' else 'Preprocessor'
        df_merged = df_merged[df_merged[filter_col] == filter_value]
    
    results_dict = {}
    
    # Group by the specified column
    unique_methods = df_merged[group_by].unique()
    if len(unique_methods) == 0:
        raise ValueError("No entries found after filtering; check group_by/filter_value inputs.")

    for method in unique_methods:
        df_method = df_merged[df_merged[group_by] == method]
        results_dict[method] = {}
        
        for cohort in df_method['LOCO'].unique():
            df_cohort = df_method[df_method['LOCO'] == cohort]
            
            # Extract set_a and set_b values
            set_a_rows = df_cohort[df_cohort['Set'] == set_a]
            set_b_rows = df_cohort[df_cohort['Set'] == set_b]

            if set_a_rows.empty or set_b_rows.empty:
                raise ValueError(
                    f"Missing '{set_a}'/'{set_b}' entries for cohort '{cohort}' and method '{method}'."
                )

            set_a_row = set_a_rows.iloc[0]
            set_b_row = set_b_rows.iloc[0]
            
            # Build result dictionary for this cohort
            cohort_results = {}
            for metric_name in metric_names:
                cohort_results[f'{metric_name}_{set_a}'] = set_a_row[f'{metric_name}_mean']
                cohort_results[f'{metric_name}_{set_b}'] = set_b_row[f'{metric_name}_mean']
            
            results_dict[method][cohort] = cohort_results
    
    return results_dict, mode

def extract_metrics_for_plotting(results_dict, metrics=['auroc', 'logloss'], sets=('test_in', 'test_out')):
    """
    Extract specific metrics from general results dictionary for plotting.
    
    Parameters
    ----------
    results_dict : dict
        Nested dictionary from prepare_generalization_data with structure:
        {
            'method': {
                'cohort': {
                    'auroc_<set_a>': float,
                    'auroc_<set_b>': float,
                    'logloss_<set_a>': float,
                    'logloss_<set_b>': float,
                    'sensitivity_<set_a>': float,
                    ... (other metrics)
                }
            }
        }
    metrics : list of str, default=['auroc', 'logloss']
        List of exactly 2 metrics to extract for x and y axes
    sets : sequence of str, default=('test_in', 'test_out')
        Exactly two evaluation splits. The first entry acts as the reference
        split and the second as the comparison split when computing gaps.
        
    Returns
    -------
    dict
        Filtered dictionary with only the specified metrics:
        {
            'method': {
                'cohort': {
                    'metric1_<set_a>': float,
                    'metric1_<set_b>': float,
                    'metric2_<set_a>': float,
                    'metric2_<set_b>': float
                }
            }
        }
    
    Examples
    --------
    >>> # Extract AUROC and ECE for standard plot
    >>> filtered = extract_metrics_for_plotting(results_dict, metrics=['auroc', 'brier'])
    >>> fig, ax = plot_generalization_gaps(filtered, name='LR', mode='preprocessing')
    
    >>> # Extract sensitivity and specificity
    >>> filtered = extract_metrics_for_plotting(results_dict, metrics=['sensitivity', 'specificity'])
    >>> fig, ax = plot_generalization_gaps(filtered, name='LR', mode='preprocessing')
    """
    if len(metrics) != 2:
        raise ValueError(f"Exactly 2 metrics must be specified, got {len(metrics)}")
    if len(sets) != 2:
        raise ValueError(f"Exactly 2 sets must be specified, got {len(sets)}")

    set_a, set_b = sets
    
    filtered_dict = {}
    
    for method, cohort_data in results_dict.items():
        filtered_dict[method] = {}
        
        for cohort, metric_values in cohort_data.items():
            filtered_dict[method][cohort] = {}
            
            # Extract only the specified metrics
            for metric in metrics:
                test_in_key = f'{metric}_{set_a}'
                test_out_key = f'{metric}_{set_b}'
                
                if test_in_key not in metric_values or test_out_key not in metric_values:
                    raise KeyError(f"Metric '{metric}' not found in results. Available keys: {list(metric_values.keys())}")
                
                filtered_dict[method][cohort][test_in_key] = metric_values[test_in_key]
                filtered_dict[method][cohort][test_out_key] = metric_values[test_out_key]
    
    return filtered_dict

def run_full_generalization_gap_analysis_overall(
    results_loco_test_cohorts,
    metrics,
    model_keys,
    sets = ('test_in', 'test_out'),
    group_by = "Preprocessor",
):
    """
    Run the full generalization gap analysis pipeline for multiple models.

    Parameters
    ----------
    results_loco_test_cohorts : Any
        Nested LOCO performance results structure as expected by `prepare_loco_performance_data`.
    metrics : list of str
        List of primary metrics to analyze, e.g. METRICS = ['Discrimination', 'Calibration'].
        The first two entries will be used for the main performance metrics.
    model_keys : list of str
        Model identifiers to filter on (e.g. ['LR', 'XB']). Each entry is passed as
        `filter_value` to `prepare_generalization_data_overall`.
    sets : sequence of str, default ('test_in', 'test_out')
        Evaluation splits to compare. Note that 'val_in' is XGBoost's early-stopping
        `eval_set` but a clean holdout for LR, so a 'val_in' gap is not the same
        quantity across models and should not be compared between them directly.
    group_by : str, default 'Preprocessor'
        Column used to group models in `prepare_generalization_data_overall`.

    Returns
    -------
    metrics_dicts : dict
        Nested dict mapping model_key -> metric_name -> plotting-ready structures,
        as returned by `extract_metrics_for_plotting`.
    results_dicts : dict
        Dict mapping model_key -> raw generalization results dict,
        as returned by `prepare_generalization_data_overall`.
    mode : Any
        The `mode` object returned by `prepare_generalization_data_overall`
        (same for all models, so the last one is returned).
    metrics_dfs : dict
        Dict with keys {metrics[0], metrics[1], 'Sensitivity', 'Specificity'} and
        values the corresponding DataFrames produced by `prepare_loco_performance_data`.
    """

    # Prepare data for each metric
    df_discr, _ = prepare_loco_performance_data(
        results_test_cohorts=results_loco_test_cohorts,
        sets=sets,
        metric=metrics[0],
    )

    df_calib, _ = prepare_loco_performance_data(
        results_test_cohorts=results_loco_test_cohorts,
        sets=sets,
        metric=metrics[1],
    )

    df_sens, _ = prepare_loco_performance_data(
        results_test_cohorts=results_loco_test_cohorts,
        sets=sets,
        metric="Sensitivity",
    )

    df_spec, _ = prepare_loco_performance_data(
        results_test_cohorts=results_loco_test_cohorts,
        sets=sets,
        metric="Specificity",
    )

    metrics_dfs = {
        metrics[0]: df_discr,
        metrics[1]: df_calib,
        "Sensitivity": df_sens,
        "Specificity": df_spec,
    }

    results_dicts = {}
    metrics_dicts = {}
    mode = None

    for model_key in model_keys:
        results_dict, mode = prepare_generalization_data_overall(
            metric_dfs=metrics_dfs,
            group_by=group_by,
            filter_value=model_key,
            sets=sets,
        )

        results_dicts[model_key] = results_dict
        metrics_dicts[model_key] = extract_metrics_for_plotting(
            results_dict,
            metrics=metrics,
            sets=sets,
        )

    return metrics_dicts, results_dicts, mode, metrics_dfs

def plot_generalization_gaps_overall(
    results_dict,
    name='Model',
    mode='preprocessing',
    as_fold_change=False,
    cohort_palette=None,
    figsize=(7, 5),
    ax=None,
    sets=('test_in', 'test_out'),
    axis_config=None,
):
    """
    Create a 2D scatter plot showing AUROC gap vs LogLoss gap across preprocessing methods or models.
    Points with the same marker represent the same preprocessing method (or model).
    Points closer to origin (0,0) indicate better generalization.
    Clustering of markers indicates similar generalization across methods/models.
    
    Parameters
    ----------
    results_dict : dict
        Nested dictionary with structure:
        {
            'method_or_model': {
                'cohort_name': {
                    'auroc_<set_a>': float,
                    'auroc_<set_b>': float,
                    'logloss_<set_a>': float,
                    'logloss_<set_b>': float
                }
            }
        }
    name : str
        Name of the model (when mode='preprocessing') or preprocessing method (when mode='model')
    mode : str, default='preprocessing'
        Either 'preprocessing' to compare preprocessing methods for a given model,
        or 'model' to compare models for a given preprocessing method
    as_fold_change : bool, default=False
        If True, compute gaps as fold changes; if False, compute as differences
    cohort_palette : sequence or dict, optional
        Colors for cohorts. Provide a sequence aligned with cohort order or
        a dict mapping cohort name to color. If None, uses a viridis colormap.
    figsize : tuple, default=(7, 5)
        Figure size. Only used if ax is None
    ax : matplotlib.axes.Axes, optional
        If provided, plot into this axes. If None, create new figure and axes
    sets : sequence of str, default=('test_in', 'test_out')
        Exactly two evaluation splits. The first entry acts as the reference
        split and the second as the comparison split when computing gaps.
    axis_config : mapping or callable, optional
        Optional overrides for axis limits/ticks per metric. Accepts the same
        structure used by ``plot_between_test_cohort_metrics_overall``.
        
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
    
    # Markers for different methods/models
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Get methods/models
    methods = list(results_dict.keys())
    n_methods = len(methods)
    
    if n_methods > len(marker_styles):
        raise ValueError(f"Too many methods/models ({n_methods}). Maximum supported: {len(marker_styles)}")
    
    if n_methods == 0:
        raise ValueError("results_dict is empty; nothing to plot.")

    # Get all cohorts (assuming all methods have same cohorts)
    cohorts = list(results_dict[methods[0]].keys())
    n_cohorts = len(cohorts)
    
    # Color scheme for cohorts
    if cohort_palette is None:
        cohort_colors = list(plt.cm.get_cmap('viridis', n_cohorts).colors)
    elif isinstance(cohort_palette, dict):
        missing = [cohort for cohort in cohorts if cohort not in cohort_palette]
        if missing:
            raise ValueError(
                "cohort_palette dict is missing colors for cohorts: " + ", ".join(missing)
            )
        cohort_colors = [cohort_palette[cohort] for cohort in cohorts]
    else:
        if len(cohort_palette) < n_cohorts:
            raise ValueError(
                f"cohort_palette must provide at least {n_cohorts} colors; got {len(cohort_palette)}"
            )
        cohort_colors = list(cohort_palette[:n_cohorts])

    cohort_color_map = dict(zip(cohorts, cohort_colors))
    
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    
    if n_cohorts == 0:
        raise ValueError("results_dict does not contain any cohorts to visualize.")

    # Determine metric names from the first cohort entry
    sample_results = results_dict[methods[0]][cohorts[0]]
    metric_prefixes = []
    for key in sample_results.keys():
        for set_name in set_names:
            suffix = f'_{set_name}'
            if key.endswith(suffix):
                metric_prefixes.append(key[: -len(suffix)])
                break
    metric_names = sorted(set(metric_prefixes), key=lambda k: metric_prefixes.index(k))
    if len(metric_names) != 2:
        raise ValueError(
            "plot_generalization_gaps expects exactly two metrics per cohort "
            "and matching values for the provided sets."
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

    # Store all gaps for axis limits
    all_x_gaps = []
    all_y_gaps = []

    # Store gaps by method for computing averages
    method_gaps = {method: {'x': [], 'y': []} for method in methods}

    x_lower_better = is_lower_better(metric_x)
    y_lower_better = is_lower_better(metric_y)
    
    # Process each method/model
    for method_idx, method in enumerate(methods):
        marker = marker_styles[method_idx]
        
        # Process each cohort
        for cohort_idx, cohort in enumerate(cohorts):
            results = results_dict[method][cohort]

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

            all_x_gaps.append(gap_x)
            all_y_gaps.append(gap_y)

            # Store for average calculation
            method_gaps[method]['x'].append(gap_x)
            method_gaps[method]['y'].append(gap_y)
            
            # Plot cohort point
            ax.scatter(gap_x, gap_y,
                      s=60, alpha=0.7, 
                      color=cohort_color_map[cohort],
                      marker=marker,
                      edgecolor='black', linewidth=1.5,
                      zorder=1)
    
    # Plot average gaps for each method in black
    for method_idx, method in enumerate(methods):
        marker = marker_styles[method_idx]
        avg_x_gap = np.nanmean(method_gaps[method]['x'])
        avg_y_gap = np.nanmean(method_gaps[method]['y'])
        
        # Plot average point in black
        ax.scatter(avg_x_gap, avg_y_gap,
                  s=100, alpha=0.9,
                  color='black',
                  marker=marker,
                  edgecolor='white', linewidth=2,
                  zorder=3)
    
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
    
    # Create legend organized as: [Cohorts + Average] on first row, [Methods] on second row
    legend_elements = []
    
    # First row: Add cohort colors
    from matplotlib.patches import Patch
    for cohort_idx, cohort in enumerate(cohorts):
        legend_elements.append(
            Patch(facecolor=cohort_color_map[cohort], edgecolor='black',
                  label=cohort, linewidth=1.5)
        )
    
    # Add average indicator to first row (as rectangle like cohorts)
    legend_elements.append(
        Patch(facecolor='black', edgecolor='white', linewidth=1.5,
            label='Average')
    )
    
    # Second row: Add methods with their markers
    for method_idx, method in enumerate(methods):
        legend_elements.append(
            Line2D([0], [0], marker=marker_styles[method_idx], color='w',
                   markerfacecolor='gray', markersize=8,
                   markeredgecolor='black', markeredgewidth=1.5,
                   label=method)
        )
    
    # Create horizontal legend with items organized in rows
    n_cohorts_with_avg = n_cohorts + 1  # cohorts + average
    ax.legend(handles=legend_elements, 
             loc='upper left', 
             frameon=True, 
             shadow=True,
             fontsize=9,
             ncol=n_cohorts_with_avg,  # First row: all cohorts + average
             columnspacing=1.0,
             handletextpad=0.5)
    
    # Only call tight_layout if we created the figure
    if fig is not None:
        plt.tight_layout()
    
    return fig, ax


def _resolve_metric_axis_config(
    metric_name,
    axis_role,
    axis_config=None,
):
    """Return axis overrides for the specified metric/axis role."""

    if axis_config is None:
        return None

    if callable(axis_config):
        payload = axis_config(metric_name, axis_role)
    else:
        metric_entry = axis_config.get(metric_name) if hasattr(axis_config, 'get') else None
        if metric_entry is None:
            return None
        if isinstance(metric_entry, dict) and axis_role in metric_entry:
            payload = metric_entry.get(axis_role)
        else:
            payload = metric_entry

    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise TypeError("Axis config must resolve to a mapping of axis properties.")

    allowed = {'ylim', 'yticks', 'yticklabels', 'xlim', 'xticks', 'xticklabels'}
    unexpected = set(payload.keys()).difference(allowed)
    if unexpected:
        keys_display = ", ".join(sorted(unexpected))
        raise ValueError(f"Unsupported axis config keys: {keys_display}.")

    return payload


def _plot_dual_metric_bars(
    summary_df,
    *,
    model_names,
    preprocessors,
    metric_primary,
    metric_secondary,
    label_primary,
    label_secondary,
    bar_width,
    show_sem,
    hatch,
    include_legend,
    colors,
    figsize,
    ax=None,
    axis_config=None,
):
    if not model_names:
        raise ValueError("model_names must contain at least one entry.")
    if not preprocessors:
        raise ValueError("preprocessors must contain at least one entry.")

    preprocessor_list = list(preprocessors)
    summary_unique = summary_df.drop_duplicates(['ModelKey', 'Preprocessor', 'Metric']).copy()
    summary_index = summary_unique.set_index(['ModelKey', 'Preprocessor', 'Metric'], drop=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax_secondary = ax.twinx()

    positions = np.arange(len(preprocessor_list), dtype=float)
    n_models = len(model_names)
    inner_width = bar_width / max(n_models, 1)

    model_primary_values = {model: [] for model in model_names}
    model_secondary_values = {model: [] for model in model_names}
    model_primary_lower = {model: [] for model in model_names}
    model_primary_upper = {model: [] for model in model_names}
    model_secondary_lower = {model: [] for model in model_names}
    model_secondary_upper = {model: [] for model in model_names}

    for proc in preprocessor_list:
        for model_key in model_names:
            try:
                primary_row = summary_index.loc[(model_key, proc, metric_primary)]
            except KeyError as exc:
                raise ValueError(
                    f"Metric '{metric_primary}' missing for model '{model_key}' and preprocessor '{proc}'."
                ) from exc
            if isinstance(primary_row, pd.DataFrame):
                primary_row = primary_row.iloc[0]

            try:
                secondary_row = summary_index.loc[(model_key, proc, metric_secondary)]
            except KeyError as exc:
                raise ValueError(
                    f"Metric '{metric_secondary}' missing for model '{model_key}' and preprocessor '{proc}'."
                ) from exc
            if isinstance(secondary_row, pd.DataFrame):
                secondary_row = secondary_row.iloc[0]

            primary_val = primary_row.get('Pooled Mean', np.nan)
            primary_val = float(primary_val) if not pd.isna(primary_val) else np.nan
            secondary_val = secondary_row.get('Pooled Mean', np.nan)
            secondary_val = float(secondary_val) if not pd.isna(secondary_val) else np.nan

            model_primary_values[model_key].append(primary_val)
            model_secondary_values[model_key].append(secondary_val)

            if show_sem:
                primary_sem = primary_row.get('Pooled SEM', np.nan)
                secondary_sem = secondary_row.get('Pooled SEM', np.nan)
                primary_err = 0.0 if pd.isna(primary_sem) else max(float(primary_sem), 0.0)
                secondary_err = 0.0 if pd.isna(secondary_sem) else max(float(secondary_sem), 0.0)
            else:
                primary_err = 0.0
                secondary_err = 0.0

            model_primary_lower[model_key].append(primary_err)
            model_primary_upper[model_key].append(primary_err)
            model_secondary_lower[model_key].append(secondary_err)
            model_secondary_upper[model_key].append(secondary_err)

    def _prepare_yerr(lower, upper):
        if not show_sem:
            return None
        has_error = any(err > 0 for err in lower + upper)
        if not has_error:
            return None
        return np.array([lower, upper])

    for idx, model_key in enumerate(model_names):
        color = colors[model_key]
        center_offset = (idx - (n_models - 1) / 2.0) * inner_width

        primary_positions = positions - (bar_width / 2.0) + center_offset
        secondary_positions = positions + (bar_width / 2.0) + center_offset

        primary_yerr = _prepare_yerr(model_primary_lower[model_key], model_primary_upper[model_key])
        secondary_yerr = _prepare_yerr(model_secondary_lower[model_key], model_secondary_upper[model_key])

        ax.bar(
            primary_positions,
            model_primary_values[model_key],
            width=inner_width * 0.9,
            color=color,
            alpha=0.95,
            edgecolor='white',
            linewidth=1.2,
            zorder=3,
            yerr=primary_yerr,
            error_kw={'capsize': 4, 'linewidth': 1.0, 'alpha': 0.8}
        )

        ax_secondary.bar(
            secondary_positions,
            model_secondary_values[model_key],
            width=inner_width * 0.9,
            color=color,
            alpha=0.6,
            edgecolor='white',
            linewidth=1.2,
            hatch=hatch,
            zorder=3,
            yerr=secondary_yerr,
            error_kw={'capsize': 4, 'linewidth': 1.0, 'alpha': 0.8}
        )

    ax.set_xlabel('Feature Representation', fontweight='bold')
    ax.set_ylabel(label_primary, fontweight='bold')
    ax_secondary.set_ylabel(label_secondary, fontweight='bold')
    ax.set_xticks(positions)
    ax.set_xticklabels(preprocessor_list, fontweight='bold')

    def _compute_limits(values, lower, upper):
        arr = np.array([v for v in values if not np.isnan(v)], dtype=float)
        if arr.size == 0:
            return None
        min_val = arr.min()
        max_val = arr.max()
        if show_sem and (any(lower) or any(upper)):
            candidates_min = [min_val]
            candidates_min.extend(v - l for v, l in zip(values, lower) if not np.isnan(v))
            candidates_max = [max_val]
            candidates_max.extend(v + u for v, u in zip(values, upper) if not np.isnan(v))
            min_val = min(candidates_min)
            max_val = max(candidates_max)
        span = max_val - min_val
        padding = max(span * 0.1, 0.01)
        return (min_val - padding, max_val + padding)

    primary_values_flat = [val for model_key in model_names for val in model_primary_values[model_key]]
    primary_lower_flat = [err for model_key in model_names for err in model_primary_lower[model_key]]
    primary_upper_flat = [err for model_key in model_names for err in model_primary_upper[model_key]]
    secondary_values_flat = [val for model_key in model_names for val in model_secondary_values[model_key]]
    secondary_lower_flat = [err for model_key in model_names for err in model_secondary_lower[model_key]]
    secondary_upper_flat = [err for model_key in model_names for err in model_secondary_upper[model_key]]

    primary_limits = _compute_limits(primary_values_flat, primary_lower_flat, primary_upper_flat)
    secondary_limits = _compute_limits(secondary_values_flat, secondary_lower_flat, secondary_upper_flat)

    if primary_limits is not None:
        ax.set_ylim(primary_limits)
    if secondary_limits is not None:
        ax_secondary.set_ylim(secondary_limits)

    primary_override = _resolve_metric_axis_config(metric_primary, 'primary', axis_config)
    secondary_override = _resolve_metric_axis_config(metric_secondary, 'secondary', axis_config)

    if primary_override is not None:
        if 'ylim' in primary_override and primary_override['ylim'] is not None:
            ax.set_ylim(primary_override['ylim'])
        if 'yticks' in primary_override and primary_override['yticks'] is not None:
            ax.set_yticks(list(primary_override['yticks']))
        if 'yticklabels' in primary_override and primary_override['yticklabels'] is not None:
            ax.set_yticklabels(list(primary_override['yticklabels']))

    if secondary_override is not None:
        if 'ylim' in secondary_override and secondary_override['ylim'] is not None:
            ax_secondary.set_ylim(secondary_override['ylim'])
        if 'yticks' in secondary_override and secondary_override['yticks'] is not None:
            ax_secondary.set_yticks(list(secondary_override['yticks']))
        if 'yticklabels' in secondary_override and secondary_override['yticklabels'] is not None:
            ax_secondary.set_yticklabels(list(secondary_override['yticklabels']))

    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5, zorder=0)

    if include_legend:
        from matplotlib.patches import Patch

        model_handles = [
            Patch(facecolor=colors[model], edgecolor='white', linewidth=1.3, label=model)
            for model in model_names
        ]
        metric_handles = [
            Patch(facecolor='gray', alpha=0.9, edgecolor='white', linewidth=1.3, label=f"{label_primary}"),
            Patch(facecolor='gray', alpha=0.65, edgecolor='white', linewidth=1.3, hatch=hatch, label=f"{label_secondary}")
        ]
        handles = model_handles + metric_handles
        ncol = min(max(len(handles), 1), 4)
        ax.legend(
            handles=handles,
            loc='upper left',
            framealpha=0.95,
            fontsize=9,
            title_fontsize=10,
            ncol=ncol,
            columnspacing=1.0,
            handletextpad=0.5
        )

    fig.tight_layout()
    return fig, ax, ax_secondary


def plot_between_test_cohort_metrics_from_summary(
    summary_df,
    model_names,
    preprocessor_names=None,
    metrics=("AUROC", "Brier"),
    palette="okabe_ito",
    metric_labels=None,
    figsize=(7.0, 5.5),
    bar_width=0.28,
    show_sem=True,
    hatch="///",
    include_legend=True,
    ax=None,
    axis_config=None,
):
    """Plot pooled metrics for the given summary table.

    Parameters
    ----------
    axis_config : mapping or callable, optional
        Optional overrides for axis limits or ticks keyed by metric name. If a
        callable is supplied it receives ``(metric_name, axis_role)`` and
        should return a dictionary with ``ylim``/``yticks``/``yticklabels``
        entries.
    """

    if summary_df is None or summary_df.empty:
        raise ValueError("summary_df must contain pooled metric data.")

    required_cols = {'ModelKey', 'Preprocessor', 'Metric', 'Pooled Mean'}
    missing_cols = required_cols.difference(summary_df.columns)
    if missing_cols:
        missing_str = ", ".join(sorted(missing_cols))
        raise ValueError(f"summary_df is missing required columns: {missing_str}.")

    if isinstance(model_names, str):
        candidate_models = (model_names,)
    else:
        candidate_models = tuple(
            dict.fromkeys(
                [name for name in model_names if isinstance(name, str) and name.strip()]
            )
        )

    if not candidate_models:
        raise ValueError("model_names must provide at least one non-empty identifier.")

    if len(metrics) != 2:
        raise ValueError("metrics must contain exactly two entries.")
    if metric_labels is not None and len(metric_labels) != 2:
        raise ValueError("metric_labels must contain exactly two entries when provided.")

    summary_df = summary_df.copy()
    summary_df['ModelKey'] = summary_df['ModelKey'].astype(str)
    summary_df['Preprocessor'] = summary_df['Preprocessor'].astype(str)
    summary_df['Metric'] = summary_df['Metric'].astype(str)

    summary_df = summary_df[summary_df['ModelKey'].isin(candidate_models)]
    if summary_df.empty:
        raise ValueError(f"Requested models {candidate_models} not found in summary dataframe.")

    summary_df = summary_df[summary_df['Metric'].isin(metrics)]
    if summary_df.empty:
        raise ValueError(f"Requested metrics {metrics} not available for models {candidate_models}.")

    observed_preprocessors = [proc for proc in summary_df['Preprocessor'].tolist() if proc]
    observed_preprocessors = list(dict.fromkeys(observed_preprocessors))

    if preprocessor_names is None:
        target_preprocessors = observed_preprocessors
    else:
        preprocessor_candidates = [
            proc for proc in preprocessor_names if isinstance(proc, str) and proc.strip()
        ]
        target_preprocessors = list(dict.fromkeys(preprocessor_candidates))
        missing_preprocessors = [proc for proc in target_preprocessors if proc not in observed_preprocessors]
        if missing_preprocessors:
            missing_str = ", ".join(missing_preprocessors)
            raise ValueError(
                f"Preprocessors {missing_str} not present for requested models {candidate_models}."
            )

    if not target_preprocessors:
        raise ValueError("No preprocessors detected for the selected model and metrics.")

    metric_primary, metric_secondary = metrics
    label_primary, label_secondary = metric_labels if metric_labels is not None else metrics

    colors = create_model_colors(candidate_models, palette=palette)

    return _plot_dual_metric_bars(
        summary_df,
        model_names=candidate_models,
        preprocessors=target_preprocessors,
        metric_primary=metric_primary,
        metric_secondary=metric_secondary,
        label_primary=label_primary,
        label_secondary=label_secondary,
        bar_width=bar_width,
        show_sem=show_sem,
        hatch=hatch,
        include_legend=include_legend,
        colors=colors,
        figsize=figsize,
        ax=ax,
        axis_config=axis_config,
    )

def plot_between_test_cohort_metrics_overall(
    meta_summary,
    model_names,
    metrics=("AUROC", "Brier"),
    set_name="test_out",
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
    """Plot pooled metrics from ``summarize_loco_results_all`` for one model.

    Parameters
    ----------
    meta_summary : dict
        Output of ``summarize_loco_results_all``.
    model_name : Sequence[str]
        Tuple/list with one or two model identifiers to visualize (e.g., ('LR', 'RF')).
    metrics : sequence of str, default ('AUROC', 'Brier')
        Two metrics to visualise; first metric uses the left axis.
    set_name : str, default 'test_out'
        Evaluation split to plot ('train', 'val', 'test_in', 'test_out').
    preprocessors : sequence of str, optional
        Order of preprocessors; defaults to order observed in the summary.
    palette : str or mapping, default 'okabe_ito'
        Palette name or explicit mapping for preprocessors.
    metric_labels : sequence of str, optional
        Display names for the metrics; defaults to ``metrics``.
    figsize : tuple of float, default (7.0, 5.5)
        Size of the matplotlib figure if ``ax`` is ``None``.
    bar_width : float, default 0.28
        Width of each bar per metric.
    show_sem : bool, default True
        Whether to draw SEM-based error bars when available.
    hatch : str, default '///'
        Hatch pattern for the secondary metric bars.
    include_legend : bool, default True
        Whether to include a legend describing preprocessors and metrics.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on; if ``None`` a new figure and axes are created.
    axis_config : mapping or callable, optional
        Optional axis overrides forwarded to ``plot_between_test_cohort_metrics_from_summary``.
    preprocessor_parser : callable, optional
        Custom function ``(model_label, model_key) -> preprocessor``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot.
    ax : matplotlib.axes.Axes
        Primary axis for the first metric.
    ax_secondary : matplotlib.axes.Axes
        Twin axis for the second metric.
    """

    if isinstance(model_names, str):
        model_names = (model_names,)
    elif isinstance(model_names, (list, tuple)):
        model_names = tuple(model_names)
    else:
        raise TypeError("model_names must be a string or a sequence of strings.")

    model_names = tuple(dict.fromkeys([name for name in model_names if isinstance(name, str) and name.strip()]))
    if not model_names:
        raise ValueError("model_names must provide at least one non-empty identifier.")
    if len(model_names) > 2:
        raise ValueError("model_names can contain at most two entries for comparison.")

    if len(metrics) != 2:
        raise ValueError("metrics must contain exactly two entries.")
    if metric_labels is not None and len(metric_labels) != 2:
        raise ValueError("metric_labels must contain exactly two entries when provided.")

    disease_summary = meta_summary.get('disease')
    if disease_summary is None:
        raise ValueError("meta_summary does not contain a 'disease' section.")
    if set_name not in disease_summary:
        available_sets = ", ".join(sorted(disease_summary.keys()))
        raise ValueError(
            f"set_name='{set_name}' not found. Available sets: {available_sets}."
        )

    df_set = disease_summary[set_name]
    if df_set is None or df_set.empty:
        raise ValueError(
            f"No data available for set '{set_name}'. Ensure meta-analysis was computed."
        )

    df_models = []
    missing_models = []
    model_series = df_set['Model'].astype(str)
    # Match the parsed estimator exactly; substring matching on the raw label
    # would make 'LR' match 'XB+CLR'.
    estimator_series = model_series.apply(lambda s: parse_pipeline(s)[1])
    for target in model_names:
        mask = estimator_series == target
        if not mask.any():
            missing_models.append(target)
            continue
        df_target = df_set.loc[mask].copy()
        df_target['ModelKey'] = target
        df_models.append(df_target)

    if missing_models:
        missing_str = ", ".join(missing_models)
        raise ValueError(
            f"Models [{missing_str}] not found in summary for set '{set_name}'."
        )

    df_model = pd.concat(df_models, ignore_index=True)
    if 'Metric' not in df_model:
        raise ValueError("Expected a 'Metric' column in the summary DataFrame.")

    df_model['Metric'] = df_model['Metric'].astype(str)
    df_model = df_model[df_model['Metric'].isin(metrics)]
    if df_model.empty:
        raise ValueError(
            f"Requested metrics {metrics} not available for models {model_names}."
        )

    numeric_cols = [col for col in ['Pooled Mean', 'Lower CI', 'Upper CI', 'Pooled SEM'] if col in df_model]
    for col in numeric_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

    if preprocessor_parser is None:
        def _preprocessor_parser(model_label, _model_key):
            preprocessor, _ = parse_pipeline(model_label)
            return preprocessor
    else:
        _preprocessor_parser = preprocessor_parser

    df_model['Preprocessor'] = df_model.apply(
        lambda row: _preprocessor_parser(row['Model'], row.get('ModelKey', model_names[0])), axis=1
    )
    df_model['Preprocessor'] = df_model['Preprocessor'].replace('', 'Unknown').fillna('Unknown')

    summary_rows = []
    for (model_key, proc, metric), grp in df_model.groupby(['ModelKey', 'Preprocessor', 'Metric'], dropna=False):
        row = grp.iloc[0]
        summary_rows.append({
            'ModelKey': model_key,
            'Preprocessor': proc,
            'Metric': metric,
            'Pooled Mean': row.get('Pooled Mean', np.nan),
            'Pooled SEM': row.get('Pooled SEM', np.nan),
            'Lower CI': row.get('Lower CI', np.nan),
            'Upper CI': row.get('Upper CI', np.nan)
        })
    summary_df = pd.DataFrame(summary_rows)
    return plot_between_test_cohort_metrics_from_summary(
        summary_df=summary_df,
        model_names=model_names,
        preprocessor_names=preprocessor_names,
        metrics=metrics,
        palette=palette,
        metric_labels=metric_labels,
        figsize=figsize,
        bar_width=bar_width,
        show_sem=show_sem,
        hatch=hatch,
        include_legend=include_legend,
        ax=ax,
        axis_config=axis_config,
    )