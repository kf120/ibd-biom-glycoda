# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import warnings

import numpy as np
from scipy.stats import t, sem
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, matthews_corrcoef, balanced_accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

def compute_ece(y_true, y_proba, n_bins):
    """
    Return ECE using quantile-style binning via sklearn.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_proba : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration
    
    Returns:
    --------
    ece : float
        Expected Calibration Error with quantile binning
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Use sklearn's calibration_curve with quantile strategy
    # This returns the true fraction of positives and mean predicted probabilities per bin
    prob_true, prob_pred = calibration_curve(
        y_true, 
        y_proba, 
        n_bins=n_bins, 
        strategy='quantile'  # Use quantile binning
    )
    
    # Compute bin sizes
    # Need to reconstruct which samples fall into which bin
    quantiles = np.linspace(0, 1, n_bins + 1)
    bin_edges = np.quantile(y_proba, quantiles)
    bin_edges[-1] = bin_edges[-1] + 1e-8  # Ensure last bin captures max values
    
    # Assign samples to bins
    bin_indices = np.digitize(y_proba, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    # Compute ECE
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(len(prob_true)):
        # Count samples in this bin
        n_i = np.sum(bin_indices == i)
        
        if n_i > 0:
            bin_error = np.abs(prob_true[i] - prob_pred[i])
            ece += (n_i / n_samples) * bin_error
    
    return ece


def compute_brier_score_decomposition(y_true, y_proba):
    """Return Brier score decomposition terms."""
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)

    brier = brier_score_loss(y_true, y_proba)
    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1 - base_rate)
    n = len(y_true)

    # Aggregate samples with identical predicted probabilities.
    unique_preds, inverse, counts = np.unique(
        y_proba, return_inverse=True, return_counts=True
    )
    inverse = inverse.ravel()
    sum_true_per_bin = np.bincount(inverse, weights=y_true, minlength=len(unique_preds))
    observed_freq = sum_true_per_bin / counts
    weights = counts / n

    reliability = np.sum(weights * (unique_preds - observed_freq) ** 2)
    resolution = np.sum(weights * (observed_freq - base_rate) ** 2)

    resolution_ratio = resolution / uncertainty if uncertainty > 0 else np.nan
    decomposition_check = uncertainty - resolution + reliability

    return {
        'brier': brier,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
        'resolution_ratio': resolution_ratio,
        'decomposition_check': decomposition_check
    }

def compute_calibration_metrics(y_true, y_proba, n_bins=10):
    """
    Compute calibration metrics: ECE and Brier score decomposition.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities
    n_bins : int, default=10
        Number of bins for ECE only
    
    Returns
    -------
    metrics : dict
        Dictionary with calibration metrics
    """    
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    # Handle edge case: if all predictions are the same
    if len(np.unique(y_proba)) == 1:
        base_rate = np.mean(y_true)
        uncertainty = base_rate * (1 - base_rate)
        resolution = 0.0
        return {
            'ece': 0.0,
            'brier': brier_score_loss(y_true, y_proba),
            'reliability': 0.0,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'resolution_ratio': (resolution / uncertainty) if uncertainty > 0 else np.nan,
        }
    
    ece = compute_ece(y_true, y_proba, n_bins)
    brier_metrics = compute_brier_score_decomposition(y_true, y_proba)

    return {
        'ece': ece,
        **brier_metrics
    }

def compute_ovr_metrics(y_true, y_pred, n_classes):
    """Compute one-versus-rest threshold metrics for each observed class.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True integer-encoded labels.
    y_pred : array-like of shape (n_samples,)
        Predicted integer-encoded labels.
    n_classes : int
        Total number of classes the model can predict.

    Returns
    -------
    dict
        Per-class metric dictionaries covering sensitivity, specificity, and related scores.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class_metrics = {}
    
    # Only evaluate on classes actually present
    present = [cls for cls in range(n_classes) if cls in y_true]
    
    for cls in present:
        # Binarize for one-vs-rest
        true_bin = (y_true == cls).astype(int)
        pred_bin = (y_pred == cls).astype(int)
        
        tp = np.sum((true_bin == 1) & (pred_bin == 1))
        tn = np.sum((true_bin == 0) & (pred_bin == 0))
        fp = np.sum((true_bin == 0) & (pred_bin == 1))
        fn = np.sum((true_bin == 1) & (pred_bin == 0))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        mcr = (fp + fn) / len(true_bin)
        mcc = matthews_corrcoef(true_bin, pred_bin) if len(np.unique(pred_bin)) > 1 else 0
        
        per_class_metrics[cls] = {
            'Recall (TPR)': recall,
            'Precision': precision,
            'TNR (Specificity)': tnr,
            'FPR (1 - Specificity)': fpr,
            'FNR (1 - TPR)': fnr,
            'MCR': mcr,
            'MCC': mcc
            }
    
    return per_class_metrics

def compute_macro_avg_metrics(y_true, y_pred, n_classes):
    """Compute macro-averaged one-versus-rest metrics for multiclass tasks.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True integer-encoded labels.
    y_pred : array-like of shape (n_samples,)
        Predicted integer-encoded labels.
    n_classes : int
        Total number of classes the model can predict.

    Returns
    -------
    dict
        Macro-averaged threshold metrics across observed classes.
    """
    # Get per-class metrics
    per_class = compute_ovr_metrics(y_true, y_pred, n_classes)
    
    # Macro-average over present classes
    metrics_sum = {key: 0.0 for key in next(iter(per_class.values()))}
    n_present = len(per_class)
    
    for cls_metrics in per_class.values():
        for key, val in cls_metrics.items():
            metrics_sum[key] += val
    
    # Divide by number of classes present to get macro-average
    macro_metrics = {key: metrics_sum[key] / n_present for key in metrics_sum}
    return macro_metrics

CALIBRATION_METRIC_NAMES = frozenset(
    {'ECE', 'Reliability', 'Resolution', 'Uncertainty', 'Resolution Ratio'}
)
THRESHOLD_METRIC_NAMES = frozenset(
    {'MCC', 'BAcc', 'Sensitivity', 'Precision', 'Specificity', 'FPR', 'FNR', 'MCR'}
)
ALL_METRIC_NAMES = (
    frozenset({'AUROC', 'AUPRC', 'LogLoss', 'Brier'})
    | CALIBRATION_METRIC_NAMES
    | THRESHOLD_METRIC_NAMES
)

LOWER_IS_BETTER_KEYWORDS = frozenset({
    'ece', 'logloss', 'loss', 'brier', 'mce', 'mcr', 'fpr', 'fnr',
    'uncertainty', 'error', 'nll', 'rmse', 'mae', 'mse', 'calibration',
    'misclassification', 'reliability'
})


def is_lower_better(metric_name):
    """Return True if lower values of ``metric_name`` indicate better performance.

    Shared by the generalization-gap plots and the LOCO worst-cohort summary so
    new metrics get a consistent direction in both.
    """
    name_norm = metric_name.lower().replace('_', ' ')
    return any(keyword in name_norm for keyword in LOWER_IS_BETTER_KEYWORDS)


def compute_scoring_metrics(y_true, y_pred_proba, metrics=None, model_classes=None):
    """Compute discrimination, calibration, and threshold metrics.

    Only the metric families needed to satisfy ``metrics`` are computed;
    requesting a small subset skips the unrequested families entirely
    rather than computing everything and filtering afterwards.

    Parameters
    ----------
    y_true : array-like
        Observed labels.
    y_pred_proba : array-like
        Predicted probabilities aligned with model classes.
    metrics : list, optional
        Subset of metric names to return; defaults to the full set.
    model_classes : array-like, optional
        Explicit class ordering for the probability columns.

    Returns
    -------
    dict
        Dictionary containing discrimination scores, calibration summaries, and threshold metrics.
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    requested = set(metrics) if metrics is not None else set(ALL_METRIC_NAMES)
    need_auroc = 'AUROC' in requested
    need_auprc = 'AUPRC' in requested
    need_logloss = 'LogLoss' in requested
    need_brier = 'Brier' in requested
    need_calibration = bool(requested & CALIBRATION_METRIC_NAMES)
    need_threshold = bool(requested & THRESHOLD_METRIC_NAMES)

    # Get unique classes in test set
    test_classes = np.unique(y_true)
    n_test_classes = len(test_classes)

    # Infer model classes if not provided
    if model_classes is None:
        n_model_classes = y_pred_proba.shape[1] if y_pred_proba.ndim > 1 else 2
        model_classes = np.arange(n_model_classes)
    else:
        model_classes = np.array(model_classes)
        n_model_classes = len(model_classes)

    # Check if all test classes are in model classes
    if not np.all(np.isin(test_classes, model_classes)):
        raise ValueError(
            f"Test set contains classes {test_classes} not present in "
            f"model classes {model_classes}"
        )

    auroc = auprc = log_loss_value = brier_score = None
    ece = reliability = resolution = uncertainty = resolution_ratio = None
    tpr = precision = tnr = fpr = fnr = misclassification_rate = mcc = bacc = None

    if n_test_classes > 2 or n_model_classes > 2:
        # Multiclass case

        # Align predictions with test classes
        class_indices = np.array([np.where(model_classes == c)[0][0] for c in test_classes])
        y_pred_proba_aligned = y_pred_proba[:, class_indices]

        # Renormalize probabilities
        y_pred_proba_aligned = y_pred_proba_aligned / y_pred_proba_aligned.sum(axis=1, keepdims=True)

        # y_true_bin is shared by AUROC, AUPRC, and Brier
        if need_auroc or need_auprc or need_brier:
            y_true_bin = label_binarize(y_true, classes=test_classes)

            # Expand binary labels to match multiclass probability columns.
            if n_test_classes == 2 and y_true_bin.shape[1] == 1:
                y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])

            if need_auroc:
                auroc = roc_auc_score(y_true_bin, y_pred_proba_aligned, multi_class='ovr', average='macro')
            if need_auprc:
                auprc = average_precision_score(y_true_bin, y_pred_proba_aligned, average='macro')
            if need_brier:
                brier_score = np.mean(np.sum((y_true_bin - y_pred_proba_aligned) ** 2, axis=1))

        if need_logloss:
            log_loss_value = log_loss(y_true, y_pred_proba_aligned)

        if need_calibration:
            # Compute calibration metrics (OVR macro-averaging)
            ece_list = []
            reliability_list = []
            resolution_list = []
            uncertainty_list = []
            resolution_ratio_list = []

            for class_idx, class_label in enumerate(test_classes):
                # One-vs-Rest: binary problem for this class
                y_true_binary = (y_true == class_label).astype(int)
                y_proba_class = y_pred_proba_aligned[:, class_idx]

                cal_metrics = compute_calibration_metrics(y_true_binary, y_proba_class)
                ece_list.append(cal_metrics['ece'])
                reliability_list.append(cal_metrics['reliability'])
                resolution_list.append(cal_metrics['resolution'])
                uncertainty_list.append(cal_metrics['uncertainty'])
                resolution_ratio_list.append(cal_metrics['resolution_ratio'])

            # Macro-average calibration metrics
            ece = np.mean(ece_list)
            reliability = np.mean(reliability_list)
            resolution = np.mean(resolution_list)
            uncertainty = np.mean(uncertainty_list)
            resolution_ratio = np.nanmean(resolution_ratio_list)

        if need_threshold:
            # Compute threshold-based metrics using OVR macro-averaging
            y_pred = test_classes[np.argmax(y_pred_proba_aligned, axis=1)]

            macro_avg_metrics = compute_macro_avg_metrics(y_true, y_pred, n_test_classes)

            tpr = macro_avg_metrics['Recall (TPR)']
            precision = macro_avg_metrics['Precision']
            tnr = macro_avg_metrics['TNR (Specificity)']
            fpr = macro_avg_metrics['FPR (1 - Specificity)']
            fnr = macro_avg_metrics['FNR (1 - TPR)']
            misclassification_rate = macro_avg_metrics['MCR']
            mcc = macro_avg_metrics['MCC']
            bacc = balanced_accuracy_score(y_true, y_pred)

    else:
        # Binary case
        if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
            y_pred_proba_aligned = y_pred_proba.ravel()
        else:
            y_pred_proba_aligned = y_pred_proba[:, 1]

        if need_auroc:
            auroc = roc_auc_score(y_true, y_pred_proba_aligned)
        if need_auprc:
            auprc = average_precision_score(y_true, y_pred_proba_aligned)
        if need_logloss:
            log_loss_value = log_loss(y_true, y_pred_proba_aligned)
        if need_brier:
            brier_score = brier_score_loss(y_true, y_pred_proba_aligned)

        if need_calibration:
            cal_metrics = compute_calibration_metrics(y_true, y_pred_proba_aligned)
            ece = cal_metrics['ece']
            reliability = cal_metrics['reliability']
            resolution = cal_metrics['resolution']
            uncertainty = cal_metrics['uncertainty']
            resolution_ratio = cal_metrics['resolution_ratio']

        if need_threshold:
            # Compute threshold-based metrics using the conventional half threshold
            y_pred = (y_pred_proba_aligned > 0.5).astype(int)

            # Confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            # Compute threshold-based metrics
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            misclassification_rate = (fp + fn) / len(y_true)

            mcc = matthews_corrcoef(y_true, y_pred)
            bacc = balanced_accuracy_score(y_true, y_pred)

    # Compile only the families actually computed
    all_results = {}
    if need_auroc:
        all_results['AUROC'] = auroc
    if need_auprc:
        all_results['AUPRC'] = auprc
    if need_logloss:
        all_results['LogLoss'] = log_loss_value
    if need_brier:
        all_results['Brier'] = brier_score
    if need_calibration:
        all_results['ECE'] = ece
        all_results['Reliability'] = reliability
        all_results['Resolution'] = resolution
        all_results['Uncertainty'] = uncertainty
        all_results['Resolution Ratio'] = resolution_ratio
    if need_threshold:
        all_results['MCC'] = mcc
        all_results['BAcc'] = bacc
        all_results['Sensitivity'] = tpr
        all_results['Precision'] = precision
        all_results['Specificity'] = tnr
        all_results['FPR'] = fpr
        all_results['FNR'] = fnr
        all_results['MCR'] = misclassification_rate

    if metrics is not None:
        results = {metric: all_results[metric] for metric in metrics if metric in all_results}
    else:
        results = all_results

    return results


def summarize_scoring_metrics(metrics_list):
    """
    Aggregate values of metrics across multiple seeds.

    Mean and CI are rounded for display; SEM is kept at full precision because
    ``compute_pooled_stats`` uses it as an inverse-variance weight and skips
    ``SEM <= 0``, so a rounded-to-zero SEM would drop that group from the pool.

    Metrics are taken as the union across seeds; one present in only some seeds
    is summarized over those seeds with a warning.

    Parameters:
    -----------
    metrics_list : list of dict
        List of metric dictionaries from multiple seeds.

    Returns:
    --------
    dict
        Dictionary containing mean, full-precision SEM, and 95% CI for each metric.
    """
    if not metrics_list:
        return {}

    all_metric_names = set()
    for metric_dict in metrics_list:
        all_metric_names.update(metric_dict.keys())

    summarized = {}
    for metric in sorted(all_metric_names):
        values = [m[metric] for m in metrics_list if metric in m]
        if len(values) != len(metrics_list):
            warnings.warn(
                f"Metric '{metric}' is present in only {len(values)}/{len(metrics_list)} "
                "seed(s); summarizing over the seeds where it is available. Its SEM/CI "
                "may not be directly comparable to metrics computed from the full seed count."
            )
        mean = np.mean(values)
        metric_sem = sem(values)
        ci = t.interval(0.95, len(values)-1, loc=mean, scale=metric_sem)
        summarized[metric] = {
            'Mean': round(mean, 3),
            'SEM': float(metric_sem),
            'CI': (round(ci[0], 3), round(ci[1], 3))
        }
    return summarized

def aggregate_all_metrics(metrics, model_keys, set_type, var, all_metrics):
    """Aggregate scoring metrics for each model and variable of interest.

    Parameters
    ----------
    metrics : dict
        Nested metrics structured by model, split, and variable.
    model_keys : iterable
        Identifiers for the models to aggregate.
    set_type : str
        Evaluation split to read (for example ``test_in``).
    var : str
        Variable key within the nested metrics.
    all_metrics : iterable
        Metric names expected in the output.

    Returns
    -------
    dict
        Aggregated statistics per model and metric.
    """
    aggregated_results = {}
    for model_key in model_keys:
        aggregated_results[model_key] = {}
        if var in metrics[model_key][set_type]:
            # Aggregate the scoring metrics for the given model, set type, and variable.
            model_agg = summarize_scoring_metrics(metrics[model_key][set_type][var])
            for metric in all_metrics:
                if metric in model_agg:
                    aggregated_results[model_key][metric] = {
                        'Mean': model_agg[metric]['Mean'],
                        'SEM': model_agg[metric]['SEM'],
                        'CI': model_agg[metric]['CI']
                    }
        else:
            warnings.warn(
                f"Variable '{var}' not found for model '{model_key}' in set '{set_type}'. "
                "This model produced no results for this split -- check for a stale "
                "PIPELINE_KEYS/MODEL_KEYS mismatch."
            )
    return aggregated_results
