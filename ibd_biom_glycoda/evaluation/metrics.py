# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
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
    brier = brier_score_loss(y_true, y_proba)
    base_rate = np.mean(y_true)
    uncertainty = base_rate * (1 - base_rate)

    unique_preds = np.unique(y_proba)
    reliability = 0.0
    resolution = 0.0

    for pred_val in unique_preds:
        mask = y_proba == pred_val
        n_k = np.sum(mask)
        if n_k == 0:
            continue

        observed_freq = np.mean(y_true[mask])
        reliability += (n_k / len(y_true)) * (pred_val - observed_freq) ** 2
        resolution += (n_k / len(y_true)) * (observed_freq - base_rate) ** 2

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
        return {
            'ece': 0.0,
            'brier': brier_score_loss(y_true, y_proba),
            'reliability': 0.0,
            'resolution': 0.0,
            'uncertainty': np.mean(y_true) * (1 - np.mean(y_true))
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

def compute_scoring_metrics(y_true, y_pred_proba, metrics=None, model_classes=None):
    """Compute discrimination, calibration, and threshold metrics.

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
    
    # Initialize variables
    ece = None
    reliability = None
    resolution = None
    uncertainty = None
    
    if n_test_classes > 2 or n_model_classes > 2:
        # Multiclass case
        
        # Align predictions with test classes
        class_indices = np.array([np.where(model_classes == c)[0][0] for c in test_classes])
        y_pred_proba_aligned = y_pred_proba[:, class_indices]
        
        # Renormalize probabilities
        y_pred_proba_aligned = y_pred_proba_aligned / y_pred_proba_aligned.sum(axis=1, keepdims=True)
        
        # Binarize true labels using test classes
        y_true_bin = label_binarize(y_true, classes=test_classes)
        
        # Handle case where only two classes appear in the test set while the model is multiclass
        if n_test_classes == 2 and y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        
        # Compute non-threshold-based metrics
        auroc = roc_auc_score(y_true_bin, y_pred_proba_aligned, multi_class='ovr', average='macro')
        auprc = average_precision_score(y_true_bin, y_pred_proba_aligned, average='macro')
        log_loss_value = log_loss(y_true, y_pred_proba_aligned)
        brier_score = np.mean(np.sum((y_true_bin - y_pred_proba_aligned) ** 2, axis=1))

        # Compute calibration metrics (OVR macro-averaging)
        ece_list = []
        reliability_list = []
        resolution_list = []
        uncertainty_list = []
        
        for class_idx, class_label in enumerate(test_classes):
            # One-vs-Rest: binary problem for this class
            y_true_binary = (y_true == class_label).astype(int)
            y_proba_class = y_pred_proba_aligned[:, class_idx]
            
            cal_metrics = compute_calibration_metrics(y_true_binary, y_proba_class)
            ece_list.append(cal_metrics['ece'])
            reliability_list.append(cal_metrics['reliability'])
            resolution_list.append(cal_metrics['resolution'])
            uncertainty_list.append(cal_metrics['uncertainty'])
        
        # Macro-average calibration metrics
        ece = np.mean(ece_list)
        reliability = np.mean(reliability_list)
        resolution = np.mean(resolution_list)
        uncertainty = np.mean(uncertainty_list)

        # Compute threshold-based metrics using OVR macro-averaging
        y_pred = test_classes[np.argmax(y_pred_proba_aligned, axis=1)]
        
        macro_avg_metrics = compute_macro_avg_metrics(y_true, y_pred, n_test_classes)
        
        tpr = macro_avg_metrics['Sensitivity']
        precision = macro_avg_metrics['Precision']
        tnr = macro_avg_metrics['Specificity']
        fpr = macro_avg_metrics['FPR']
        fnr = macro_avg_metrics['FNR']
        misclassification_rate = macro_avg_metrics['MCR']
        mcc = macro_avg_metrics['MCC']
        bacc = balanced_accuracy_score(y_true, y_pred)

    else:
        # Binary case
        if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 1:
            y_pred_proba_aligned = y_pred_proba.ravel()
        else:
            y_pred_proba_aligned = y_pred_proba[:, 1]
        
        # Compute non-threshold-based metrics
        auroc = roc_auc_score(y_true, y_pred_proba_aligned)
        auprc = average_precision_score(y_true, y_pred_proba_aligned)
        log_loss_value = log_loss(y_true, y_pred_proba_aligned)
        brier_score = brier_score_loss(y_true, y_pred_proba_aligned)

        # Compute calibration metrics for binary case
        cal_metrics = compute_calibration_metrics(y_true, y_pred_proba_aligned)
        ece = cal_metrics['ece']
        reliability = cal_metrics['reliability']
        resolution = cal_metrics['resolution']
        uncertainty = cal_metrics['uncertainty']

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

    # Compile results
    all_results = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'LogLoss': log_loss_value,
        'Brier': brier_score,
        'MCC': mcc,
        'BAcc': bacc,
        'ECE': ece,
        'Reliability': reliability,
        'Resolution': resolution,
        'Uncertainty': uncertainty,
        'Sensitivity': tpr,
        'Precision': precision,
        'Specificity': tnr,
        'FPR': fpr,
        'FNR': fnr,
        'MCR': misclassification_rate,
    }

    if metrics is not None:
        results = {metric: all_results[metric] for metric in metrics if metric in all_results}
    else:
        results = all_results
        
    return results


def summarize_scoring_metrics(metrics_list):
    """
    Aggregate values of metrics across multiple seeds and round to 3 decimal points.

    Parameters:
    -----------
    metrics_list : list of dict
        List of metric dictionaries from multiple seeds.

    Returns:
    --------
    dict
        Dictionary containing mean and 95% CI for each metric, rounded to 3 decimal points.
    """
    summarized = {}
    for metric in metrics_list[0].keys():
        values = [m[metric] for m in metrics_list]
        mean = np.mean(values)
        ci = t.interval(0.95, len(values)-1, loc=mean, scale=sem(values))
        summarized[metric] = {
            'Mean': round(mean, 3),
            'SEM': round(sem(values), 3),
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
            print(f"Variable '{var}' not found for model '{model_key}' in set '{set_type}'.")
    return aggregated_results