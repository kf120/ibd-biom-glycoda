# -*- coding: utf-8 -*-
"""
@author: Konstantinos Flevaris
"""
import logging
import warnings

import numpy as np
from scipy.stats import t, sem
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, matthews_corrcoef, balanced_accuracy_score
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__)

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

THRESHOLD_METRIC_NAMES = frozenset(
    {'MCC', 'BAcc', 'Sensitivity', 'Precision', 'Specificity', 'NPV', 'FPR', 'FNR', 'MCR'}
)
# The model-based c-statistic is a binary-only quantity: it is defined on a
# single predicted probability per participant. The multiclass branch returns
# NaN for it rather than inventing a macro-average.
BINARY_ONLY_METRIC_NAMES = frozenset({'ModelBasedAUROC'})

ALL_METRIC_NAMES = (
    frozenset({'AUROC', 'AUPRC', 'LogLoss', 'Brier'})
    | THRESHOLD_METRIC_NAMES
    | BINARY_ONLY_METRIC_NAMES
)

# Support counts accompany every metric dictionary so that no subgroup or cohort
# number can be read without the sample it came from. They are summed, not
# averaged, when fold results are combined, so they are named separately from the
# scoring metrics and are returned regardless of the requested ``metrics`` subset.
SUPPORT_KEYS = ('n', 'n_cases', 'n_controls')

# 1.0 when the slice had both outcome classes and its metrics were computed,
# 0.0 when the metrics are NaN because the slice was not estimable.
ESTIMABLE_KEY = 'estimable'

# Case fraction is deliberately NOT stored beside the support counts. Fold
# aggregation sums counts and averages metrics, and the mean of per-fold case
# fractions is not the case fraction of the pooled sample unless every fold is
# the same size. Derive it from the summed ``n_cases`` and ``n`` instead, which
# is what ``case_fraction`` below does.

LOWER_IS_BETTER_KEYWORDS = frozenset({
    'logloss', 'loss', 'brier', 'mce', 'mcr', 'fpr', 'fnr',
    'error', 'nll', 'rmse', 'mae', 'mse', 'misclassification'
})

# Minimum support below which a subgroup estimate is flagged rather than read as
# evidence. Pre-specified in the analysis plan; not tuned to observed results.
MIN_CASES_FOR_SUPPORT = 20
MIN_CONTROLS_FOR_SUPPORT = 20

SUPPORT_FLAG_OK = 'OK'
SUPPORT_FLAG_LOW = 'LOW SUPPORT'
SUPPORT_FLAG_NOT_ESTIMABLE = 'NOT ESTIMABLE'


def is_lower_better(metric_name):
    """Return True if lower values of ``metric_name`` indicate better performance.

    Shared by the generalization-gap plots and the LOCO worst-cohort summary so
    new metrics get a consistent direction in both.
    """
    name_norm = metric_name.lower().replace('_', ' ')
    return any(keyword in name_norm for keyword in LOWER_IS_BETTER_KEYWORDS)


def compute_support(y_true):
    """Return the support counts that accompany every metric dictionary.

    Parameters
    ----------
    y_true : array-like
        Observed binary labels, where 1 denotes a case.

    Returns
    -------
    dict
        ``n``, ``n_cases``, and ``n_controls``. With more than two label values
        the case/control split is meaningless, so those two are NaN while ``n``
        still reports the slice size.
    """
    y = np.asarray(y_true)
    if np.unique(y).size > 2:
        logger.debug("Case/control support undefined for %d label values.", np.unique(y).size)
        return {'n': int(y.size), 'n_cases': np.nan, 'n_controls': np.nan}
    n_cases = int(np.sum(y == 1))
    return {'n': int(y.size), 'n_cases': n_cases, 'n_controls': int(y.size) - n_cases}


def case_fraction(support):
    """Return cases divided by total from a support dictionary, or NaN if empty.

    Derived on demand rather than stored, so that it stays correct after fold
    aggregation sums the counts.
    """
    n = support.get('n', 0)
    if not n:
        return np.nan
    return support['n_cases'] / n


def support_flag(
    support,
    min_cases=MIN_CASES_FOR_SUPPORT,
    min_controls=MIN_CONTROLS_FOR_SUPPORT,
):
    """Classify a group's support as OK, low, or non-estimable.

    ``NOT ESTIMABLE`` takes precedence over ``LOW SUPPORT``: a group with no
    cases cannot yield a discrimination or threshold metric at all, so flagging
    it as merely sparse would overstate what is there.

    Parameters
    ----------
    support : dict
        Dictionary containing ``n``, ``n_cases``, and ``n_controls``.
    min_cases : int, optional
        Case count below which the group is flagged. Default 20.
    min_controls : int, optional
        Control count below which the group is flagged. Default 20.

    Returns
    -------
    str
        One of ``NOT ESTIMABLE``, ``LOW SUPPORT``, or ``OK``.
    """
    n_cases = support.get('n_cases', 0)
    n_controls = support.get('n_controls', 0)
    if n_cases == 0 or n_controls == 0:
        return SUPPORT_FLAG_NOT_ESTIMABLE
    if n_cases < min_cases or n_controls < min_controls:
        return SUPPORT_FLAG_LOW
    return SUPPORT_FLAG_OK


def compute_model_based_c_statistic(y_pred_proba):
    """Compute the c-statistic expected under the model's own predicted risks.

    This is the discrimination a perfectly calibrated model would achieve in a
    population with this distribution of predicted probabilities, computed from
    the predictions alone without using the observed outcomes. Comparing it with
    the observed AUROC separates two explanations for a low value in a held-out
    cohort: a narrow spread of predicted risks means the cohort is intrinsically
    hard to separate (case-mix), whereas an observed AUROC well below the
    model-based value means the coefficients did not transport.

    Each ordered pair of distinct participants ``(i, j)`` is weighted by the
    probability that ``i`` is a case and ``j`` a control, ``p_i * (1 - p_j)``, and
    contributes 1 when ``p_i > p_j`` and 0.5 when the two predictions are tied.

    Implemented in ``O(n log n)`` by sorting once and walking tie blocks, since a
    pairwise form would be quadratic in the cohort size.

    Parameters
    ----------
    y_pred_proba : array-like
        Predicted probability of the case class, one per participant.

    Returns
    -------
    float
        Model-based c-statistic. NaN when fewer than two participants are
        supplied, or when every prediction is 0 or every prediction is 1, which
        leaves no case-control pair any weight.

    References
    ----------
    van Klaveren D, Gonen M, Steyerberg EW, Vergouwe Y. A new concordance
    measure for risk prediction models in external validation settings.
    Stat Med. 2016;35(23):4136-4152.
    """
    p = np.asarray(y_pred_proba, dtype=float).ravel()
    if p.size < 2:
        logger.debug("Model-based c-statistic not estimable: fewer than two predictions.")
        return np.nan
    if np.any(np.isnan(p)):
        logger.debug("Model-based c-statistic not estimable: predictions contain NaN.")
        return np.nan

    order = np.argsort(p, kind='mergesort')
    p_sorted = p[order]
    case_w = p_sorted                # weight of being a case
    control_w = 1.0 - p_sorted       # weight of being a control

    total_control_w = control_w.sum()
    # Denominator excludes self-pairs: a pair needs two distinct participants.
    denominator = float(np.sum(case_w * (total_control_w - control_w)))
    if denominator <= 0.0:
        logger.debug("Model-based c-statistic not estimable: no case-control pair carries weight.")
        return np.nan

    # Cumulative control weight strictly below each tie block, and the block's
    # own total, so ties can be given half credit without a pairwise loop.
    cum_control_w = np.concatenate(([0.0], np.cumsum(control_w)))
    block_starts = np.flatnonzero(np.concatenate(([True], p_sorted[1:] != p_sorted[:-1])))
    block_ends = np.concatenate((block_starts[1:], [p_sorted.size]))

    block_index = np.repeat(np.arange(block_starts.size), block_ends - block_starts)
    control_w_below = cum_control_w[block_starts][block_index]
    control_w_in_block = (cum_control_w[block_ends] - cum_control_w[block_starts])[block_index]

    concordant_w = control_w_below + 0.5 * (control_w_in_block - control_w)
    numerator = float(np.sum(case_w * concordant_w))

    return numerator / denominator


def compute_scoring_metrics(
    y_true,
    y_pred_proba,
    metrics=None,
    model_classes=None,
    threshold=0.5,
):
    """Compute discrimination, probability-accuracy, and threshold metrics.

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
        Subset of metric names to return; defaults to the full set. The support
        counts in ``SUPPORT_KEYS`` are returned regardless of this filter.
    model_classes : array-like, optional
        Explicit class ordering for the probability columns.
    threshold : float, optional
        Probability above which a participant is classified as a case in the
        binary branch. Default 0.5, matching the historical fixed cut-off. A
        probability exactly equal to ``threshold`` classifies as a control.
        Ignored by the multiclass branch, which uses argmax.

    Returns
    -------
    dict
        Discrimination, probability-accuracy, and threshold scores, always
        accompanied by the ``n``, ``n_cases``, and ``n_controls`` support counts
        so that no value can be read without its sample.

    Notes
    -----
    Threshold metrics use a zero-denominator convention of 0.0 rather than NaN
    (see ``tests/evaluation/test_metric_conventions.py``). Read them beside the
    support counts: a specificity of 1.0 from zero controls means "no controls",
    not "no false positives".
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    requested = set(metrics) if metrics is not None else set(ALL_METRIC_NAMES)
    need_auroc = 'AUROC' in requested
    need_auprc = 'AUPRC' in requested
    need_logloss = 'LogLoss' in requested
    need_brier = 'Brier' in requested
    need_model_based = 'ModelBasedAUROC' in requested
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
    tpr = precision = tnr = fpr = fnr = misclassification_rate = mcc = bacc = None
    npv = None
    # Binary-only quantity. The multiclass branch leaves it NaN rather than
    # macro-averaging a measure that has no accepted multiclass form.
    model_based_auroc = np.nan

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

        if need_model_based:
            model_based_auroc = compute_model_based_c_statistic(y_pred_proba_aligned)

        if need_threshold:
            # Strictly above the threshold classifies as a case, so a probability
            # exactly equal to it predicts the control class.
            y_pred = (y_pred_proba_aligned > threshold).astype(int)

            # Confusion matrix elements
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            # Compute threshold-based metrics. Empty denominators return 0.0, the
            # convention already fixed for the pre-existing metrics; NPV follows
            # it so the whole threshold family reads the same way.
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
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
    if need_model_based:
        all_results['ModelBasedAUROC'] = model_based_auroc
    if need_threshold:
        all_results['MCC'] = mcc
        all_results['BAcc'] = bacc
        all_results['Sensitivity'] = tpr
        all_results['Precision'] = precision
        # Undefined in the multiclass branch, which has no NPV macro-average.
        all_results['NPV'] = npv if npv is not None else np.nan
        all_results['Specificity'] = tnr
        all_results['FPR'] = fpr
        all_results['FNR'] = fnr
        all_results['MCR'] = misclassification_rate

    if metrics is not None:
        results = {metric: all_results[metric] for metric in metrics if metric in all_results}
    else:
        results = all_results

    # Support counts bypass the metric filter: every number above is
    # uninterpretable without them, so a caller must not be able to request a
    # subgroup metric without the sample it was computed on.
    results.update(compute_support(y_true))

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
