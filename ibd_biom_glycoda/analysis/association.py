# -*- coding: utf-8 -*-
"""
@author: Kostis Flevaris
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from scipy import stats
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from adjustText import adjust_text

from ibd_biom_glycoda.utils.helpers import natural_sort_key

def preprocess_linear_modeling(df, disease_cat=['Non-IBD', 'CD', 'UC'], center_age=True):
    """
    Ensure categorical encoding for ``Sex``, ``Cohort`` and ``DISEASE`` 
    and optionally center the ``Age`` column around 40 years old

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing at least ``Age``, ``Sex``, ``Cohort`` and
        ``DISEASE`` columns.
    disease_cat : list of str, default ['Non-IBD', 'CD', 'UC']
        Ordered categories to use for the ``DISEASE`` categorical.
    center_age : bool, default True
        If True, center the ``Age`` column around 40 years old.

    Returns
    -------
    pandas.DataFrame
        Modified copy of ``df`` with standardized age (if requested) and
        appropriate categorical dtypes.
    """
    df = df.copy()

    if center_age:
        # Center Age around 40
        df['Age'] = df['Age'] - 40

    # Set categorical variables
    df["Sex"] = df["Sex"].astype("category")
    df["Cohort"] = df["Cohort"].astype("category")
    df["DISEASE"] = pd.Categorical(df["DISEASE"], categories=disease_cat, ordered=True)

    return df

def check_ols_diagnostics(results_fixed, alpha=0.05, cooks_concern=0.5, verbose = True):
    """
    Per-GP diagnostics that vary across models: functional form (RESET),
    influential observations (Cook's distance), and residual shape.

    Parameters
    ----------
    results_fixed : dict
        Mapping of glycan name -> fitted OLS result.
    alpha : float
        Significance level for RESET test.
    cooks_concern : float
        Cook's distance threshold for genuine concern (default 0.5).
        The conservative 4/n threshold is always reported alongside.
    verbose : bool
        If True, prints summary table and flags.

    Returns
    -------
    pd.DataFrame
        One row per GP with diagnostic metrics.
    """

    diag_recs = []
    for glycan, ols in results_fixed.items():
        resid = ols.resid
        n = len(resid)
        conservative_threshold = 4 / n

        # Ramsey RESET — functional form misspecification
        try:
            reset_res = linear_reset(ols, power=2, use_f=True)
            reset_p = float(reset_res.pvalue)
            reset_f = float(reset_res.fvalue)
        except Exception as exc:
            reset_p = np.nan
            reset_f = np.nan

        # Cook's distance — influential observations
        infl = ols.get_influence()
        cooks = infl.cooks_distance[0]
        max_cooks = float(cooks.max())
        n_above_conservative = int((cooks > conservative_threshold).sum())
        n_above_concern = int((cooks > cooks_concern).sum())

        # Cook's distance distribution
        cooks_median = float(np.median(cooks))
        cooks_p95 = float(np.percentile(cooks, 95))
        cooks_p99 = float(np.percentile(cooks, 99))

        # Residual shape
        skew = float(stats.skew(resid))
        kurt = float(stats.kurtosis(resid))

        diag_recs.append({
            'glycan': glycan,
            'n_obs': n,
            'reset_F': reset_f,
            'reset_p': reset_p,
            'reset_fail': reset_p < alpha if np.isfinite(reset_p) else None,
            'max_cooks_d': max_cooks,
            'cooks_median': cooks_median,
            'cooks_p95': cooks_p95,
            'cooks_p99': cooks_p99,
            'n_cooks_above_4n': n_above_conservative,
            'cooks_4n_threshold': conservative_threshold,
            'n_cooks_above_concern': n_above_concern,
            'cooks_concern_threshold': cooks_concern,
            'resid_skew': skew,
            'resid_kurtosis': kurt,
        })

    df_diag = pd.DataFrame(diag_recs)

    if verbose:
        n_reset_fail = df_diag['reset_fail'].sum()
        n_concern = (df_diag['n_cooks_above_concern'] > 0).sum()

        print("=" * 68)
        print("  PER-GP OLS DIAGNOSTICS")
        print("=" * 68)
        print(f"  Models checked:            {len(df_diag)}")
        print(f"  RESET failures (p<{alpha}):   {n_reset_fail} / {len(df_diag)}")
        print(f"  GPs with Cook's D > {cooks_concern}:   {n_concern} / {len(df_diag)}")
        print()

        # Flag potentially problematic GPs (RESET or Cook's > concern)
        problems = df_diag[
            df_diag['reset_fail'] | (df_diag['n_cooks_above_concern'] > 0)
        ]
        if not problems.empty:
            print("  Flagged GPs:")
            for _, row in problems.iterrows():
                flags = []
                if row['reset_fail']:
                    flags.append(f"RESET p={row['reset_p']:.4g}")
                if row['n_cooks_above_concern'] > 0:
                    flags.append(
                        f"{row['n_cooks_above_concern']} obs with "
                        f"Cook's D > {cooks_concern} "
                        f"(max={row['max_cooks_d']:.4f})"
                    )
                print(f"    {row['glycan']:25s} {'; '.join(flags)}")
        else:
            print("  No GPs flagged for genuine concern.")

        # Cook's distance distribution summary
        print()
        print("  Cook's distance distribution across all GPs:")
        print(f"    Median:  {df_diag['cooks_median'].min():.6f} – "
              f"{df_diag['cooks_median'].max():.6f}")
        print(f"    P95:     {df_diag['cooks_p95'].min():.4f} – "
              f"{df_diag['cooks_p95'].max():.4f}")
        print(f"    P99:     {df_diag['cooks_p99'].min():.4f} – "
              f"{df_diag['cooks_p99'].max():.4f}")
        print(f"    Max:     {df_diag['max_cooks_d'].min():.4f} – "
              f"{df_diag['max_cooks_d'].max():.4f}")

        # Conservative 4/n summary
        print()
        print(f"  Note: Using conservative 4/n threshold "
              f"({df_diag['cooks_4n_threshold'].iloc[0]:.4f}),")
        print(f"  all GPs show {df_diag['n_cooks_above_4n'].min()}-"
              f"{df_diag['n_cooks_above_4n'].max()} observations above threshold.")
        print(f"  However, max Cook's D across all GPs is "
              f"{df_diag['max_cooks_d'].max():.4f},")
        print(f"  well below the concern threshold of {cooks_concern}.")
        print("=" * 68)

    return df_diag


def check_vif(ols_fit, vif_threshold=5.0, verbose=True):
    """
    VIF and condition number for one OLS model. Since the design matrix
    is identical across all GPs, this only needs to be run once.

    Parameters
    ----------
    ols_fit : statsmodels OLS result
        Any single fitted model (design matrix is shared).
    vif_threshold : float
        VIF values at or above this are flagged.
    verbose : bool
        If True, prints formatted summary.

    Returns
    -------
    pd.DataFrame
        One row per predictor with VIF values.
    """

    X = ols_fit.model.exog
    names = ols_fit.model.exog_names
    cond_number = float(np.linalg.cond(X))

    vif_recs = []
    for i, name in enumerate(names):
        vif = float(variance_inflation_factor(X, i))
        is_intercept = name.lower() in ('intercept', 'const')
        flagged = (not is_intercept) and np.isfinite(vif) and (vif >= vif_threshold)
        vif_recs.append({
            'predictor': name,
            'VIF': vif,
            'flagged': flagged,
        })

    df_vif = pd.DataFrame(vif_recs)

    if verbose:
        n_flagged = df_vif['flagged'].sum()
        print("=" * 68)
        print("  VIF & CONDITION NUMBER (shared design matrix)")
        print("=" * 68)
        print(f"  Condition number: {cond_number:.1f}")
        print(f"  VIF threshold:    {vif_threshold}")
        print(f"  Predictors flagged: {n_flagged} / {len(df_vif)}")
        print()
        for _, row in df_vif.iterrows():
            flag = " * " if row['flagged'] else ""
            print(f"    {row['predictor']:40s}  VIF = {row['VIF']:.1f}{flag}")
        print("=" * 68)

    return df_vif

def collect_model_terms(glycan, fit, prefixes=None):
    """
    Collect fixed-effect parameter estimates from a model.

    Parameters
    ----------
    glycan : str
        Name of the glycan (used in the output records).
    fit : statsmodels result
        Fitted model result object with ``params``, ``bse`` and ``pvalues``.
    prefixes : list of str
        Parameter name prefixes to collect (e.g. "C(DISEASE)", "Age").

    Returns
    -------
    list of dict
        List of records with keys: 'glycan', 'param', 'est_fixed',
        'se_fixed' and 'pval_fixed'.
    """

    if prefixes is None:
        prefixes = ["C(DISEASE", "Age", "C(Sex"]

    params = fit.params
    ses = fit.bse
    pvals = fit.pvalues

    records = []
    for param in params.index:
        if any(param.startswith(pref) for pref in prefixes):
            records.append({
                'glycan': glycan,
                'param': param,
                'est_fixed': params[param],
                'se_fixed': ses[param],
                'pval_fixed': pvals[param],
            })
    return records

def _run_one_perm(perm_idx, df_model, coda_cols, formula_rhs, cov_type, coef_names, thresholds, seed):
    rng = np.random.default_rng(seed + perm_idx)
    df_perm = df_model.copy()
    df_perm['DISEASE'] = rng.permutation(df_perm['DISEASE'].values)

    perm_pvals = {c: [] for c in coef_names}
    for glycan in coda_cols:
        formula = f"{glycan} ~ {formula_rhs}"
        fit = smf.ols(formula, data=df_perm).fit(cov_type=cov_type)
        for c in coef_names:
            if c in fit.pvalues:
                perm_pvals[c].append(fit.pvalues[c])
            else:
                perm_pvals[c].append(1.0)

    result = {}
    for c in coef_names:
        pvals = np.array(perm_pvals[c])
        result[c] = np.array([(pvals < t).sum() for t in thresholds])
    return result


def run_permutation_fdr(
    df_model, coda_cols, formula_rhs, cov_type, coef_names,
    n_perm=1000, seed=42, n_jobs=-1,
):
    """
    Estimate permutation-based FDR for selected model coefficients.

    Parameters
    ----------
    df_model : pandas.DataFrame
        Modeling dataframe containing response glycans and predictors.
    coda_cols : list of str
        Glycan column names to fit as responses.
    formula_rhs : str
        Right-hand side of the regression formula (predictor terms).
    cov_type : str
        Covariance estimator passed to statsmodels ``fit(cov_type=...)``.
    coef_names : list of str
        Coefficient names for which permutation FDR is computed.
    n_perm : int, default 1000
        Number of disease-label permutations.
    seed : int, default 42
        Base random seed for reproducible permutations.
    n_jobs : int, default -1
        Number of parallel workers for permutations.

    Returns
    -------
    dict of pandas.DataFrame
        Mapping ``coefficient -> results dataframe`` with columns
        ``glycan``, ``raw_p``, and ``perm_fdr``, sorted by ``raw_p``.
    """

    thresholds = np.arange(0.001, 1.001, 0.001)

    # Observed p-values
    obs_pvals = {c: [] for c in coef_names}
    for glycan in coda_cols:
        formula = f"{glycan} ~ {formula_rhs}"
        fit = smf.ols(formula, data=df_model).fit(cov_type=cov_type)
        for c in coef_names:
            if c in fit.pvalues:
                obs_pvals[c].append(fit.pvalues[c])
            else:
                obs_pvals[c].append(np.nan)
    obs_pvals = {c: np.array(v) for c, v in obs_pvals.items()}

    # Parallel permutations via joblib
    perm_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
        delayed(_run_one_perm)(
            i, df_model, coda_cols, formula_rhs, cov_type, coef_names, thresholds, seed
        )
        for i in range(n_perm)
    )

    # Aggregate null counts
    null_counts = {c: np.zeros((n_perm, len(thresholds))) for c in coef_names}
    for i, res in enumerate(perm_results):
        for c in coef_names:
            null_counts[c][i, :] = res[c]

    # Compute FDR per coefficient
    results = {}
    for c in coef_names:
        pvals = obs_pvals[c]
        sorted_idx = np.argsort(pvals)
        adjusted = np.ones(len(pvals))

        for i, idx in enumerate(sorted_idx):
            p = pvals[idx]
            n_disc = (pvals <= p).sum()
            t_idx = max(np.searchsorted(thresholds, p, side='right') - 1, 0)
            mean_null = null_counts[c][:, t_idx].mean()
            if n_disc > 0:
                adjusted[idx] = min(mean_null / n_disc, 1.0)

        for i in range(len(sorted_idx) - 2, -1, -1):
            adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i]], adjusted[sorted_idx[i + 1]])

        results[c] = pd.DataFrame({
            'glycan': coda_cols,
            'raw_p': pvals,
            'perm_fdr': adjusted,
        }).sort_values('raw_p')

    return results


def prepare_volcano_data(df_compare, alpha, correction_method='fdr_bh'):
    """
    Prepare data for volcano plot (main disease effects only).
    
    Parameters:
    -----------
    df_compare : DataFrame
        Comparison dataframe with est_fixed, pval_fixed_adj, param columns
    alpha : float
        Significance threshold
    
    Returns:
    --------
    df_volcano : DataFrame
        Data for volcano plot with columns: Disease, est_fixed, log10p, Significant,
        CI_lower, CI_upper
    fdr_line : float
        Y-axis position for FDR threshold line
    disease_groups : list
        Detected disease groups
    """
    # Filter for main disease effects only (no interactions)
    df_main = df_compare[
        df_compare['param'].str.startswith("C(DISEASE") &
        ~df_compare['param'].str.contains(":", na=False)
    ].copy()
    
    # Extract disease from parameter name
    df_main['Disease'] = df_main['param'].apply(
        lambda x: x.split("[T.")[1].rstrip("]") if "[T." in x else None
    )
    
    # Calculate log10 p-values
    if correction_method == 'fdr_bh':
        df_main['log10p'] = -np.log10(df_main['pval_fixed_adj_bh'])
        df_main['Significant'] = df_main['pval_fixed_adj_bh'] < alpha
    else:
        df_main['log10p'] = -np.log10(df_main['pval_fixed_adj_by'])
        df_main['Significant'] = df_main['pval_fixed_adj_by'] < alpha

    # GP labels for annotations (e.g., GP9 from GP9_FA2_3_G1)
    df_main['GP_label'] = df_main['glycan'].str.extract(r'(GP\d+)', expand=False)

    # Confidence intervals for fixed effects
    df_main['CI_lower'] = df_main['est_fixed'] - 1.96 * df_main['se_fixed']
    df_main['CI_upper'] = df_main['est_fixed'] + 1.96 * df_main['se_fixed']
    
    # FDR line
    fdr_line = -np.log10(alpha)
    
    # Get unique disease groups
    disease_groups = sorted(df_main['Disease'].unique())

    # Mark top 10 points above FDR line for labeling
    df_main['TopLabel'] = False
    top_idx = (
        df_main[df_main['log10p'] >= fdr_line]
        .sort_values('log10p', ascending=False)
        .head(10)
        .index
    )
    df_main.loc[top_idx, 'TopLabel'] = True
    
    return df_main, fdr_line, disease_groups


def prepare_barplot_data(df_compare, alpha, correction_method='fdr_bh'):
    """
    Prepare data for disease effect bar plot.
    
    Parameters:
    -----------
    df_compare : DataFrame
        Comparison dataframe with columns: glycan, param, est_fixed, pval_fixed_adj
    alpha : float
        Significance threshold
    correction_method : str
        Correction method for p-values ('fdr_bh' or 'fdr_by')

    Returns:
    --------
    df_bar : DataFrame
        Data for bar plot with columns: Glycan, Disease, EffectSize, p_value,
        Significant, CI_lower, CI_upper
    disease_groups : list
        Detected disease groups
    """
    df_main = df_compare[
        df_compare['param'].str.startswith("C(DISEASE") &
        ~df_compare['param'].str.contains(":", na=False)
    ].copy()

    df_main['Disease'] = df_main['param'].apply(
        lambda x: x.split("[T.")[1].rstrip("]") if "[T." in x else None
    )

    if correction_method == 'fdr_bh':
        df_main['pval_fixed_adj'] = df_main['pval_fixed_adj_bh']
    else:
        df_main['pval_fixed_adj'] = df_main['pval_fixed_adj_by']

    df_bar = df_main[[
        'glycan',
        'Disease',
        'est_fixed',
        'pval_fixed_adj',
        'se_fixed'
    ]].rename(columns={
        'glycan': 'Glycan',
        'est_fixed': 'EffectSize',
        'pval_fixed_adj': 'p_value'
    })

    df_bar["Significant"] = df_bar["p_value"] < alpha

    # Confidence intervals for fixed effects
    df_bar['CI_lower'] = df_bar['EffectSize'] - 1.96 * df_bar['se_fixed']
    df_bar['CI_upper'] = df_bar['EffectSize'] + 1.96 * df_bar['se_fixed']
    
    # Sort glycans naturally
    glycan_order = sorted(
        df_bar['Glycan'].unique(), 
        key=natural_sort_key
    )
    df_bar['Glycan'] = pd.Categorical(
        df_bar['Glycan'], 
        categories=glycan_order, 
        ordered=True
    )
    df_bar = df_bar.sort_values('Glycan')
    
    # Get unique disease groups (only main effects)
    disease_groups = sorted(df_bar['Disease'].unique())
    
    return df_bar, disease_groups
    
def plot_disease_volcano(df_volcano, fdr_line, disease_colors, ax, legend_order=None):
    """
    Plot volcano plot of disease main effects.
    
    Parameters:
    -----------
    df_volcano : DataFrame
        Volcano data from prepare_volcano_data()
    fdr_line : float
        Y-position for FDR threshold line
    disease_colors : dict
        Mapping of disease -> color
    reference_group : str
        Name of reference group (e.g., 'HC', 'Non-IBD')
    ax : matplotlib axis
        Axis to plot on
    legend_order : list, optional
        Ordered list of disease names for legend display.
        If None, uses sorted disease order.
    
    Returns:
    --------
    ax : matplotlib axis
    """
    disease_groups = sorted(df_volcano['Disease'].unique())
    
    # Plot each disease group
    for disease in disease_groups:
        for is_sig, alpha_val in [(True, 1.0), (False, 0.3)]:
            subset = df_volcano[
                (df_volcano["Disease"] == disease) & 
                (df_volcano["Significant"] == is_sig)
            ]
            
            if len(subset) == 0:
                continue
            
            pts = ax.scatter(
                subset['est_fixed'],
                subset['log10p'],
                marker='o',
                s=60,
                facecolors=disease_colors[disease] if is_sig else 'white',
                edgecolors=disease_colors[disease],
                alpha=alpha_val,
                linewidth=1.5
            )
    
    # Add FDR threshold line and y-axis label
    ax.axhline(fdr_line, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel(r"$-\log_{10}$(FDR-adjusted p-value)")
    
    # Prepare legend
    ordered_diseases = legend_order or disease_groups
    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=disease_colors[d], markeredgecolor=disease_colors[d],
               markersize=8, label=d)
        for d in ordered_diseases
    ]
    handles.append(
        Line2D([0], [0], linestyle='--', color='black', linewidth=1, label='FDR threshold')
    )
    ax.legend(handles=handles, fontsize='small', frameon=False)
    
    # Annotate top 10 points above FDR line
    if 'TopLabel' in df_volcano.columns:
        df_labels = df_volcano[df_volcano['TopLabel']].copy()
        df_labels = df_labels.dropna(subset=['GP_label'])
        
        texts = []
        for _, row in df_labels.iterrows():
            texts.append(
                ax.text(
                    row['est_fixed'],
                    row['log10p'],
                    row['GP_label'],
                    fontsize='x-small',
                    ha='center',
                    va='center',
                )
            )
        
        adjust_text(
            texts,
            x=df_volcano['est_fixed'].values,
            y=df_volcano['log10p'].values,
            arrowprops=dict(
                arrowstyle='-',
                color='grey',
                lw=0.5,
                alpha=0.6,
                shrinkA=5,   
                shrinkB=5,   
            ),
            expand=(1.5, 1.5),
            force_text=(0.8, 0.8),
            force_points=(0.5, 0.5),
            ax=ax,
        )

        ax.grid(alpha=0.3)
    
    return ax


def plot_disease_barplot(
    df_bar,
    disease_colors,
    reference_group,
    ax,
    legend_order=None,
    group_spacing=1.25,
):
    """
    Plot grouped horizontal bars for disease effects by glycan.

    Parameters
    ----------
    df_bar : pandas.DataFrame
        Bar-plot input with columns ``Glycan``, ``Disease``, ``EffectSize``,
        and ``Significant``.
    disease_colors : dict
        Mapping from disease group name to plotting color.
    reference_group : str
        Label used in the x-axis title for the comparison baseline.
    ax : matplotlib.axes.Axes
        Axis on which the grouped bar plot is drawn.
    legend_order : list of str, optional
        Order of disease groups in plotting/legend. If None, uses sorted
        disease names from ``df_bar``.
    group_spacing : float, default 1.25
        Vertical spacing between glycan groups. Values > 1 increase the
        whitespace between glycans so grouped disease bars are easier to
        distinguish.

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the rendered grouped disease-effect bars.
    """
    disease_groups = legend_order or sorted(df_bar['Disease'].unique())
    glycan_order = df_bar['Glycan'].cat.categories.tolist()
    n_glycans = len(glycan_order)
    n_diseases = len(disease_groups)

    bar_height = 0.8 / n_diseases
    base_positions = np.arange(n_glycans, dtype=float) * group_spacing
    glycan_to_y = {g: y for g, y in zip(glycan_order, base_positions)}

    for d_idx, disease in enumerate(disease_groups):
        sub = df_bar[df_bar['Disease'] == disease].copy()
        offset = (d_idx - (n_diseases - 1) / 2) * bar_height
        y_positions = [glycan_to_y[g] + offset for g in sub['Glycan']]

        sig = sub['Significant'].values
        colors = [
            disease_colors[disease] if s else 'white'
            for s in sig
        ]
        edge_colors = [disease_colors[disease]] * len(sub)

        bars = ax.barh(
            y_positions,
            sub['EffectSize'].values,
            height=bar_height * 0.9,
            color=colors,
            edgecolor=edge_colors,
            linewidth=1.2,
            zorder=3,
        )

        # Keep non-significant bars visible with white fill + hatch.
        for bar, s in zip(bars, sig):
            bar.set_alpha(0.7)
            if not s:
                bar.set_hatch('///')
                bar.set_linewidth(1.0)

        handles = [
            Patch(facecolor=disease_colors[d], edgecolor=disease_colors[d], label=d)
            for d in disease_groups
        ]
        ax.legend(handles=handles, fontsize='small', frameon=False, loc='lower right')

    ax.axvline(0, color='black', linestyle='--', linewidth=1, zorder=2)

    ax.set_yticks(base_positions)
    ax.set_yticklabels(glycan_order, fontsize='small')
    ax.invert_yaxis()
    ax.set_xlabel(f"Disease effect vs. {reference_group} (at age 40)")
    ax.set_ylabel("")

    ax.grid(axis='x', alpha=0.3, zorder=1)

    return ax


def analyze_disease_main_effects(df_compare, alpha, reference_group='Reference', correction_method='fdr_bh', disease_colors=None):
    """
    Core function to analyze and prepare disease effect plots.
    
    Parameters:
    -----------
    df_compare : DataFrame
        Comparison dataframe with columns: param, est_fixed, pval_fixed_adj
    alpha : float
        Significance threshold (FDR)
    reference_group : str
        Name of reference group (e.g., 'HC', 'Non-IBD')
    correction_method : str
        Method for multiple testing correction ('fdr_bh')
    disease_colors : dict, optional
        Custom color mapping. If None, uses defaults.
    
    Returns:
    --------
    df_volcano : DataFrame
        Volcano plot data
    df_bar : DataFrame
        Bar plot data
    fdr_line : float
        FDR threshold line position
    disease_colors : dict
        Color mapping used
    reference_group : str
        Reference group name
    """
    # Prepare data
    df_volcano, fdr_line, disease_groups_volcano = prepare_volcano_data(df_compare, alpha, correction_method)
    df_bar, disease_groups_bar = prepare_barplot_data(df_compare, alpha, correction_method)
    
    # Verify consistency
    disease_groups = sorted(set(disease_groups_volcano) | set(disease_groups_bar))
    
    # Print summary
    print("=" * 80)
    print("DISEASE EFFECTS ANALYSIS")
    print("=" * 80)
    print(f"\nReference group: {reference_group}")
    print(f"Comparison groups: {', '.join(disease_groups)}")
    print(f"FDR threshold: {alpha:.4f}")
    print(f"\nColors assigned:")
    for disease, color in disease_colors.items():
        print(f"  {disease}: {color}")
    
    # Summary statistics
    print(f"\nVolcano plot:")
    for disease in disease_groups:
        n_sig = (df_volcano['Disease'] == disease) & df_volcano['Significant']
        print(f"  {disease}: {n_sig.sum()} significant glycans")
    
    print(f"\nBar plot:")
    for disease in disease_groups:
        n_sig = (df_bar['Disease'] == disease) & df_bar['Significant']
        print(f"  {disease}: {n_sig.sum()} significant glycans")
    
    print("=" * 80)
    
    return df_volcano, df_bar, fdr_line, disease_colors, reference_group

def prepare_interaction_data(results_fixed, df_compare, correction_method='fdr_bh'):
    """
    Prepare data for disease interaction plots.
    
    Parameters:
    -----------
    results_fixed : dict
        Dictionary of OLS results {glycan: model}
    df_compare : DataFrame
        Comparison dataframe with columns: glycan, param, sig_fixed, pval_fixed_adj
    correction_method : str
        Method for multiple testing correction ('fdr_bh')

    Returns:
    --------
    df_int : DataFrame
        Interaction data with columns: Glycan, Disease, Interaction, Estimate, 
        SE, P_value, Significant, CI_lower, CI_upper
    disease_groups : list
        Detected disease groups
    """
    interaction_data = []
    
    for glycan, ols in results_fixed.items():
        for param in ols.params.index:
            # Only interaction terms
            if param.startswith("C(DISEASE") and ":" in param:
                # Get significance from df_compare
                sig_mask = (df_compare['glycan'] == glycan) & (df_compare['param'] == param)
                if sig_mask.any():
                    if correction_method == 'fdr_bh':
                        sig_fixed_col = 'sig_fixed_bh'
                        pval_adj_col = 'pval_fixed_adj_bh'
                    elif correction_method == 'fdr_by':
                        sig_fixed_col = 'sig_fixed_by'
                        pval_adj_col = 'pval_fixed_adj_by'
                    sig = df_compare.loc[sig_mask, sig_fixed_col].item()
                    pval = df_compare.loc[sig_mask, pval_adj_col].item()
                else:
                    sig = False
                    pval = 1.0
                
                # Parse interaction type and disease
                if ":Age" in param:
                    interaction_type = "Age"
                elif ":C(Sex" in param:
                    interaction_type = "Sex"
                else:
                    continue
                
                # Extract disease name
                disease = param.split("[T.")[1].split("]")[0]
                
                interaction_data.append({
                    'Glycan': glycan,
                    'Disease': disease,
                    'Interaction': interaction_type,
                    'Parameter': param,
                    'Estimate': ols.params[param],
                    'SE': ols.bse[param],
                    'P_value': pval,
                    'Significant': bool(sig)
                })
    
    df_int = pd.DataFrame(interaction_data)
    
    # Sort glycans naturally
    glycan_order = sorted(df_int['Glycan'].unique(), key=natural_sort_key)
    df_int['Glycan'] = pd.Categorical(
        df_int['Glycan'], 
        categories=glycan_order, 
        ordered=True
    )
    df_int = df_int.sort_values(['Interaction', 'Glycan'])
    
    # Calculate confidence intervals
    df_int['CI_lower'] = df_int['Estimate'] - 1.96 * df_int['SE']
    df_int['CI_upper'] = df_int['Estimate'] + 1.96 * df_int['SE']
    
    # Get unique disease groups
    disease_groups = sorted(df_int['Disease'].unique())
    
    return df_int, disease_groups

def plot_interaction_lollipop(df_int, interaction_type, disease_colors, ax, legend_order=None):
    """
    Plot lollipop plot for a specific interaction type (Age or Sex).
    
    Parameters:
    -----------
    df_int : DataFrame
        Interaction data from prepare_interaction_data()
    interaction_type : str
        'Age' or 'Sex'
    disease_colors : dict
        Mapping of disease -> color
    ax : matplotlib axis
        Axis to plot on
    legend_order : list, optional
        Ordered list of disease names for legend display.
        If None, uses sorted disease order.
    
    Returns:
    --------
    ax : matplotlib axis
    """
    # Filter for this interaction type
    df_subset = df_int[df_int['Interaction'] == interaction_type].copy()
    
    # Get glycan order (reversed so GP1 is at top)
    glycan_order = df_subset['Glycan'].cat.categories.tolist()
    glycan_order_reversed = list(reversed(glycan_order))
    glycan_positions = {g: i for i, g in enumerate(glycan_order_reversed)}
    
    disease_groups = sorted(df_subset['Disease'].unique())
    n_diseases = len(disease_groups)
    
    # Calculate offsets for multiple diseases
    if n_diseases == 1:
        offsets = {disease_groups[0]: 0}
    elif n_diseases == 2:
        offsets = {disease_groups[0]: 0.15, disease_groups[1]: -0.15}
    else:
        # For 3+ diseases, spread them out
        offset_range = 0.3
        offset_step = offset_range / (n_diseases - 1) if n_diseases > 1 else 0
        offsets = {
            disease: offset_range/2 - i*offset_step 
            for i, disease in enumerate(disease_groups)
        }
    
    legend_entries = {}
    
    for disease in disease_groups:
        df_disease = df_subset[df_subset['Disease'] == disease]
        offset = offsets[disease]
        
        # Separate significant and non-significant
        sig_mask = df_disease['Significant']
        
        # Plot non-significant effects first
        if (~sig_mask).any():
            df_nonsig = df_disease[~sig_mask]
            y_nonsig = [glycan_positions[g] + offset for g in df_nonsig['Glycan']]
            
            # Error bars (very light and thin)
            ax.hlines(y_nonsig, df_nonsig['CI_lower'], df_nonsig['CI_upper'],
                     color=disease_colors[disease], linewidth=0.8, alpha=0.15, zorder=1)
            # Lollipop heads (very faint hollow)
            ax.scatter(df_nonsig['Estimate'], y_nonsig,
                      facecolors='white', edgecolors=disease_colors[disease],
                      s=60, zorder=1, linewidth=1, alpha=0.2)
        
        # Plot significant effects
        if sig_mask.any():
            df_sig = df_disease[sig_mask]
            y_sig = [glycan_positions[g] + offset for g in df_sig['Glycan']]
            
            # Error bars (bold and prominent)
            ax.hlines(y_sig, df_sig['CI_lower'], df_sig['CI_upper'],
                     color=disease_colors[disease], linewidth=2.5, alpha=0.9, zorder=3)
            # Lollipop heads (solid and prominent)
            ax.scatter(df_sig['Estimate'], y_sig,
                      color=disease_colors[disease], s=100, zorder=4,
                      edgecolors='white', linewidth=2)
            
            # Add to legend dictionary
            legend_entries[disease] = Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=disease_colors[disease], markersize=8,
                                            markeredgecolor='white', markeredgewidth=1.5,
                                            label=f'{disease}')
    
    # Styling
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5, zorder=2)
    
    # Set y-ticks and labels (GP1 at top)
    ax.set_yticks(range(len(glycan_order)))
    ax.set_yticklabels(glycan_order_reversed)
    
    ax.set_xlabel('Change in disease effect per year of age (β)')
    ax.set_title(f'Disease × {interaction_type} Interaction')
    ax.grid(axis='x', alpha=0.3, zorder=0)
    
    # Order legend entries
    if legend_entries:
        if legend_order is not None:
            ordered_handles = [legend_entries[disease] for disease in legend_order 
                              if disease in legend_entries]
        else:
            ordered_handles = list(legend_entries.values())
        
        ax.legend(handles=ordered_handles, fontsize='small', 
                 loc='lower right', frameon=False)
    
    return ax


def analyze_disease_interactions(results_fixed, df_compare, alpha,
                                 reference_group='Reference',
                                 correction_method='fdr_bh',
                                 disease_colors=None):
    """
    Core function to analyze and prepare disease interaction plots.
    
    Parameters:
    -----------
    results_fixed : dict
        Dictionary of OLS results {glycan: model}
    df_compare : DataFrame
        Comparison dataframe with columns: glycan, param, sig_fixed, pval_fixed_adj
    alpha : float
        Significance threshold
    reference_group : str
        Name of reference group (for reporting)
    correction_method : str
        Method for multiple testing correction ('fdr_bh' or 'fdr_by')
    disease_colors : dict, optional
        Custom color mapping. If None, uses defaults.
    
    Returns:
    --------
    df_int : DataFrame
        Interaction data
    disease_colors : dict
        Color mapping used
    reference_group : str
        Reference group name
    """
    # Prepare data
    df_int, disease_groups = prepare_interaction_data(results_fixed, df_compare, correction_method)
    
    # Print summary
    print("=" * 80)
    print("DISEASE INTERACTIONS ANALYSIS")
    print("=" * 80)
    print(f"\nReference group: {reference_group}")
    print(f"Disease groups: {', '.join(disease_groups)}")
    print(f"FDR threshold: {alpha:.4f}")
    print(f"\nColors assigned:")
    for disease, color in disease_colors.items():
        if disease in disease_groups:
            print(f"  {disease}: {color}")
    
    # Summary statistics
    for interaction_type in df_int['Interaction'].unique():
        print(f"\n{interaction_type} Interactions:")
        for disease in disease_groups:
            df_subset = df_int[
                (df_int['Interaction'] == interaction_type) & 
                (df_int['Disease'] == disease)
            ]
            n_sig = df_subset['Significant'].sum()
            n_total = len(df_subset)
            print(f"  {disease}: {n_sig}/{n_total} significant glycans")
            
            if n_sig > 0:
                top_effects = df_subset[df_subset['Significant']].nlargest(
                    min(3, n_sig), 'Estimate', keep='all'
                )
                if len(top_effects) > 0:
                    print(f"    Top positive effects:")
                    for _, row in top_effects.iterrows():
                        print(f"      {row['Glycan']}: β={row['Estimate']:.4f}, "
                              f"p={row['P_value']:.2e}")
    
    print("=" * 80)
    
    return df_int