# -*- coding: utf-8 -*-
"""
@author: Kostis Flevaris
"""

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def calculate_age_acceleration(df, glycan_age_col, chron_age_col, out_col, inplace=False):
    """
    Calculate glycan age acceleration as the difference between glycan age and chronological age.
    """
    if glycan_age_col not in df.columns:
        raise KeyError(f"'{glycan_age_col}' not found in df.columns")
    if chron_age_col not in df.columns:
        raise KeyError(f"'{chron_age_col}' not found in df.columns")

    out_df = df if inplace else df.copy()
    out_df[out_col] = out_df[glycan_age_col] - out_df[chron_age_col]
    return out_df

def map_padj_to_confidence(p_adj):
    """
    Map adjusted p-values to confidence thresholds:
      *   : p_adj < 0.10  (90%)
      **  : p_adj < 0.05  (95%)
      *** : p_adj < 0.01  (99%)
      ns  : otherwise
      na  : missing / not testable
    """
    if np.isnan(p_adj):
        return "na"
    if p_adj < 0.01:
        return "***"
    if p_adj < 0.05:
        return "**"
    if p_adj < 0.10:
        return "*"
    return "ns"

def compute_fdr_adjusted_pvals(df, disease_col, value_col, comparisons, fdr_method="fdr_bh", alternative="two-sided"):
    """
    Compute raw p-values for predefined comparisons and apply BH-FDR correction
    within the provided dataframe.

    Returns:
      dict mapping (g1, g2) -> adjusted p-value
      Only includes comparisons with enough data (>=2 per group).
    """
    raw_pvals = []
    valid_comps = []

    for g1, g2 in comparisons:
        x = df.loc[df[disease_col] == g1, value_col].dropna().values
        y = df.loc[df[disease_col] == g2, value_col].dropna().values

        if len(x) < 2 or len(y) < 2:
            continue

        p = mannwhitneyu(x, y, alternative=alternative, method="asymptotic").pvalue
        raw_pvals.append(float(p))
        valid_comps.append((g1, g2))

    if not raw_pvals:
        return {}

    _, p_adj, _, _ = multipletests(raw_pvals, method=fdr_method)
    return {comp: float(p_adj[i]) for i, comp in enumerate(valid_comps)}

def plot_violin_with_points(
    plot_df,
    disease_col,
    value_col,
    disease_order,
    custom_colors,
    ax,
    title=None,
    xlabel=None,
    ylabel=None,
    ylim=None,
    seed=0
):
    """
    Plot violin distributions with jittered points and median labels.

    Parameters
    ----------
    plot_df : pandas.DataFrame
        Input data for plotting.
    disease_col : str
        Column name with disease/group labels.
    value_col : str
        Column name with numeric values to visualize.
    disease_order : list of str
        Group display order on the x-axis.
    custom_colors : dict
        Mapping from group name to color.
    ax : matplotlib.axes.Axes
        Axis to draw on.
    title, xlabel, ylabel : str, optional
        Text labels for plot title and axes.
    ylim : tuple, optional
        Y-axis limits.
    seed : int, default 0
        Random seed for jitter placement.

    Returns
    -------
    None
    """
    if plot_df.empty:
        ax.set_axis_off()
        return

    palette = [custom_colors.get(group, '#4c72b0') for group in disease_order]

    violin_data = [
        plot_df.loc[plot_df[disease_col] == group, value_col].dropna().values
        for group in disease_order
    ]

    filtered = [(g, v, c) for g, v, c in zip(disease_order, violin_data, palette) if len(v) > 0]
    if not filtered:
        ax.set_axis_off()
        return

    disease_order_f, violin_data_f, palette_f = zip(*filtered)
    positions = np.arange(1, len(disease_order_f) + 1)

    parts = ax.violinplot(
        violin_data_f,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=True,
        widths=0.85
    )

    for body, color in zip(parts['bodies'], palette_f):
        body.set_facecolor(color)
        body.set_edgecolor('black')
        body.set_alpha(0.75)

    for element in ['cbars', 'cmins', 'cmaxes']:
        parts[element].set_color('black')
        parts[element].set_linewidth(1)

    rng = np.random.default_rng(seed)
    for pos, values, color in zip(positions, violin_data_f, palette_f):
        jitter = rng.normal(loc=pos, scale=0.04, size=len(values))
        ax.scatter(
            jitter, values,
            color=color, edgecolor='black', linewidth=0.4, alpha=0.6, s=18
        )

        # Median annotation
        median_val = float(np.median(values))
        ax.text(
            pos + 0.18,
            median_val,
            f"{median_val:.1f}",
            ha='left',
            va='center',
            fontsize=9,
            fontweight='bold',
            color='black',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white',
                edgecolor='black',
                linewidth=0.6,
                alpha=0.8
            )
        )

    # Add sample sizes under tick labels
    xticklabels = []
    for group in disease_order_f:
        n = int((plot_df[disease_col] == group).sum())
        xticklabels.append(f"{group}\n(n={n})")

    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(0.5, len(disease_order_f) + 0.5)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title, pad=12)

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        vmin = plot_df[value_col].min()
        vmax = plot_df[value_col].max()
        ypad = 0.05 * (vmax - vmin) if vmax > vmin else 1.0
        ax.set_ylim(vmin - ypad, vmax + ypad)

def plot_age_acceleration_by_cohort(
    df,
    axes,
    cohort_col="Cohort",
    disease_col="DISEASE",
    glycan_age_col="GlycanAge",
    chron_age_col="Age",
    age_acc_col="AgeAcceleration",
    disease_colors=None,
    cohort_order=None,
    fdr_method="fdr_bh",
    alternative="two-sided",
    show_ns=True,
    show_na=True,
    seed=0,
    add_figure_note=True,
 ):
    """
    Plot cohort-wise age-acceleration violins for multiclass and binary disease views.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table with cohort, disease, glycan age, and chronological age.
    axes : array-like of matplotlib.axes.Axes
        Pre-created axes in row-major order with two columns per cohort.
    cohort_col, disease_col, glycan_age_col, chron_age_col, age_acc_col : str, optional
        Column names used for cohort grouping and age-acceleration computation.
    disease_colors : dict, optional
        Mapping from disease label to color.
    cohort_order : list of str, optional
        Preferred cohort display order.
    fdr_method : str, default "fdr_bh"
        Multiple-testing correction method for pairwise comparisons.
    alternative : str, default "two-sided"
        Alternative hypothesis used in Mann-Whitney U tests.
    show_ns : bool, default True
        Whether to include non-significant labels in subplot titles.
    show_na : bool, default True
        Whether to include not-testable labels in subplot titles.
    seed : int, default 0
        Base seed for jitter reproducibility.
    add_figure_note : bool, default True
        Whether to add a confidence legend note under the figure.

    Returns
    -------
    None
    """
    df_delta = calculate_age_acceleration(
        df,
        glycan_age_col=glycan_age_col,
        chron_age_col=chron_age_col,
        out_col=age_acc_col,
        inplace=False
    )

    base_cols = [cohort_col, disease_col, age_acc_col]
    missing = [c for c in base_cols if c not in df_delta.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    base_df = df_delta[base_cols].dropna(subset=[age_acc_col]).copy()
    if base_df.empty:
        raise ValueError(f"No Age Acceleration values available in '{age_acc_col}'.")

    cohorts = list(pd.unique(base_df[cohort_col]))
    if cohort_order is not None:
        ordered = [c for c in cohort_order if c in cohorts]
        remaining = [c for c in cohorts if c not in ordered]
        cohorts = ordered + remaining
    if len(cohorts) == 0:
        raise ValueError(f"No cohorts found in '{cohort_col}'.")

    nrows = len(cohorts)
    expected_axes = nrows * 2
    axes = np.atleast_1d(axes).flatten()

    if len(axes) < expected_axes:
        raise ValueError(
            f"Not enough axes provided. Need {expected_axes} (n_cohorts={nrows} × 2), got {len(axes)}."
        )

    global_min = float(base_df[age_acc_col].min())
    global_max = float(base_df[age_acc_col].max())
    m = max(abs(global_min), abs(global_max))
    ylim = (-m, m)

    multiclass_order_default = ["HC", "SC", "CD", "UC"]
    multiclass_colors = disease_colors

    binary_mapping = {"HC": "Non-IBD", "SC": "Non-IBD", "CD": "IBD", "UC": "IBD"}
    binary_order = ["Non-IBD", "IBD"]
    binary_colors = {
        "Non-IBD": multiclass_colors.get("HC", "#4c72b0"),
        "IBD": multiclass_colors.get("CD", "#dd8452")
    }

    multiclass_comparisons = [("SC", "HC"), ("CD", "HC"), ("UC", "HC")]
    binary_comparisons = [("IBD", "Non-IBD")]

    def format_label(case, ctrl, p_adj):
        stars = map_padj_to_confidence(p_adj)
        if (stars == "ns") and (not show_ns):
            return ""
        if (stars == "NA") and (not show_na):
            return ""
        return f"{case} vs {ctrl}: {stars}"

    for r, cohort in enumerate(cohorts):
        cohort_df = base_df[base_df[cohort_col] == cohort].copy()

        ax_mult = axes[r * 2 + 0]
        ax_bin = axes[r * 2 + 1]

        present = set(cohort_df[disease_col].unique())
        disease_order = [g for g in multiclass_order_default if g in present]

        p_adj_map_multi = compute_fdr_adjusted_pvals(
            df=cohort_df,
            disease_col=disease_col,
            value_col=age_acc_col,
            comparisons=multiclass_comparisons,
            fdr_method=fdr_method,
            alternative=alternative
        )

        title_bits = []
        for case, ctrl in multiclass_comparisons:
            if case in present and ctrl in present:
                p_adj = p_adj_map_multi.get((case, ctrl), np.nan)
            else:
                p_adj = np.nan
            bit = format_label(case, ctrl, p_adj)
            if bit:
                title_bits.append(bit)

        suffix_mult = " | ".join(title_bits) if title_bits else "No valid comparisons"
        title_mult = f"{cohort} | {suffix_mult}"

        plot_violin_with_points(
            plot_df=cohort_df,
            disease_col=disease_col,
            value_col=age_acc_col,
            disease_order=disease_order,
            custom_colors=multiclass_colors,
            ax=ax_mult,
            title=title_mult,
            xlabel="Disease group",
            ylabel="BAA (years)" if r == 0 else None,
            ylim=ylim,
            seed=seed + r * 10
        )

        tmp = cohort_df[cohort_df[disease_col].isin(binary_mapping.keys())].copy()
        tmp[disease_col] = tmp[disease_col].map(binary_mapping)

        p_adj_map_bin = compute_fdr_adjusted_pvals(
            df=tmp,
            disease_col=disease_col,
            value_col=age_acc_col,
            comparisons=binary_comparisons,
            fdr_method=fdr_method,
            alternative=alternative
        )
        p_adj_bin = p_adj_map_bin.get(("IBD", "Non-IBD"), np.nan)
        bit_bin = format_label("IBD", "Non-IBD", p_adj_bin)
        suffix_bin = bit_bin if bit_bin else "IBD vs Non-IBD: (ns suppressed)"
        title_bin = f"{cohort} | {suffix_bin}"

        plot_violin_with_points(
            plot_df=tmp,
            disease_col=disease_col,
            value_col=age_acc_col,
            disease_order=binary_order,
            custom_colors=binary_colors,
            ax=ax_bin,
            title=title_bin,
            xlabel="Disease group (binary)",
            ylabel="BAA (years)" if r == 0 else None,
            ylim=ylim,
            seed=seed + r * 10 + 1
        )

    if add_figure_note:
        fig = axes[0].figure
        fig.text(
            0.5, 0.01,
            "FDR-adjusted p < 0.10, ** p < 0.05, *** p < 0.01 (BH correction)",
            ha="center", va="bottom", fontsize=10
        )