"""Reusable plotting components for global glycome similarity analyses."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ibd_biom_glycoda.analysis.similarity import BetaDiversityResults
from ibd_biom_glycoda.plotting.style import create_figure_with_panels

__all__ = ["plot_cohort_ordination_grid"]


def _plot_ordination_panel(
    axis: Axes,
    coordinates: pd.DataFrame,
    *,
    cluster_col: str,
    cluster_order: Sequence[str],
    colors: Mapping[str, object],
    title: str,
) -> None:
    """Plot one cohort's ordination coordinates."""
    variance_1 = coordinates["Explained_Var1"].iloc[0]
    variance_2 = coordinates["Explained_Var2"].iloc[0]
    clusters = [
        cluster
        for cluster in cluster_order
        if cluster in coordinates[cluster_col].values
    ]
    for cluster in clusters:
        cluster_data = coordinates[coordinates[cluster_col] == cluster]
        axis.scatter(
            cluster_data["Dim1"],
            cluster_data["Dim2"],
            c=[colors[cluster]],
            label=cluster,
            s=50,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    if np.isfinite([variance_1, variance_2]).all():
        axis.set_xlabel(
            f"Dim1 ({variance_1 * 100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
        axis.set_ylabel(
            f"Dim2 ({variance_2 * 100:.1f}%)",
            fontsize=12,
            fontweight="bold",
        )
    else:
        axis.set_xlabel("Dim1", fontsize=12, fontweight="bold")
        axis.set_ylabel("Dim2", fontsize=12, fontweight="bold")

    axis.set_title(title, fontsize=13, fontweight="bold", pad=10)
    axis.axhline(0, linestyle="--", color="gray", lw=0.5, alpha=0.5)
    axis.axvline(0, linestyle="--", color="gray", lw=0.5, alpha=0.5)
    axis.grid(True, alpha=0.2, linestyle=":", linewidth=0.5)
    axis.tick_params(labelsize=10)


def plot_cohort_ordination_grid(
    results: BetaDiversityResults,
    *,
    cluster_col: str,
    cluster_order: Sequence[str],
    colors: Mapping[str, object],
    cohort_order: Sequence[str],
    legend_title: str,
    figsize: tuple[float, float] = (7.0, 7.0),
) -> Figure:
    """Plot cohort-specific ordinations in a consistent four-panel layout.

    Parameters
    ----------
    results : dict
        Cohort-keyed ordination results produced by
        :func:`generate_beta_diversity_pca_results`.
    cluster_col : str
        Ordination column defining point labels and colors.
    cluster_order : sequence of str
        Display order for the cluster levels.
    colors : mapping
        Mapping from cluster labels to Matplotlib-compatible colors.
    cohort_order : sequence of str
        Display order for cohorts. Every named cohort must be present.
    legend_title : str
        Title shown above the shared legend.
    figsize : tuple of float, default=(7, 7)
        Figure dimensions in inches.

    Returns
    -------
    matplotlib.figure.Figure
        Completed ordination grid. Saving remains the caller's responsibility.
    """
    cohorts = list(cohort_order)
    missing = [cohort for cohort in cohorts if cohort not in results]
    if missing:
        raise KeyError(f"ordination results are missing cohorts: {missing}")
    if len(cohorts) != 4:
        raise ValueError("the similarity ordination grid requires exactly four cohorts")

    figure, axes = create_figure_with_panels(figsize=figsize, layout=(2, 2))
    figure.subplots_adjust(wspace=0.35, hspace=0.45, right=0.84)
    for axis, cohort in zip(axes, cohorts):
        _plot_ordination_panel(
            axis,
            results[cohort][0],
            cluster_col=cluster_col,
            cluster_order=cluster_order,
            colors=colors,
            title=cohort,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
    )
    return figure
