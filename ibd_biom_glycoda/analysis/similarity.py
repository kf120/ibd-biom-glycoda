# -*- coding: utf-8 -*-
"""
@author: Kostis Flevaris
"""

import pandas as pd
import numpy as np
import glycowork.glycan_data.stats as glwstats
from collections import OrderedDict

from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

def compute_dunn_index(distance_matrix, labels):
    """
    Compute the Dunn index for cluster validation.

    The Dunn index is defined as the ratio of the minimum inter-cluster
    separation to the maximum intra-cluster diameter. Higher values indicate
    better separated and tighter clusters.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Square (N, N) distance matrix between points.
    labels : array-like
        Cluster labels for each of the N points.

    Returns
    -------
    float
        Dunn index value. Returns ``np.nan`` if computation is not possible
        (for example, fewer than two clusters).
    """
    unique = np.unique(labels)
    intra, inter = [], []

    for i, lab_i in enumerate(unique):
        idx_i = np.where(labels == lab_i)[0]
        # Compute intra-cluster diameter
        if idx_i.size > 1:
            intra.append(distance_matrix[np.ix_(idx_i, idx_i)].max())
        else:
            intra.append(0.0)

        # Compute pairwise inter-cluster separations
        for lab_j in unique[i + 1:]:
            idx_j = np.where(labels == lab_j)[0]
            inter.append(distance_matrix[np.ix_(idx_i, idx_j)].min())

    if not intra or not inter:
        return np.nan

    max_diam = max(intra)
    min_sep = min(inter)
    return np.inf if max_diam == 0 else min_sep / max_diam

def generate_beta_diversity_pca_results(
    df, 
    feature_cols, 
    stratify_col='Cohort', 
    cluster_col='DISEASE', 
    cohort_order=None,
):
    """Compute classical MDS coordinates and clustering metrics per stratum.

    This function performs classical multidimensional scaling (equivalent to
    PCA on a double-centered distance matrix) on the Euclidean distances of
    the provided feature columns. It returns PCA-like coordinates and
    clustering metrics (Dunn index and silhouette) for each stratum of the
    supplied ``stratify_col``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing feature columns and grouping variables.
    feature_cols : list of str
        Names of columns in ``df`` to use for distance computation.
    stratify_col : str, default 'Cohort'
        Column name used to split the dataframe into strata.
    cluster_col : str, default 'DISEASE'
        Column name used for labeling/clustering within each stratum.
    cohort_order : list of str, optional
        If provided, order the returned results (and downstream printing)
        according to this sequence of strata names.

    Returns
    -------
    dict
        Mapping from stratum name -> tuple of
        (pca_df, metrics_df). ``pca_df`` contains columns ``Dim1``, ``Dim2``,
        the cluster label column, and explained variance for the first two
        dimensions. ``metrics_df`` contains simple clustering metrics for the
        stratum.
    """
    results = {}

    for stratum_name, stratum_df in df.groupby(stratify_col):

    # Compute Euclidean distance on provided features
        X = stratum_df[feature_cols].values
        D = squareform(pdist(X, metric='euclidean'))

    # Perform classical MDS using double centering
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H.dot(D ** 2).dot(H)

        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(B)

        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

        # Get coordinates for first two principal components
        coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
        explained = eigvals / eigvals.sum()

        # Create PCA DataFrame
        pca_df = pd.DataFrame(coords, columns=['Dim1', 'Dim2'])
        pca_df[cluster_col] = stratum_df[cluster_col].values
        pca_df['Explained_Var1'] = explained[0]
        pca_df['Explained_Var2'] = explained[1]

        # Relabel disease categories for display consistency
        if cluster_col == 'DISEASE':
            if pca_df[cluster_col].nunique() == 2:
                pca_df[cluster_col] = pca_df[cluster_col].replace({'Control': 'Non-IBD', 'Case': 'IBD'})
            else:
                categories = ['HC', 'SC', 'CD', 'UC']
                pca_df[cluster_col] = pd.Categorical(
                    pca_df[cluster_col], categories=categories, ordered=True
                )

    # Compute clustering metrics for the stratum
        silhouette = silhouette_score(D, stratum_df[cluster_col], metric='precomputed')
        dunn = compute_dunn_index(D, stratum_df[cluster_col].values)

        metrics_df = pd.DataFrame({
            stratify_col: [stratum_name],
            "Dunn index": [round(dunn, 3)],
            "Mean silhouette width": [round(silhouette, 3)]
        })

        results[stratum_name] = (pca_df, metrics_df)

    if cohort_order is not None:
        ordered = OrderedDict()
        for name in cohort_order:
            if name in results:
                ordered[name] = results[name]
        for name, value in results.items():
            if name not in ordered:
                ordered[name] = value
        return ordered

    return results

def plot_beta_diversity_pca_panel(ax, pca_df, cluster_col, cluster_order, colors, title=None):
    """
    Plot a single PCA panel on a given axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    pca_df : pd.DataFrame
        DataFrame with columns: Dim1, Dim2, cluster_col, Explained_Var1, Explained_Var2.
    cluster_col : str
        Column name for clustering/coloring (e.g., 'DISEASE').
    colors : dict
        Dictionary mapping cluster values to RGB tuples.
        Example: {'HC': (0.0, 0.6, 0.5), 'CD': (0.9, 0.6, 0.0)}
    title : str, optional
        Panel title (e.g., 'IT').
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The modified axis.
    """
    
    # Read explained variance for axis labels
    var1 = pca_df['Explained_Var1'].iloc[0] if 'Explained_Var1' in pca_df.columns else np.nan
    var2 = pca_df['Explained_Var2'].iloc[0] if 'Explained_Var2' in pca_df.columns else np.nan

    # Determine clusters to plot (respect provided order if any)
    if cluster_order is None:
        clusters_to_plot = pca_df[cluster_col].unique()
    else:
        clusters_to_plot = [c for c in cluster_order if c in pca_df[cluster_col].values]

    # Draw points per cluster
    for cluster in clusters_to_plot:
        cluster_data = pca_df[pca_df[cluster_col] == cluster]
        ax.scatter(
            cluster_data['Dim1'],
            cluster_data['Dim2'],
            c=[colors[cluster]],
            label=cluster,
            s=50,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5
        )
    
    # Axis labels with variance
    if not np.isnan(var1) and not np.isnan(var2):
        ax.set_xlabel(f"Dim1 ({var1 * 100:.1f}%)", fontsize=12, fontweight='bold')
        ax.set_ylabel(f"Dim2 ({var2 * 100:.1f}%)", fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel("Dim1", fontsize=12, fontweight='bold')
        ax.set_ylabel("Dim2", fontsize=12, fontweight='bold')
    
    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    # Reference lines
    ax.axhline(0, linestyle='--', color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, linestyle='--', color='gray', lw=0.5, alpha=0.5)
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Ticks
    ax.tick_params(labelsize=10)
    
    return ax