"""
@author: Kostis Flevaris
"""

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score

from ibd_biom_glycoda.data.transformations import (
    apply_clr_transformation,
    residualize_coda,
)

BetaDiversityResults = dict[str, tuple[pd.DataFrame, pd.DataFrame]]


@dataclass(frozen=True)
class SimilarityRepresentations:
    """Validated representations used by the global similarity analysis.

    Attributes
    ----------
    closed_percentages : pandas.DataFrame
        Metadata and strictly positive glycan percentages closed to the
        requested total.
    clr : pandas.DataFrame
        Metadata and deterministic CLR coordinates. Each feature row sums to
        zero within numerical tolerance.
    percentage_residualized : pandas.DataFrame
        Closed-percentage features residualized on the declared covariates.
    clr_residualized : pandas.DataFrame
        CLR coordinates residualized on the same declared covariates.
    contract_summary : pandas.DataFrame
        Compact validation record for the analyzed samples and features.
    """

    closed_percentages: pd.DataFrame
    clr: pd.DataFrame
    percentage_residualized: pd.DataFrame
    clr_residualized: pd.DataFrame
    contract_summary: pd.DataFrame


@dataclass(frozen=True)
class GeometryComparison:
    """Matched percentage and CLR ordination results for one endpoint."""

    metrics: pd.DataFrame
    percentage_ordination: BetaDiversityResults
    clr_ordination: BetaDiversityResults


def _validate_similarity_frame(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    covariates: Sequence[str],
    *,
    closure_total: float,
    closure_atol: float,
) -> pd.DataFrame:
    """Validate sample identity, metadata, and closed-composition semantics."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not frame.index.is_unique:
        raise ValueError("frame must have a unique sample index")
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    if not covariates:
        raise ValueError("residualization covariates cannot be empty")
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("feature_cols must contain unique feature names")
    if closure_total <= 0:
        raise ValueError("closure_total must be positive")
    if closure_atol <= 0:
        raise ValueError("closure_atol must be positive")

    required = [*feature_cols, *covariates]
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        raise KeyError(f"frame is missing columns: {missing_columns}")
    if frame[list(covariates)].isna().any().any():
        raise ValueError("residualization covariates cannot contain missing values")

    values = frame.loc[:, feature_cols].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("compositional features must contain only finite values")
    if not (values > 0).all().all():
        raise ValueError("deterministic CLR requires strictly positive features")

    row_sums = values.sum(axis=1)
    if not np.allclose(row_sums, closure_total, rtol=0.0, atol=closure_atol):
        maximum_error = float(np.abs(row_sums - closure_total).max())
        raise ValueError(
            f"compositional rows must sum to {closure_total}; "
            f"maximum absolute error is {maximum_error:.3g}"
        )
    return values


def prepare_similarity_representations(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    residualization_covariates: Sequence[str],
    closure_total: float = 100.0,
    closure_atol: float = 1e-8,
    clr_atol: float = 1e-10,
    random_state: int = 42,
) -> SimilarityRepresentations:
    """Prepare validated percentage and deterministic CLR representations.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per uniquely indexed sample, containing metadata and positive
        compositional features already closed to ``closure_total``.
    feature_cols : sequence of str
        Ordered identities of the compositional glycan features.
    residualization_covariates : sequence of str
        Metadata columns removed from both representations before descriptive
        cohort-specific ordination.
    closure_total : float, default=100
        Required row total for the percentage representation.
    closure_atol : float, default=1e-8
        Absolute tolerance for the closure invariant.
    clr_atol : float, default=1e-10
        Absolute tolerance for the CLR zero-sum invariant.
    random_state : int, default=42
        Explicit random-state contract. Deterministic CLR does not draw noise.

    Returns
    -------
    SimilarityRepresentations
        Validated original, CLR, residualized, and audit representations with
        sample and feature identities preserved.
    """
    features = list(feature_cols)
    covariates = list(residualization_covariates)
    if clr_atol <= 0:
        raise ValueError("clr_atol must be positive")
    values = _validate_similarity_frame(
        frame,
        features,
        covariates,
        closure_total=closure_total,
        closure_atol=closure_atol,
    )

    clr_values = apply_clr_transformation(
        values,
        group1=frame.index.tolist(),
        group2=[],
        gamma=0.0,
        custom_scale=None,
        random_state=random_state,
    )
    clr_row_sums = clr_values.sum(axis=1)
    if not np.allclose(clr_row_sums, 0.0, rtol=0.0, atol=clr_atol):
        maximum_error = float(np.abs(clr_row_sums).max())
        raise ValueError(
            "CLR coordinates must sum to zero within each sample; "
            f"maximum absolute error is {maximum_error:.3g}"
        )

    closed_frame = frame.copy()
    closed_frame[features] = values
    clr_frame = frame.copy()
    clr_frame[features] = clr_values
    percentage_residualized = frame.copy()
    percentage_residualized[features] = residualize_coda(
        values,
        frame[covariates],
        covariates=covariates,
    )
    clr_residualized = clr_frame.copy()
    clr_residualized[features] = residualize_coda(
        clr_values,
        frame[covariates],
        covariates=covariates,
    )

    row_sums = values.sum(axis=1)
    contract_summary = pd.DataFrame(
        {
            "Check": [
                "Analyzed samples",
                "Analyzed glycan peaks",
                "Minimum peak percentage",
                "Maximum absolute closure error",
                "Maximum absolute CLR row sum",
            ],
            "Value": [
                len(values),
                len(features),
                values.min().min(),
                np.abs(row_sums - closure_total).max(),
                np.abs(clr_row_sums).max(),
            ],
        }
    )
    return SimilarityRepresentations(
        closed_percentages=closed_frame,
        clr=clr_frame,
        percentage_residualized=percentage_residualized,
        clr_residualized=clr_residualized,
        contract_summary=contract_summary,
    )


def _aligned_inputs(
    coordinates: pd.DataFrame | np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Validate and align multivariate coordinates with sample metadata."""
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")

    if isinstance(coordinates, pd.DataFrame):
        if not coordinates.index.is_unique or not metadata.index.is_unique:
            raise ValueError("coordinates and metadata must have unique sample indices")
        missing = coordinates.index.difference(metadata.index)
        if not missing.empty:
            raise ValueError(f"metadata is missing {len(missing)} coordinate samples")
        aligned_metadata = metadata.loc[coordinates.index].copy()
        values = coordinates.to_numpy(dtype=float)
    else:
        values = np.asarray(coordinates, dtype=float)
        aligned_metadata = metadata.copy()

    if values.ndim != 2:
        raise ValueError(
            f"coordinates must be two-dimensional, got shape {values.shape}"
        )
    if values.shape[0] != len(aligned_metadata):
        raise ValueError(
            "coordinates and metadata must contain the same number of samples: "
            f"{values.shape[0]} != {len(aligned_metadata)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("coordinates must contain only finite values")
    return values, aligned_metadata


def _encode_columns(metadata: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Encode numeric and categorical columns without an intercept."""
    missing_columns = [column for column in columns if column not in metadata]
    if missing_columns:
        raise KeyError(f"metadata is missing columns: {missing_columns}")

    encoded = []
    for column in columns:
        series = metadata[column]
        if series.isna().any():
            raise ValueError(f"metadata column {column!r} contains missing values")
        if pd.api.types.is_numeric_dtype(series.dtype):
            encoded.append(series.to_numpy(dtype=float)[:, None])
        else:
            dummies = pd.get_dummies(
                series.astype("category"),
                prefix=column,
                drop_first=True,
                dtype=float,
            )
            encoded.append(dummies.to_numpy(dtype=float))

    if not encoded:
        return np.empty((len(metadata), 0), dtype=float)
    return np.column_stack(encoded)


def _permutation_order(
    block_codes: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return row indices independently permuted within each block."""
    order = np.arange(len(block_codes))
    for block in np.unique(block_codes):
        positions = np.flatnonzero(block_codes == block)
        order[positions] = rng.permutation(positions)
    return order


def _term_statistic(
    response: np.ndarray,
    reduced_basis: np.ndarray,
    term_basis: np.ndarray,
    df_term: int,
    df_residual: int,
) -> tuple[float, float, float]:
    """Calculate a partial multivariate pseudo-F from orthogonal bases."""
    term_ss = float(np.square(term_basis.T @ response).sum())
    fitted_ss = float(np.square(reduced_basis.T @ response).sum()) + term_ss
    residual_ss = max(float(np.square(response).sum()) - fitted_ss, 0.0)
    denominator = residual_ss / df_residual
    pseudo_f = np.inf if denominator == 0.0 else (term_ss / df_term) / denominator
    partial_r2 = term_ss / (term_ss + residual_ss) if term_ss + residual_ss else 0.0
    return pseudo_f, partial_r2, residual_ss


def freedman_lane_permanova(
    coordinates: pd.DataFrame | np.ndarray,
    metadata: pd.DataFrame,
    *,
    term: str,
    covariates: list[str],
    blocks: str,
    n_permutations: int = 999,
    random_state: int | np.random.Generator = 42,
) -> dict[str, float | int]:
    """Test a partial multivariate effect with blocked Freedman--Lane permutations.

    Euclidean coordinates are used directly. Passing deterministic CLR
    coordinates therefore gives the exact Aitchison-distance pseudo-F without
    constructing a large square distance matrix.

    Parameters
    ----------
    coordinates : pandas.DataFrame or numpy.ndarray
        Sample-by-feature Euclidean coordinates.
    metadata : pandas.DataFrame
        Sample metadata containing the tested term, covariates, and blocks.
    term : str
        Metadata column whose adjusted multivariate effect is tested.
    covariates : list of str
        Columns included in both the reduced and full models.
    blocks : str
        Column restricting residual permutations to exchangeable strata.
    n_permutations : int, default=999
        Number of residual permutations.
    random_state : int or numpy.random.Generator, default=42
        Reproducible permutation generator.

    Returns
    -------
    dict
        Pseudo-F, partial R-squared, permutation p-value, degrees of freedom,
        and permutation resolution.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    response, metadata = _aligned_inputs(coordinates, metadata)
    required = [term, blocks, *covariates]
    missing_columns = [column for column in required if column not in metadata]
    if missing_columns:
        raise KeyError(f"metadata is missing columns: {missing_columns}")
    if metadata[blocks].isna().any():
        raise ValueError(f"permutation block {blocks!r} contains missing values")

    reduced_columns = list(dict.fromkeys(covariates))
    reduced = np.column_stack(
        [np.ones(len(metadata)), _encode_columns(metadata, reduced_columns)]
    )
    term_matrix = _encode_columns(metadata, [term])
    full = np.column_stack([reduced, term_matrix])

    rank_reduced = np.linalg.matrix_rank(reduced)
    rank_full = np.linalg.matrix_rank(full)
    if rank_reduced != reduced.shape[1]:
        raise ValueError("reduced design matrix is rank deficient")
    df_term = rank_full - rank_reduced
    df_residual = len(metadata) - rank_full
    if df_term < 1:
        raise ValueError(f"term {term!r} adds no estimable degrees of freedom")
    if df_residual < 1:
        raise ValueError("full model has no residual degrees of freedom")

    reduced_basis = np.linalg.qr(reduced, mode="reduced")[0]
    residualized_term = term_matrix - reduced_basis @ (reduced_basis.T @ term_matrix)
    term_basis = np.linalg.qr(residualized_term, mode="reduced")[0][:, :df_term]

    observed_f, partial_r2, _ = _term_statistic(
        response,
        reduced_basis,
        term_basis,
        df_term,
        df_residual,
    )
    reduced_fitted = reduced_basis @ (reduced_basis.T @ response)
    reduced_residuals = response - reduced_fitted

    block_codes = pd.factorize(metadata[blocks], sort=True)[0]
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    exceedances = 0
    for _ in range(n_permutations):
        order = _permutation_order(block_codes, rng)
        permuted_response = reduced_fitted + reduced_residuals[order]
        permuted_f, _, _ = _term_statistic(
            permuted_response,
            reduced_basis,
            term_basis,
            df_term,
            df_residual,
        )
        exceedances += int(permuted_f >= observed_f - 1e-12)

    return {
        "pseudo_f": observed_f,
        "partial_r2": partial_r2,
        "p_value": (exceedances + 1) / (n_permutations + 1),
        "df_term": df_term,
        "df_residual": df_residual,
        "n_permutations": n_permutations,
        "p_resolution": 1 / (n_permutations + 1),
    }


def _dispersion_f(coordinates: np.ndarray, labels: np.ndarray) -> float:
    """Calculate the centroid-based PERMDISP F statistic."""
    unique_labels = np.unique(labels)
    distances = np.empty(len(labels), dtype=float)
    for label in unique_labels:
        mask = labels == label
        centroid = coordinates[mask].mean(axis=0)
        distances[mask] = np.linalg.norm(coordinates[mask] - centroid, axis=1)

    grand_mean = distances.mean()
    between_ss = sum(
        np.count_nonzero(labels == label)
        * (distances[labels == label].mean() - grand_mean) ** 2
        for label in unique_labels
    )
    within_ss = sum(
        np.square(distances[labels == label] - distances[labels == label].mean()).sum()
        for label in unique_labels
    )
    df_between = len(unique_labels) - 1
    df_within = len(labels) - len(unique_labels)
    denominator = within_ss / df_within
    return np.inf if denominator == 0.0 else (between_ss / df_between) / denominator


def blocked_permdisp(
    coordinates: pd.DataFrame | np.ndarray,
    metadata: pd.DataFrame,
    *,
    group: str,
    blocks: str,
    n_permutations: int = 999,
    random_state: int | np.random.Generator = 42,
) -> dict[str, float | int]:
    """Test centroid dispersion differences with labels permuted within blocks.

    Parameters
    ----------
    coordinates : pandas.DataFrame or numpy.ndarray
        Sample-by-feature Euclidean coordinates.
    metadata : pandas.DataFrame
        Sample metadata containing group and block columns.
    group : str
        Column defining the groups whose dispersions are compared.
    blocks : str
        Column restricting label permutations to exchangeable strata.
    n_permutations : int, default=999
        Number of restricted label permutations.
    random_state : int or numpy.random.Generator, default=42
        Reproducible permutation generator.

    Returns
    -------
    dict
        PERMDISP F, permutation p-value, degrees of freedom, and resolution.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")

    values, metadata = _aligned_inputs(coordinates, metadata)
    missing_columns = [column for column in [group, blocks] if column not in metadata]
    if missing_columns:
        raise KeyError(f"metadata is missing columns: {missing_columns}")
    if metadata[[group, blocks]].isna().any().any():
        raise ValueError(
            "PERMDISP group and block columns cannot contain missing values"
        )

    labels = metadata[group].astype(str).to_numpy()
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError(f"group {group!r} must contain at least two levels")
    if np.any(counts < 2):
        raise ValueError("each PERMDISP group must contain at least two samples")
    if len(labels) - len(unique_labels) < 1:
        raise ValueError("PERMDISP has no within-group degrees of freedom")

    observed_f = _dispersion_f(values, labels)
    block_codes = pd.factorize(metadata[blocks], sort=True)[0]
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    exceedances = 0
    for _ in range(n_permutations):
        order = _permutation_order(block_codes, rng)
        permuted_f = _dispersion_f(values, labels[order])
        exceedances += int(permuted_f >= observed_f - 1e-12)

    return {
        "f_statistic": observed_f,
        "p_value": (exceedances + 1) / (n_permutations + 1),
        "df_between": len(unique_labels) - 1,
        "df_within": len(labels) - len(unique_labels),
        "n_permutations": n_permutations,
        "p_resolution": 1 / (n_permutations + 1),
    }


def compute_dunn_index(
    distance_matrix: np.ndarray,
    labels: Sequence[object] | np.ndarray,
) -> float:
    """Compute the Dunn index for cluster validation.

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
    distances = np.asarray(distance_matrix, dtype=float)
    label_values = np.asarray(labels)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square")
    if len(label_values) != len(distances):
        raise ValueError("labels must match the distance-matrix sample count")

    unique = np.unique(label_values)
    intra = []
    inter = []

    for index, label_i in enumerate(unique):
        indices_i = np.where(label_values == label_i)[0]
        if indices_i.size > 1:
            intra.append(distances[np.ix_(indices_i, indices_i)].max())
        else:
            intra.append(0.0)

        for label_j in unique[index + 1 :]:
            indices_j = np.where(label_values == label_j)[0]
            inter.append(distances[np.ix_(indices_i, indices_j)].min())

    if not intra or not inter:
        return np.nan

    maximum_diameter = max(intra)
    minimum_separation = min(inter)
    return (
        np.inf
        if maximum_diameter == 0
        else float(minimum_separation / maximum_diameter)
    )


def generate_beta_diversity_pca_results(
    frame: pd.DataFrame,
    feature_cols: Sequence[str],
    stratify_col: str = "Cohort",
    cluster_col: str = "DISEASE",
    cohort_order: Sequence[str] | None = None,
) -> BetaDiversityResults:
    """Compute classical MDS coordinates and clustering metrics per stratum.

    This function performs classical multidimensional scaling (equivalent to
    PCA on a double-centered distance matrix) on the Euclidean distances of
    the provided feature columns. It returns PCA-like coordinates and
    clustering metrics (Dunn index and silhouette) for each stratum of the
    supplied ``stratify_col``.

    Parameters
    ----------
    frame : pandas.DataFrame
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
    features = list(feature_cols)
    required = [*features, stratify_col, cluster_col]
    missing_columns = [column for column in required if column not in frame]
    if missing_columns:
        raise KeyError(f"frame is missing columns: {missing_columns}")

    results: BetaDiversityResults = {}
    for stratum_name, stratum_frame in frame.groupby(stratify_col):
        coordinates = stratum_frame[features].to_numpy(dtype=float)
        distances = squareform(pdist(coordinates, metric="euclidean"))

        # Classical MDS via double centering.
        sample_count = distances.shape[0]
        centering = (
            np.eye(sample_count) - np.ones((sample_count, sample_count)) / sample_count
        )
        gram_matrix = -0.5 * centering.dot(distances**2).dot(centering)

        eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        ordination = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])
        explained = eigenvalues / eigenvalues.sum()

        ordination_frame = pd.DataFrame(
            ordination,
            index=stratum_frame.index,
            columns=["Dim1", "Dim2"],
        )
        ordination_frame[cluster_col] = stratum_frame[cluster_col]
        ordination_frame["Explained_Var1"] = explained[0]
        ordination_frame["Explained_Var2"] = explained[1]

        labels = stratum_frame[cluster_col].to_numpy()
        silhouette = silhouette_score(distances, labels, metric="precomputed")
        dunn = compute_dunn_index(distances, labels)

        metrics_frame = pd.DataFrame(
            {
                stratify_col: [stratum_name],
                "Dunn index": [round(dunn, 3)],
                "Mean silhouette width": [round(silhouette, 3)],
            }
        )

        results[stratum_name] = (ordination_frame, metrics_frame)

    if cohort_order is not None:
        ordered: BetaDiversityResults = OrderedDict()
        for name in cohort_order:
            if name in results:
                ordered[name] = results[name]
        for name, value in results.items():
            if name not in ordered:
                ordered[name] = value
        return ordered

    return results


def _collect_geometry_metrics(
    endpoint: str,
    representation: str,
    results: BetaDiversityResults,
) -> pd.DataFrame:
    """Collect cohort-level clustering metrics with explicit semantics."""
    metrics = pd.concat(
        [cohort_metrics for _, cohort_metrics in results.values()],
        ignore_index=True,
    )
    metrics.insert(0, "Representation", representation)
    metrics.insert(0, "Endpoint", endpoint)
    return metrics


def compare_similarity_geometries(
    representations: SimilarityRepresentations,
    feature_cols: Sequence[str],
    *,
    endpoint: str,
    stratify_col: str,
    cluster_col: str,
    stratum_order: Sequence[str],
) -> GeometryComparison:
    """Compare matched percentage and Aitchison descriptive geometries.

    Parameters
    ----------
    representations : SimilarityRepresentations
        Validated representations for one explicitly defined endpoint.
    feature_cols : sequence of str
        Ordered feature identities used in both representations.
    endpoint : str
        Human-readable endpoint definition recorded in the metric output.
    stratify_col : str
        Metadata column defining separate ordination strata.
    cluster_col : str
        Metadata column defining the labels used for descriptive metrics.
    stratum_order : sequence of str
        Required display order of the strata.

    Returns
    -------
    GeometryComparison
        Matched metric table and both sets of cohort ordination results.
    """
    features = list(feature_cols)
    strata = list(stratum_order)
    percentage_results = generate_beta_diversity_pca_results(
        representations.percentage_residualized,
        features,
        stratify_col=stratify_col,
        cluster_col=cluster_col,
        cohort_order=strata,
    )
    clr_results = generate_beta_diversity_pca_results(
        representations.clr_residualized,
        features,
        stratify_col=stratify_col,
        cluster_col=cluster_col,
        cohort_order=strata,
    )
    metrics = pd.concat(
        [
            _collect_geometry_metrics(
                endpoint,
                "Relative abundance",
                percentage_results,
            ),
            _collect_geometry_metrics(
                endpoint,
                "Deterministic CLR (Aitchison)",
                clr_results,
            ),
        ],
        ignore_index=True,
    )
    return GeometryComparison(
        metrics=metrics,
        percentage_ordination=percentage_results,
        clr_ordination=clr_results,
    )


def run_global_similarity_inference(
    representations: SimilarityRepresentations,
    feature_cols: Sequence[str],
    *,
    endpoint: str,
    term: str,
    covariates: Sequence[str],
    blocks: str,
    n_permutations: int = 999,
    random_state: int = 42,
) -> pd.Series:
    """Run adjusted PERMANOVA and matching blocked PERMDISP on CLR data.

    Parameters
    ----------
    representations : SimilarityRepresentations
        Validated deterministic CLR representation for one endpoint.
    feature_cols : sequence of str
        Ordered CLR feature identities defining Aitchison geometry.
    endpoint : str
        Human-readable endpoint definition recorded in the output.
    term : str
        Metadata column whose adjusted global effect is tested.
    covariates : sequence of str
        Reduced-model covariates, including the blocking variable when it is
        also an adjustment factor.
    blocks : str
        Metadata column restricting exchangeability for both tests.
    n_permutations : int, default=999
        Number of blocked permutations.
    random_state : int, default=42
        Reproducible permutation seed.

    Returns
    -------
    pandas.Series
        Complete inferential record, including effect magnitude, test
        statistics, degrees of freedom, p-values, and permutation resolution.
    """
    clr_frame = representations.clr
    features = list(feature_cols)
    permanova = freedman_lane_permanova(
        clr_frame[features],
        clr_frame,
        term=term,
        covariates=list(covariates),
        blocks=blocks,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    permdisp = blocked_permdisp(
        clr_frame[features],
        clr_frame,
        group=term,
        blocks=blocks,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return pd.Series(
        {
            "Endpoint": endpoint,
            "PERMANOVA pseudo-F": permanova["pseudo_f"],
            "Disease partial R2": permanova["partial_r2"],
            "PERMANOVA df": (f"{permanova['df_term']}, {permanova['df_residual']}"),
            "PERMANOVA p": permanova["p_value"],
            "PERMDISP F": permdisp["f_statistic"],
            "PERMDISP df": (f"{permdisp['df_between']}, {permdisp['df_within']}"),
            "PERMDISP p": permdisp["p_value"],
            "Permutations": permanova["n_permutations"],
            "p-value resolution": permanova["p_resolution"],
        }
    )
