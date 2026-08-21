"""Verification tests for similarity-analysis inference."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import f_oneway

from ibd_biom_glycoda.analysis.similarity import (
    _permutation_order,
    blocked_permdisp,
    compare_similarity_geometries,
    freedman_lane_permanova,
    prepare_similarity_representations,
    run_global_similarity_inference,
)


@pytest.fixture
def similarity_frame():
    """Closed positive compositions with balanced labels in two cohorts."""
    rng = np.random.default_rng(21)
    features = ["GP1", "GP2", "GP3", "GP4"]
    compositions = rng.uniform(0.5, 3.0, size=(12, len(features)))
    compositions = 100 * compositions / compositions.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        compositions,
        index=[f"sample-{index}" for index in range(12)],
        columns=features,
    )
    frame["Disease"] = (["control"] * 3 + ["case"] * 3) * 2
    frame["Age"] = np.linspace(20, 70, len(frame))
    frame["Sex"] = ["F", "M", "F", "M", "F", "M"] * 2
    frame["Cohort"] = ["A"] * 6 + ["B"] * 6
    return frame, features


def test_prepare_similarity_representations_preserves_scientific_contract(
    similarity_frame,
):
    frame, features = similarity_frame
    original = frame.copy(deep=True)

    result = prepare_similarity_representations(
        frame,
        features,
        residualization_covariates=["Age", "Sex"],
        random_state=9,
    )

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_index_equal(result.clr.index, frame.index)
    assert list(result.clr[features].columns) == features
    np.testing.assert_allclose(result.clr[features].sum(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(
        result.closed_percentages[features].sum(axis=1),
        100.0,
        atol=1e-8,
    )
    assert result.contract_summary.loc[0, "Value"] == len(frame)


@pytest.mark.parametrize("invalid_value", [0.0, np.nan])
def test_prepare_similarity_representations_rejects_invalid_parts(
    similarity_frame,
    invalid_value,
):
    frame, features = similarity_frame
    frame.loc[frame.index[0], features[0]] = invalid_value

    with pytest.raises(ValueError):
        prepare_similarity_representations(
            frame,
            features,
            residualization_covariates=["Age", "Sex"],
        )


def test_prepare_similarity_representations_rejects_broken_closure(
    similarity_frame,
):
    frame, features = similarity_frame
    frame.loc[frame.index[0], features] *= 0.9

    with pytest.raises(ValueError, match="must sum to 100"):
        prepare_similarity_representations(
            frame,
            features,
            residualization_covariates=["Age", "Sex"],
        )


def test_similarity_workflow_components_record_endpoint_semantics(similarity_frame):
    frame, features = similarity_frame
    representations = prepare_similarity_representations(
        frame,
        features,
        residualization_covariates=["Age", "Sex"],
    )

    geometry = compare_similarity_geometries(
        representations,
        features,
        endpoint="case vs control",
        stratify_col="Cohort",
        cluster_col="Disease",
        stratum_order=["A", "B"],
    )
    inference = run_global_similarity_inference(
        representations,
        features,
        endpoint="case vs control",
        term="Disease",
        covariates=["Age", "Sex", "Cohort"],
        blocks="Cohort",
        n_permutations=19,
        random_state=4,
    )

    assert set(geometry.metrics["Representation"]) == {
        "Relative abundance",
        "Deterministic CLR (Aitchison)",
    }
    assert set(geometry.metrics["Endpoint"]) == {"case vs control"}
    assert list(geometry.clr_ordination) == ["A", "B"]
    for cohort in ["A", "B"]:
        expected_index = frame.index[frame["Cohort"] == cohort]
        pd.testing.assert_index_equal(
            geometry.clr_ordination[cohort][0].index,
            expected_index,
        )
    assert inference["Endpoint"] == "case vs control"
    assert inference["PERMANOVA df"] == "1, 7"
    assert inference["p-value resolution"] == 0.05


def test_permutation_order_never_crosses_blocks():
    """Restricted permutations preserve every sample's cohort membership."""
    block_codes = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    rng = np.random.default_rng(13)

    for _ in range(20):
        order = _permutation_order(block_codes, rng)
        np.testing.assert_array_equal(block_codes[order], block_codes)


def test_permanova_matches_univariate_anova_without_covariates():
    """A one-dimensional, one-factor pseudo-F reduces to classical ANOVA."""
    coordinates = pd.DataFrame({"x": [0.0, 0.5, 1.0, 3.0, 4.0, 5.0]})
    metadata = pd.DataFrame(
        {
            "disease": ["control"] * 3 + ["case"] * 3,
            "cohort": ["A"] * 6,
        }
    )

    result = freedman_lane_permanova(
        coordinates,
        metadata,
        term="disease",
        covariates=[],
        blocks="cohort",
        n_permutations=19,
        random_state=7,
    )
    reference_f = f_oneway(coordinates.loc[:2, "x"], coordinates.loc[3:, "x"]).statistic
    reference_partial_r2 = reference_f / (reference_f + 4)

    np.testing.assert_allclose(result["pseudo_f"], reference_f)
    np.testing.assert_allclose(result["partial_r2"], reference_partial_r2)
    assert result["p_value"] >= result["p_resolution"] == 0.05


def test_permanova_is_invariant_to_translation_and_rotation():
    """Rigid transformations cannot change Euclidean distance inference."""
    rng = np.random.default_rng(11)
    coordinates = pd.DataFrame(rng.normal(size=(24, 3)))
    metadata = pd.DataFrame(
        {
            "disease": ["control", "case"] * 12,
            "age": np.linspace(20, 70, 24),
            "cohort": np.repeat(["A", "B", "C"], 8),
        }
    )
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    transformed = coordinates.to_numpy() @ rotation + np.array([10.0, -2.0, 4.0])

    original = freedman_lane_permanova(
        coordinates,
        metadata,
        term="disease",
        covariates=["age", "cohort"],
        blocks="cohort",
        n_permutations=49,
        random_state=3,
    )
    rigid = freedman_lane_permanova(
        transformed,
        metadata,
        term="disease",
        covariates=["age", "cohort"],
        blocks="cohort",
        n_permutations=49,
        random_state=3,
    )

    for key in ["pseudo_f", "partial_r2", "p_value"]:
        np.testing.assert_allclose(original[key], rigid[key], atol=1e-12)


def test_blocked_permdisp_detects_large_dispersion_difference():
    """A strong within-group scale difference yields a small blocked p-value."""
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    coordinates = np.vstack([unit_circle, 5.0 * unit_circle])
    metadata = pd.DataFrame(
        {
            "disease": ["control"] * 20 + ["case"] * 20,
            "cohort": np.tile(np.repeat(["A", "B"], 10), 2),
        }
    )

    result = blocked_permdisp(
        coordinates,
        metadata,
        group="disease",
        blocks="cohort",
        n_permutations=199,
        random_state=5,
    )

    assert result["f_statistic"] > 100
    assert result["p_value"] == result["p_resolution"] == 0.005


def test_dataframe_coordinates_align_metadata_by_sample_identity():
    """Reordering metadata cannot detach disease labels from samples."""
    coordinates = pd.DataFrame(
        {"x": [0.0, 0.2, 3.0, 3.2]}, index=["s0", "s1", "s2", "s3"]
    )
    metadata = pd.DataFrame(
        {
            "disease": ["case", "control", "case", "control"],
            "cohort": ["A"] * 4,
        },
        index=["s2", "s0", "s3", "s1"],
    )

    ordered = freedman_lane_permanova(
        coordinates,
        metadata.loc[coordinates.index],
        term="disease",
        covariates=[],
        blocks="cohort",
        n_permutations=19,
        random_state=2,
    )
    shuffled = freedman_lane_permanova(
        coordinates,
        metadata,
        term="disease",
        covariates=[],
        blocks="cohort",
        n_permutations=19,
        random_state=2,
    )

    assert ordered == shuffled
