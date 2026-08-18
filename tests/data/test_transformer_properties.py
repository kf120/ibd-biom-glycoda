# -*- coding: utf-8 -*-
"""Metamorphic properties every feature transformer in the pipeline must satisfy.

A transformer is applied separately to the training split, the inner validation
split, the in-sample test split and the held-out cohort. For the split-wise
application to be equivalent to one fixed function of a participant's own
measurements, two properties must hold:

``subset invariance``
    ``transform(X[idx]) == transform(X)[idx]`` -- a participant's transformed
    features do not depend on which other participants share the call.

``permutation equivariance``
    ``transform(X[perm])[inv] == transform(X)`` -- they do not depend on row
    order either.

Neither property is implied by "the transform runs and returns the right shape",
and a reference test that compares a transformer against the function it calls
internally cannot detect a violation of either.
"""

import numpy as np
import pandas as pd
import pytest

from ibd_biom_glycoda.data.transformations import CLRProcessor, GlyCompareProcessor


@pytest.fixture
def composition():
    """Ten samples of five strictly positive parts, rows summing to one."""
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        rng.random((10, 5)) + 0.5,
        index=[f"S{i}" for i in range(10)],
        columns=[f"GP{j}" for j in range(5)],
    )
    return X.div(X.sum(axis=1), axis=0)


def _assert_subset_invariant(processor, X, k=5):
    full = np.asarray(processor.transform(X))
    subset = np.asarray(processor.transform(X.iloc[:k]))
    np.testing.assert_allclose(subset, full[:k], rtol=1e-10, atol=1e-10)


def _assert_permutation_equivariant(processor, X):
    order = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    full = np.asarray(processor.transform(X))
    permuted = np.asarray(processor.transform(X.iloc[order]))
    np.testing.assert_allclose(permuted[np.argsort(order)], full, rtol=1e-10, atol=1e-10)


class TestCLRProcessorProperties:
    """CLR centring is per-sample; the scale-model noise is not."""

    def test_deterministic_clr_is_subset_invariant(self, composition):
        processor = CLRProcessor(gamma=0.0, seed=42).fit(composition)
        _assert_subset_invariant(processor, composition)

    def test_deterministic_clr_is_permutation_equivariant(self, composition):
        processor = CLRProcessor(gamma=0.0, seed=42).fit(composition)
        _assert_permutation_equivariant(processor, composition)

    def test_deterministic_clr_centres_each_row_on_its_own_geometric_mean(self, composition):
        """The scientific claim behind CLR: centring uses only the row itself."""
        processor = CLRProcessor(gamma=0.0, seed=42).fit(composition)
        result = np.asarray(processor.transform(composition))

        log2_parts = np.log2(composition.to_numpy())
        expected = log2_parts - log2_parts.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_repeated_calls_are_deterministic(self, composition):
        processor = CLRProcessor(gamma=0.1, seed=42).fit(composition)
        first = np.asarray(processor.transform(composition))
        second = np.asarray(processor.transform(composition))
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("check", [_assert_subset_invariant, _assert_permutation_equivariant])
    def test_scale_model_noise_breaks_both_properties(self, composition, check):
        """Characterisation test: gamma > 0 makes the transform batch-dependent.

        ``glycowork.clr_transformation`` adds ``norm.rvs(loc=gmean, scale=gamma)``
        drawn as one ``(n_features, n_samples)`` matrix per call. Because the RNG
        stream is consumed in row order, a participant's transformed values
        depend on the size and ordering of the split they happen to sit in, so
        the model is trained on one noise realisation and evaluated on another.

        This test pins the current behaviour so that changing ``gamma``, or a
        change in the upstream implementation, is a visible event rather than a
        silent shift in the features. If it starts failing because the transform
        became deterministic, delete it and rely on the gamma=0 tests above.
        """
        processor = CLRProcessor(gamma=0.1, seed=42).fit(composition)
        with pytest.raises(AssertionError):
            check(processor, composition)


class TestGlyCompareProcessorProperties:
    """GlyCompare is an index-keyed lookup, so both properties must hold."""

    @pytest.fixture
    def processor(self, composition):
        motifs = pd.DataFrame(
            np.arange(30, dtype=float).reshape(10, 3),
            index=composition.index,
            columns=["M1", "M2", "M3"],
        )
        proc = GlyCompareProcessor(
            orig_df=None, coda_cols=[], abd_fname="a.xlsx", annot_fname="b.xlsx"
        )
        # Pre-populating the cache means fit() performs no file access.
        proc.X_glycomotif_full = motifs
        return proc.fit(composition)

    def test_subset_invariant(self, processor, composition):
        _assert_subset_invariant(processor, composition)

    def test_permutation_equivariant(self, processor, composition):
        _assert_permutation_equivariant(processor, composition)

    def test_lookup_is_keyed_on_identity_not_position(self, processor, composition):
        """Reordered input must return each sample's own motif row."""
        shuffled = composition.iloc[[3, 0, 7]]
        result = processor.transform(shuffled)
        expected = processor.X_glycomotif_full.loc[["S3", "S0", "S7"]]
        pd.testing.assert_frame_equal(result, expected)

    def test_unknown_sample_id_raises_rather_than_returning_a_neighbour(self, processor, composition):
        stranger = composition.iloc[:2].rename(index={"S0": "NOT_A_SAMPLE"})
        with pytest.raises(KeyError):
            processor.transform(stranger)
