# -*- coding: utf-8 -*-
"""Cohort one-hot decoding under leave-one-cohort-out.

The encoder is fitted on the training cohorts only, so under LOCO every held-out
participant belongs to a category the encoder never saw. With
``handle_unknown='ignore'`` those rows are encoded as all zeros, and ``argmax``
returns 0 for an all-zero row -- silently labelling the entire held-out cohort as
the first training cohort. These tests fix the decoding contract at that boundary.
"""

import numpy as np
import pandas as pd
import pytest

from ibd_biom_glycoda.analysis.loco_cv import create_group_identifiers
from ibd_biom_glycoda.analysis.pipelines import create_demographic_subgroup_names
from ibd_biom_glycoda.data.dataset import (
    UNKNOWN_COHORT_INDEX,
    UNKNOWN_COHORT_LABEL,
    cohort_index_from_onehot,
    extract_cohort_features,
)


@pytest.fixture
def fitted_encoder():
    train = pd.DataFrame({"Cohort": ["UK", "US", "IT", "UK", "US"]})
    _, locations, ohe = extract_cohort_features(train, fit=True)
    return locations, ohe


class TestCohortIndexFromOneHot:
    def test_known_categories_decode_to_their_index(self, fitted_encoder):
        locations, ohe = fitted_encoder
        df = pd.DataFrame({"Cohort": ["US", "IT", "UK"]})
        V, _, _ = extract_cohort_features(df, ohe_cohort=ohe, fit=False)

        decoded = cohort_index_from_onehot(V)

        assert [locations[i] for i in decoded] == ["US", "IT", "UK"]

    def test_held_out_cohort_is_flagged_not_folded_into_the_first_cohort(self, fitted_encoder):
        """The regression this module exists for."""
        _, ohe = fitted_encoder
        held_out = pd.DataFrame({"Cohort": ["NL", "NL", "NL"]})
        V, _, _ = extract_cohort_features(held_out, ohe_cohort=ohe, fit=False)

        assert (V.sum(axis=1) == 0).all(), "expected an all-zero encoding for an unseen cohort"
        decoded = cohort_index_from_onehot(V)

        assert (decoded == UNKNOWN_COHORT_INDEX).all()
        assert not (decoded == 0).any(), "all-zero rows must not decode to the first cohort"

    def test_mixed_known_and_unknown_rows(self, fitted_encoder):
        _, ohe = fitted_encoder
        df = pd.DataFrame({"Cohort": ["UK", "NL", "IT"]})
        V, _, _ = extract_cohort_features(df, ohe_cohort=ohe, fit=False)

        decoded = cohort_index_from_onehot(V)

        assert decoded[1] == UNKNOWN_COHORT_INDEX
        assert decoded[0] != UNKNOWN_COHORT_INDEX and decoded[2] != UNKNOWN_COHORT_INDEX

    def test_rejects_non_matrix_input(self):
        with pytest.raises(ValueError, match="2-D"):
            cohort_index_from_onehot(np.array([0, 1, 0]))


class TestGroupIdentifiersRefuseUnencodedCohorts:
    def test_all_zero_cohort_row_raises_instead_of_grouping_silently(self):
        V = np.zeros((3, 3))
        Z_bin = np.array([[0, 0], [1, 1], [0, 1]])

        with pytest.raises(ValueError, match="all-zero cohort one-hot"):
            create_group_identifiers(V, Z_bin, [0, 40, np.inf])

    def test_valid_one_hot_still_produces_the_documented_cross_product(self):
        V = np.eye(3)[[0, 1, 2]]
        Z_bin = np.array([[0, 0], [1, 0], [0, 1]])

        groups, n_groups = create_group_identifiers(V, Z_bin, [0, 40, np.inf])

        # group_id = cohort * (n_age_bins * n_sex) + age_bin * n_sex + sex
        assert n_groups == 3 * 2 * 2
        np.testing.assert_array_equal(groups, [0, 4 + 0 + 1, 8 + 2 + 0])


class TestDemographicSubgroupNames:
    def test_flat_cohort_list_yields_whole_cohort_names(self):
        """``extract_cohort_features`` returns a flat list.

        Indexing ``cohort_locations[0]`` unconditionally iterates the characters
        of the first cohort name, turning ['IT', 'UK', 'US'] into {0: 'I', 1: 'T'}.
        """
        names = create_demographic_subgroup_names([0, 40, np.inf], ["IT", "UK", "US"])

        assert names["location"][0] == "IT"
        assert names["location"][1] == "UK"
        assert names["location"][2] == "US"

    def test_nested_cohort_list_still_supported(self):
        names = create_demographic_subgroup_names([0, 40, np.inf], [["IT", "UK", "US"]])

        assert [names["location"][i] for i in range(3)] == ["IT", "UK", "US"]

    def test_unknown_cohort_index_has_its_own_label(self):
        names = create_demographic_subgroup_names([0, 40, np.inf], ["IT", "UK"])

        assert names["location"][UNKNOWN_COHORT_INDEX] == UNKNOWN_COHORT_LABEL
        assert UNKNOWN_COHORT_LABEL not in {names["location"][0], names["location"][1]}

    def test_age_and_sex_names_unchanged(self):
        names = create_demographic_subgroup_names([0, 40, np.inf], ["IT"])

        assert names["age"] == {0: "<40", 1: ">40"}
        assert names["sex"] == {0: "M", 1: "F"}
