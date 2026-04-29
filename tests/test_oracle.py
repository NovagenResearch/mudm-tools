"""Synthetic oracle: label lookup against a held-out ground-truth map."""
import pytest

from mudm_tools.active_learning.oracle import SyntheticOracle


def test_oracle_returns_labels_for_known_ids():
    labels = {"cell_1": 0, "cell_2": 3, "cell_3": 1}
    oracle = SyntheticOracle(ground_truth=labels)
    assert oracle.query("cell_1") == 0
    assert oracle.query("cell_2") == 3


def test_oracle_raises_on_unknown_id():
    oracle = SyntheticOracle(ground_truth={"cell_1": 0})
    with pytest.raises(KeyError):
        oracle.query("not_in_ground_truth")


def test_oracle_tracks_query_count():
    oracle = SyntheticOracle(ground_truth={"a": 1, "b": 2, "c": 3})
    oracle.query("a")
    oracle.query("b")
    assert oracle.queries_made == 2


def test_oracle_refuses_test_set_leak():
    """If a feature_id appears in the test_set, querying it must raise — prevents
    leakage of test labels into the AL loop."""
    oracle = SyntheticOracle(
        ground_truth={"cell_1": 0},
        test_set={"cell_1"},
    )
    with pytest.raises(ValueError, match="test_set"):
        oracle.query("cell_1")


def test_oracle_reveal_test_labels_returns_known_test_labels():
    """For evaluation: harness needs to compute test accuracy. Provide a separate
    method that returns labels for test IDs without going through query()."""
    oracle = SyntheticOracle(
        ground_truth={"a": 0, "b": 1, "c": 0},
        test_set={"a", "b"},
    )
    labels = oracle.reveal_test_labels(["a", "b"])
    assert labels == [0, 1]


def test_oracle_reveal_test_labels_rejects_non_test_ids():
    oracle = SyntheticOracle(
        ground_truth={"a": 0, "b": 1},
        test_set={"a"},
    )
    with pytest.raises(ValueError, match="not in test_set"):
        oracle.reveal_test_labels(["b"])
