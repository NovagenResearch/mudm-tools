"""Tests for Neuroglancer unified orchestrator.

Note: to_neuroglancer() no longer dispatches skeletons. For skeleton export,
call write_skeleton() directly with a NeuronMorphology from mudm_tools.swc.
"""

import json
from pathlib import Path

import pytest

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
)
from mudm_tools.neuroglancer.writer import to_neuroglancer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def point_feature():
    return MuDMFeature(
        type="Feature",
        geometry={"type": "Point", "coordinates": [1, 2, 3]},
        properties={"label": "marker"},
    )


@pytest.fixture
def line_feature():
    return MuDMFeature(
        type="Feature",
        geometry={"type": "LineString", "coordinates": [[0, 0, 0], [10, 10, 10]]},
        properties={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToNeuroglancerPointsOnly:
    def test_point_only(self, point_feature, tmp_path):
        result = to_neuroglancer(point_feature, tmp_path)
        assert "point_annotations" in result["paths"]
        assert "skeletons" not in result["paths"]


class TestToNeuroglancerLinesOnly:
    def test_line_only(self, line_feature, tmp_path):
        result = to_neuroglancer(line_feature, tmp_path)
        assert "line_annotations" in result["paths"]
        assert "skeletons" not in result["paths"]


class TestToNeuroglancerMixedCollection:
    def test_mixed_point_and_line(self, point_feature, line_feature, tmp_path):
        collection = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[point_feature, line_feature],
        )
        result = to_neuroglancer(collection, tmp_path)
        assert "point_annotations" in result["paths"]
        assert "line_annotations" in result["paths"]

    def test_directory_structure(self, point_feature, tmp_path):
        to_neuroglancer(point_feature, tmp_path)
        assert (tmp_path / "point_annotations" / "info").exists()
        assert (tmp_path / "point_annotations" / "spatial0" / "0_0_0").exists()


class TestToNeuroglancerViewerState:
    def test_no_viewer_url_without_base(self, point_feature, tmp_path):
        result = to_neuroglancer(point_feature, tmp_path)
        assert "viewer_url" not in result

    def test_viewer_url_with_base(self, point_feature, tmp_path):
        result = to_neuroglancer(
            point_feature, tmp_path, base_url="http://localhost:8080"
        )
        assert "viewer_url" in result
        assert "viewer_state" in result
        assert result["viewer_url"].startswith("http://localhost:8080/#!")
