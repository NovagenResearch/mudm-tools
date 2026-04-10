"""Tests for Neuroglancer annotation writer."""

import json
import struct
from pathlib import Path

import pytest

from mudm.model import MuDMFeature
from mudm_tools.neuroglancer.annotation_writer import (
    lines_to_annotation_binary,
    points_to_annotation_binary,
    write_annotations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _point_feature(x, y, z) -> MuDMFeature:
    return MuDMFeature(
        type="Feature",
        geometry={"type": "Point", "coordinates": [x, y, z]},
        properties={},
    )


def _line_feature(coords) -> MuDMFeature:
    return MuDMFeature(
        type="Feature",
        geometry={"type": "LineString", "coordinates": coords},
        properties={},
    )


# ---------------------------------------------------------------------------
# Tests: point binary
# ---------------------------------------------------------------------------

class TestPointsToAnnotationBinary:
    def test_single_point(self):
        data = points_to_annotation_binary([(1.0, 2.0, 3.0)], [0])
        # uint64(1) + 3×float32 + uint64(1)
        assert len(data) == 8 + 12 + 8
        count = struct.unpack_from("<Q", data, 0)[0]
        assert count == 1
        coords = struct.unpack_from("<3f", data, 8)
        assert coords == (1.0, 2.0, 3.0)
        ann_id = struct.unpack_from("<Q", data, 20)[0]
        assert ann_id == 0

    def test_multiple_points(self):
        pts = [(0, 0, 0), (10, 20, 30)]
        data = points_to_annotation_binary(pts, [0, 1])
        count = struct.unpack_from("<Q", data, 0)[0]
        assert count == 2

    def test_empty(self):
        data = points_to_annotation_binary([], [])
        count = struct.unpack_from("<Q", data, 0)[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: line binary
# ---------------------------------------------------------------------------

class TestLinesToAnnotationBinary:
    def test_single_line(self):
        line = (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)
        data = lines_to_annotation_binary([line], [0])
        # uint64(1) + 6×float32 + uint64(1)
        assert len(data) == 8 + 24 + 8
        count = struct.unpack_from("<Q", data, 0)[0]
        assert count == 1
        coords = struct.unpack_from("<6f", data, 8)
        assert coords == (0.0, 0.0, 0.0, 10.0, 10.0, 10.0)

    def test_empty(self):
        data = lines_to_annotation_binary([], [])
        count = struct.unpack_from("<Q", data, 0)[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Tests: write_annotations (disk I/O)
# ---------------------------------------------------------------------------

class TestWriteAnnotations:
    def test_point_creates_directory(self, tmp_path):
        features = [_point_feature(1, 2, 3), _point_feature(4, 5, 6)]
        out = tmp_path / "ann"
        write_annotations(out, features, "point")
        assert (out / "info").exists()
        assert (out / "by_id").is_dir()
        assert (out / "spatial0" / "0_0_0").exists()

    def test_point_info_json(self, tmp_path):
        features = [_point_feature(0, 0, 0), _point_feature(100, 100, 100)]
        out = tmp_path / "ann"
        write_annotations(out, features, "point")
        info = json.loads((out / "info").read_text())
        assert info["@type"] == "neuroglancer_annotations_v1"
        assert info["annotation_type"] == "POINT"
        assert info["lower_bound"] == [0.0, 0.0, 0.0]
        assert info["upper_bound"] == [100.0, 100.0, 100.0]
        # Required fields for Neuroglancer
        assert info["properties"] == []
        assert info["relationships"] == []
        assert info["by_id"] == {"key": "by_id"}

    def test_line_creates_files(self, tmp_path):
        features = [_line_feature([[0, 0, 0], [10, 10, 10]])]
        out = tmp_path / "ann"
        write_annotations(out, features, "line")
        assert (out / "info").exists()
        assert (out / "by_id").is_dir()
        assert (out / "spatial0" / "0_0_0").exists()

    def test_line_info_json(self, tmp_path):
        features = [_line_feature([[0, 0, 0], [10, 10, 10]])]
        out = tmp_path / "ann"
        write_annotations(out, features, "line")
        info = json.loads((out / "info").read_text())
        assert info["annotation_type"] == "LINE"

    def test_bounds_calculation(self, tmp_path):
        features = [
            _point_feature(-5, -10, 0),
            _point_feature(50, 100, 200),
        ]
        out = tmp_path / "ann"
        write_annotations(out, features, "point")
        info = json.loads((out / "info").read_text())
        assert info["lower_bound"] == [-5.0, -10.0, 0.0]
        assert info["upper_bound"] == [50.0, 100.0, 200.0]
