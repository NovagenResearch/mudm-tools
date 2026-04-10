"""Tests for mudm.polygen3d — random 3D geometry generator."""

import pytest

from mudm_tools.polygen3d import (
    generate_random_tins,
    generate_random_points_3d,
    generate_random_lines_3d,
    generate_3d_collection,
)
from mudm.model import TIN

BOUNDS = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)


class TestGenerateRandomTins:
    def test_count(self):
        tins = generate_random_tins(5, BOUNDS, triangles_per_tin=4)
        assert len(tins) == 5

    def test_triangle_faces(self):
        """Each TIN face must be a single ring of exactly 4 positions."""
        tins = generate_random_tins(3, BOUNDS, triangles_per_tin=6)
        for feat in tins:
            geom = feat.geometry
            assert isinstance(geom, TIN)
            assert len(geom.coordinates) == 6
            for face in geom.coordinates:
                assert len(face) == 1, "each face has exactly one ring"
                ring = face[0]
                assert len(ring) == 4, "triangle ring = 3 vertices + close"
                assert ring[0] == ring[-1], "ring must be closed"

    def test_coords_within_bounds(self):
        """All TIN vertices should be roughly within bounds (± radius)."""
        tins = generate_random_tins(10, BOUNDS, triangles_per_tin=4)
        # Allow radius slack: 5% of max_dim = 5
        slack = 10.0
        for feat in tins:
            geom = feat.geometry
            assert isinstance(geom, TIN)
            for face in geom.coordinates:
                for pos in face[0]:
                    assert -slack <= pos[0] <= 100 + slack
                    assert -slack <= pos[1] <= 100 + slack
                    assert -slack <= pos[2] <= 100 + slack

    def test_properties(self):
        tins = generate_random_tins(1, BOUNDS, triangles_per_tin=3)
        props = tins[0].properties
        assert props["kind"] == "tin"
        assert props["triangles"] == 3


class TestGenerateRandomPoints3D:
    def test_count(self):
        pts = generate_random_points_3d(7, BOUNDS)
        assert len(pts) == 7

    def test_3d_coords(self):
        pts = generate_random_points_3d(5, BOUNDS)
        for feat in pts:
            coords = feat.geometry.coordinates
            assert len(coords) == 3, "Point must have 3 coordinates (x, y, z)"
            assert 0 <= coords[0] <= 100
            assert 0 <= coords[1] <= 100
            assert 0 <= coords[2] <= 100


class TestGenerateRandomLines3D:
    def test_count(self):
        lines = generate_random_lines_3d(4, BOUNDS)
        assert len(lines) == 4

    def test_vertex_count(self):
        lines = generate_random_lines_3d(10, BOUNDS, min_verts=3, max_verts=8)
        for feat in lines:
            coords = feat.geometry.coordinates
            assert 3 <= len(coords) <= 8

    def test_3d_coords(self):
        lines = generate_random_lines_3d(3, BOUNDS)
        for feat in lines:
            for coord in feat.geometry.coordinates:
                assert len(coord) == 3


class TestGenerate3DCollection:
    def test_feature_count(self):
        coll = generate_3d_collection(
            n_tins=4, n_points=3, n_lines=2, bounds=BOUNDS, seed=42,
        )
        assert len(coll.features) == 9  # 4 + 3 + 2

    def test_metadata_attached(self):
        coll = generate_3d_collection(
            n_tins=2, n_points=1, n_lines=1,
            bounds=BOUNDS, n_meta_keys=3, seed=42,
        )
        for feat in coll.features:
            assert "meta1" in feat.properties
            assert "meta2" in feat.properties
            assert "meta3" in feat.properties

    def test_reproducible_with_seed(self):
        c1 = generate_3d_collection(n_tins=3, n_points=2, n_lines=1, seed=123)
        c2 = generate_3d_collection(n_tins=3, n_points=2, n_lines=1, seed=123)
        for f1, f2 in zip(c1.features, c2.features):
            assert f1.geometry == f2.geometry

    def test_validates_as_microjson(self):
        coll = generate_3d_collection(seed=0)
        coll.model_validate(coll)
