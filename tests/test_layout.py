"""Tests for shared layout module."""

import pytest
from geojson_pydantic import Point, Polygon

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    PolyhedralSurface,
    TIN,
)
from mudm.layout import (
    Bounds,
    apply_layout,
    compute_collection_offsets,
    geometry_bounds,
)


# ---------------------------------------------------------------------------
# geometry_bounds
# ---------------------------------------------------------------------------

class TestGeometryBounds:
    def test_point(self):
        p = Point(type="Point", coordinates=(1.0, 2.0, 3.0))
        b = geometry_bounds(p)
        assert b == (1.0, 2.0, 3.0, 1.0, 2.0, 3.0)

    def test_polygon_2d(self):
        p = Polygon(
            type="Polygon",
            coordinates=[[(0, 0), (10, 0), (10, 5), (0, 5), (0, 0)]],
        )
        b = geometry_bounds(p)
        assert b is not None
        assert b[0] == 0.0   # xmin
        assert b[1] == 0.0   # ymin
        assert b[2] == 0.0   # zmin (default)
        assert b[3] == 10.0  # xmax
        assert b[4] == 5.0   # ymax
        assert b[5] == 0.0   # zmax (default)

    def test_tin(self):
        tin = TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (10, 0, 0), (5, 10, 5), (0, 0, 0)]],
            ],
        )
        b = geometry_bounds(tin)
        assert b is not None
        assert b[0] == 0.0
        assert b[3] == 10.0
        assert b[5] == 5.0  # zmax

    def test_none_geometry(self):
        assert geometry_bounds(None) is None


# ---------------------------------------------------------------------------
# Helper: TIN feature factory (replaces NeuronMorphology-based helper)
# ---------------------------------------------------------------------------

def _tin_at(x_start=0.0, x_end=100.0):
    """Create a TIN geometry spanning from x_start to x_end."""
    return TIN(
        type="TIN",
        coordinates=[
            [[(x_start, 0, 0), (x_end, 0, 0), ((x_start + x_end) / 2, 10, 5), (x_start, 0, 0)]],
        ],
    )


# ---------------------------------------------------------------------------
# Row layout
# ---------------------------------------------------------------------------

class TestRowLayout:
    def _features(self, *tins):
        return [
            MuDMFeature(type="Feature", geometry=t, properties=None)
            for t in tins
        ]

    def test_single_feature_no_offset(self):
        feats = self._features(_tin_at())
        offsets = compute_collection_offsets(feats)
        assert offsets == [(0.0, 0.0, 0.0)]

    def test_default_no_layout(self):
        """Default (spacing=None) keeps coordinates as-is."""
        feats = self._features(_tin_at(0, 100), _tin_at(0, 80))
        offsets = compute_collection_offsets(feats)
        assert offsets == [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)]

    def test_two_features_auto_spacing(self):
        feats = self._features(_tin_at(0, 100), _tin_at(0, 80))
        offsets = compute_collection_offsets(feats, spacing=0.0)
        assert offsets[0] == (0.0, 0.0, 0.0)
        # Second feature should be shifted right
        assert offsets[1][0] > 100.0  # past first feature's xmax
        assert offsets[1][1] == 0.0
        assert offsets[1][2] == 0.0

    def test_fixed_spacing(self):
        feats = self._features(_tin_at(0, 100), _tin_at(0, 80))
        offsets = compute_collection_offsets(feats, spacing=50.0)
        # Second feature xmin (0) should land at first xmax (100) + gap (50) = 150
        assert abs(offsets[1][0] - 150.0) < 1e-6

    def test_empty_list(self):
        offsets = compute_collection_offsets([])
        assert offsets == []


# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------

class TestGridLayout:
    def _tin(self, width=100.0):
        return _tin_at(0.0, width)

    def _features(self, n, width=100.0):
        return [
            MuDMFeature(
                type="Feature",
                geometry=self._tin(width),
                properties=None,
            )
            for _ in range(n)
        ]

    def test_wraps_to_rows(self):
        """4 features in 2 columns -> 2 rows."""
        feats = self._features(4)
        offsets = compute_collection_offsets(
            feats, spacing=10.0, grid_max_x=2,
        )
        # Feature 0: no offset
        assert offsets[0] == (0.0, 0.0, 0.0)
        # Feature 1: shifted right
        assert offsets[1][0] > 0
        # Feature 2: back to col 0, shifted in Y
        assert abs(offsets[2][0]) < 1e-6
        assert offsets[2][1] > 0

    def test_wraps_to_layers(self):
        """5 features in 2x2 -> 5th goes to layer 1."""
        feats = self._features(5)
        offsets = compute_collection_offsets(
            feats, spacing=10.0, grid_max_x=2, grid_max_y=2,
        )
        # Feature 4: layer 1, col 0, row 0 -> dz > 0
        assert abs(offsets[4][0]) < 1e-6  # col 0
        assert abs(offsets[4][1]) < 1e-6  # row 0
        assert offsets[4][2] > 0  # layer 1

    def test_capacity_error(self):
        feats = self._features(10)
        with pytest.raises(ValueError, match="Cannot fit 10 features"):
            compute_collection_offsets(
                feats, spacing=10.0,
                grid_max_x=2, grid_max_y=2, grid_max_z=2,
            )

    def test_exact_capacity_fits(self):
        feats = self._features(8)
        offsets = compute_collection_offsets(
            feats, spacing=10.0,
            grid_max_x=2, grid_max_y=2, grid_max_z=2,
        )
        assert len(offsets) == 8

    def test_unconstrained_z_always_fits(self):
        feats = self._features(20)
        offsets = compute_collection_offsets(
            feats, spacing=10.0, grid_max_x=3,
        )
        assert len(offsets) == 20


# ---------------------------------------------------------------------------
# apply_layout end-to-end
# ---------------------------------------------------------------------------

class TestApplyLayout:
    def test_single_feature_unchanged(self):
        feat = MuDMFeature(
            type="Feature", geometry=_tin_at(), properties=None,
        )
        coll = MuDMFeatureCollection(
            type="FeatureCollection", features=[feat],
        )
        result = apply_layout(coll)
        # Single feature -- no translation
        assert result.features[0].geometry.coordinates[0][0][0][0] == 0.0

    def test_two_features_coordinates_translated(self):
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(type="Feature", geometry=_tin_at(0, 100), properties=None),
                MuDMFeature(type="Feature", geometry=_tin_at(0, 80), properties=None),
            ],
        )
        result = apply_layout(coll, spacing=50.0)
        # First feature unchanged
        assert result.features[0].geometry.coordinates[0][0][0][0] == 0.0
        # Second feature's xmin vertex should be shifted to 150 (100 + 50)
        assert result.features[1].geometry.coordinates[0][0][0][0] == pytest.approx(150.0)

    def test_original_not_modified(self):
        feat = MuDMFeature(
            type="Feature", geometry=_tin_at(0, 100), properties=None,
        )
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat, MuDMFeature(
                type="Feature", geometry=_tin_at(0, 80), properties=None,
            )],
        )
        result = apply_layout(coll, spacing=50.0)
        # Original should be unchanged
        assert coll.features[1].geometry.coordinates[0][0][0][0] == 0.0
        # Result should be different
        assert result.features[1].geometry.coordinates[0][0][0][0] != 0.0

    def test_grid_layout_coordinates(self):
        """Grid layout modifies coordinates, not just offsets."""
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(type="Feature", geometry=_tin_at(0, 100), properties=None),
                MuDMFeature(type="Feature", geometry=_tin_at(0, 100), properties=None),
                MuDMFeature(type="Feature", geometry=_tin_at(0, 100), properties=None),
                MuDMFeature(type="Feature", geometry=_tin_at(0, 100), properties=None),
            ],
        )
        result = apply_layout(coll, spacing=10.0, grid_max_x=2)
        # Feature 0: unchanged
        assert result.features[0].geometry.coordinates[0][0][0][0] == 0.0
        # Feature 2: col 0, row 1 -> x unchanged, y shifted
        assert abs(result.features[2].geometry.coordinates[0][0][0][0]) < 1e-6
        assert result.features[2].geometry.coordinates[0][0][0][1] > 0

    def test_null_geometry_feature_preserved(self):
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(type="Feature", geometry=_tin_at(0, 100), properties=None),
                MuDMFeature(type="Feature", geometry=None, properties=None),
            ],
        )
        # Should not raise
        result = apply_layout(coll, spacing=50.0)
        assert result.features[1].geometry is None

    def test_properties_preserved(self):
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(type="Feature", geometry=_tin_at(0, 100), properties={"a": 1}),
                MuDMFeature(type="Feature", geometry=_tin_at(0, 80), properties={"b": 2}),
            ],
        )
        result = apply_layout(coll, spacing=50.0)
        assert result.features[0].properties == {"a": 1}
        assert result.features[1].properties == {"b": 2}

    def test_point_geometry_layout(self):
        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(0, 0, 0)),
                    properties=None,
                ),
                MuDMFeature(
                    type="Feature",
                    geometry=Point(type="Point", coordinates=(0, 0, 0)),
                    properties=None,
                ),
            ],
        )
        result = apply_layout(coll, spacing=10.0)
        # Second point should be shifted
        assert result.features[1].geometry.coordinates[0] > 0
