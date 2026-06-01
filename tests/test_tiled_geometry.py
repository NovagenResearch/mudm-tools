"""Tests for TiledGeometry base class and tiled mode on TIN/PolyhedralSurface."""

import pytest
from pydantic import ValidationError
from mudm.model import TIN, PolyhedralSurface


class TestTINTiledMode:
    """TIN with tiles instead of inline coordinates."""

    def test_tin_with_tiles_valid(self):
        """TIN with tiles and empty coordinates is valid."""
        t = TIN(type="TIN", coordinates=[], tiles=["0/0/0/0", "2/0/0/0"])
        assert t.tiles == ["0/0/0/0", "2/0/0/0"]
        assert t.coordinates == []

    def test_tin_inline_still_works(self):
        """Existing TIN with inline coordinates still validates."""
        face = [[[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]]
        t = TIN(type="TIN", coordinates=face)
        assert len(t.coordinates) == 1
        assert t.tiles is None

    def test_tin_both_empty_fails(self):
        """TIN with no coordinates and no tiles must fail."""
        with pytest.raises(ValidationError, match="requires either coordinates or tiles"):
            TIN(type="TIN", coordinates=[])

    def test_tin_both_present_ok(self):
        """TIN with both coordinates and tiles is allowed (preview + full tiles)."""
        face = [[[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]]
        t = TIN(type="TIN", coordinates=face, tiles=["0/0/0/0"])
        assert len(t.coordinates) == 1
        assert t.tiles == ["0/0/0/0"]

    def test_tin_tiles_none_with_coords_ok(self):
        """TIN with tiles=None and valid coordinates is fine (original behavior)."""
        face = [[[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]]
        t = TIN(type="TIN", coordinates=face)
        assert t.tiles is None

    def test_tin_bbox3d_returns_none_in_tiled_mode(self):
        """bbox3d returns None when coordinates are empty (tiled mode)."""
        t = TIN(type="TIN", coordinates=[], tiles=["0/0/0/0"])
        assert t.bbox3d() is None

    def test_tin_centroid3d_returns_none_in_tiled_mode(self):
        """centroid3d returns None when coordinates are empty (tiled mode)."""
        t = TIN(type="TIN", coordinates=[], tiles=["0/0/0/0"])
        assert t.centroid3d() is None

    def test_tin_bbox3d_works_with_coords(self):
        """bbox3d still works with inline coordinates."""
        face = [[[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]]
        t = TIN(type="TIN", coordinates=face)
        bbox = t.bbox3d()
        assert bbox is not None
        assert bbox == (0, 0, 0, 1, 1, 0)

    def test_tin_triangle_validation_still_enforced(self):
        """Triangle validation still runs when coordinates are non-empty."""
        bad_face = [[[[0, 0, 0], [1, 0, 0], [0, 0, 0]]]]  # only 3 positions
        with pytest.raises(ValidationError):
            TIN(type="TIN", coordinates=bad_face)


class TestPolyhedralSurfaceTiledMode:
    """PolyhedralSurface with tiles instead of inline coordinates."""

    def test_polyhedral_with_tiles_valid(self):
        """PolyhedralSurface with tiles and empty coordinates is valid."""
        p = PolyhedralSurface(type="PolyhedralSurface", coordinates=[], tiles=["0/0/0/0"])
        assert p.tiles == ["0/0/0/0"]

    def test_polyhedral_both_empty_fails(self):
        """PolyhedralSurface with no coordinates and no tiles must fail."""
        with pytest.raises(ValidationError, match="requires either coordinates or tiles"):
            PolyhedralSurface(type="PolyhedralSurface", coordinates=[])

    def test_polyhedral_bbox3d_returns_none_in_tiled_mode(self):
        """bbox3d returns None when coordinates are empty (tiled mode)."""
        p = PolyhedralSurface(type="PolyhedralSurface", coordinates=[], tiles=["0/0/0/0"])
        assert p.bbox3d() is None

    def test_polyhedral_centroid3d_returns_none_in_tiled_mode(self):
        """centroid3d returns None when coordinates are empty (tiled mode)."""
        p = PolyhedralSurface(type="PolyhedralSurface", coordinates=[], tiles=["0/0/0/0"])
        assert p.centroid3d() is None

    def test_polyhedral_inline_still_works(self):
        """Existing PolyhedralSurface with inline coordinates still validates."""
        face = [[[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]]
        p = PolyhedralSurface(type="PolyhedralSurface", coordinates=face)
        assert len(p.coordinates) == 1


from mudm_tools.tiling3d.tilejson3d import TileEncoding, TileModel3D  # noqa: E402


class TestTileEncoding:
    """TileEncoding model validation."""

    def test_known_format_valid(self):
        enc = TileEncoding(format="glb", compression="meshopt", path="3dtiles", extension=".glb")
        assert enc.format == "glb"
        assert enc.compression == "meshopt"

    def test_unknown_format_accepted(self):
        """Semi-open enum: unknown formats are accepted."""
        enc = TileEncoding(format="custom-format", path="custom", extension=".bin")
        assert enc.format == "custom-format"

    def test_all_known_formats(self):
        for fmt in ["glb", "parquet", "arrow", "neuroglancer-precomputed"]:
            enc = TileEncoding(format=fmt, path="p", extension=".x")
            assert enc.format == fmt

    def test_all_known_compressions(self):
        for comp in ["meshopt", "draco", "zstd"]:
            enc = TileEncoding(format="glb", compression=comp, path="p", extension=".x")
            assert enc.compression == comp

    def test_compression_optional(self):
        enc = TileEncoding(format="arrow", path="parquet", extension=".arrow")
        assert enc.compression is None


class TestTileModel3DEncodings:
    """TileModel3D with new encoding fields."""

    def test_encodings_field(self):
        model = TileModel3D(
            tilejson="3.0.0",
            tiles=["{z}/{x}/{y}/{d}"],
            vector_layers=[{"id": "default", "fields": {}}],
            maxzoom=4,
            encodings=[
                TileEncoding(format="glb", compression="meshopt", path="3dtiles", extension=".glb"),
            ],
        )
        assert len(model.encodings) == 1
        assert model.encodings[0].format == "glb"

    def test_zoom_counts_field(self):
        model = TileModel3D(
            tilejson="3.0.0",
            tiles=["{z}/{x}/{y}/{d}"],
            vector_layers=[{"id": "default", "fields": {}}],
            zoom_counts={"0": 1, "1": 8, "2": 60},
        )
        assert model.zoom_counts["2"] == 60

    def test_id_fields_field(self):
        model = TileModel3D(
            tilejson="3.0.0",
            tiles=["{z}/{x}/{y}/{d}"],
            vector_layers=[{"id": "default", "fields": {}}],
            id_fields=["body_id", "instance"],
        )
        assert model.id_fields == ["body_id", "instance"]

    def test_all_new_fields_optional(self):
        """All new fields default to None — backward compatible."""
        model = TileModel3D(
            tilejson="3.0.0",
            tiles=["{z}/{x}/{y}/{d}"],
            vector_layers=[{"id": "default", "fields": {}}],
        )
        assert model.encodings is None
        assert model.zoom_counts is None
        assert model.id_fields is None

    def test_tilejson3d_roundtrip(self):
        """Serialize TileModel3D with encodings to JSON and back."""
        model = TileModel3D(
            tilejson="3.0.0",
            tiles=["{z}/{x}/{y}/{d}"],
            vector_layers=[{"id": "default", "fields": {}}],
            maxzoom=4,
            bounds3d=[8.0, 50776.0, 23224.0, 275424.0, 300544.0, 323680.0],
            encodings=[
                TileEncoding(format="glb", compression="meshopt", path="3dtiles", extension=".glb"),
                TileEncoding(
                    format="parquet", compression="zstd", path="parquet", extension=".parquet"
                ),
            ],
            zoom_counts={"0": 1, "1": 8, "2": 60},
            id_fields=["body_id", "instance"],
        )
        json_str = model.model_dump_json()
        loaded = TileModel3D.model_validate_json(json_str)
        assert loaded.maxzoom == 4
        assert len(loaded.encodings) == 2
        assert loaded.encodings[0].format == "glb"
        assert loaded.encodings[1].compression == "zstd"
        assert loaded.zoom_counts == {"0": 1, "1": 8, "2": 60}
        assert loaded.id_fields == ["body_id", "instance"]


from mudm.tilemodel import PyramidEntry, PyramidJSON  # noqa: E402


class TestPyramidJSON:
    """PyramidJSON manifest model."""

    def test_minimal_pyramid_entry(self):
        entry = PyramidEntry(id="hemibrain")
        assert entry.id == "hemibrain"
        assert entry.tilejson == "tilejson3d.json"
        assert entry.features == "features.json"

    def test_full_pyramid_entry(self):
        entry = PyramidEntry(
            id="hemibrain_full",
            label="Hemibrain v1.2.1",
            tiles=2835,
            feature_count=5099,
            size_bytes=31440000000,
        )
        assert entry.feature_count == 5099

    def test_pyramid_json_with_version(self):
        manifest = PyramidJSON(
            pyramids=[
                PyramidEntry(id="hemibrain"),
                PyramidEntry(id="mouselight_2018-04-25"),
            ]
        )
        assert manifest.version == "1.0"
        assert len(manifest.pyramids) == 2

    def test_pyramid_json_roundtrip(self):
        """Serialize to JSON and back."""
        manifest = PyramidJSON(
            pyramids=[
                PyramidEntry(
                    id="test", label="Test Pyramid", tiles=100, feature_count=50, size_bytes=1000000
                ),
            ]
        )
        json_str = manifest.model_dump_json()
        loaded = PyramidJSON.model_validate_json(json_str)
        assert loaded.pyramids[0].id == "test"
        assert loaded.version == "1.0"
