"""Tests for Arrow/GeoParquet reader: Parquet -> MuDM round-trip."""

from __future__ import annotations


import pytest
from shapely.geometry import (
    LineString as ShapelyLineString,
    MultiPolygon as ShapelyMultiPolygon,
    Point as ShapelyPoint,
    Polygon as ShapelyPolygon,
)

from mudm_tools.arrow import (
    ArrowConfig,
    from_arrow_table,
    from_geoparquet,
    to_arrow_table,
    to_geoparquet,
)
from mudm_tools.arrow._from_geometry import (
    shapely_to_microjson,
)
from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    TIN,
)

from geojson_pydantic import (
    LineString,
    MultiPolygon,
    Point,
    Polygon,
)

# ---- Fixtures ----


def _point_feat(fid="p1", x=1.0, y=2.0, z=3.0, props=None):
    return MuDMFeature(
        type="Feature",
        id=fid,
        geometry=Point(type="Point", coordinates=(x, y, z)),
        properties=props or {},
    )


def _polygon_feat(fid="pg1"):
    return MuDMFeature(
        type="Feature",
        id=fid,
        geometry=Polygon(
            type="Polygon",
            coordinates=[[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]],
        ),
        properties={"area": 100},
    )


def _polygon_with_hole_feat():
    return MuDMFeature(
        type="Feature",
        id="pwh1",
        geometry=Polygon(
            type="Polygon",
            coordinates=[
                [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)],
                [(5, 5), (15, 5), (15, 15), (5, 15), (5, 5)],
            ],
        ),
        properties={},
    )


def _linestring_feat():
    return MuDMFeature(
        type="Feature",
        id="ls1",
        geometry=LineString(
            type="LineString",
            coordinates=[(0, 0, 0), (1, 1, 1), (2, 0, 2)],
        ),
        properties={"name": "line1"},
    )


def _tin_feat():
    return MuDMFeature(
        type="Feature",
        id="t1",
        geometry=TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]],
                [[(1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)]],
            ],
        ),
        properties={},
    )


# ===== Shapely -> MuDM Tests =====


class TestShapelyToMuDM:
    def test_point_2d(self):
        s = ShapelyPoint(1, 2)
        m = shapely_to_microjson(s)
        assert isinstance(m, Point)
        assert m.coordinates[0] == 1.0
        assert m.coordinates[1] == 2.0

    def test_point_3d(self):
        s = ShapelyPoint(1, 2, 3)
        m = shapely_to_microjson(s)
        assert isinstance(m, Point)
        assert len(m.coordinates) == 3
        assert m.coordinates[2] == 3.0

    def test_linestring(self):
        s = ShapelyLineString([(0, 0, 0), (1, 1, 1)])
        m = shapely_to_microjson(s)
        assert isinstance(m, LineString)
        assert len(m.coordinates) == 2

    def test_polygon(self):
        s = ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        m = shapely_to_microjson(s)
        assert isinstance(m, Polygon)
        assert len(m.coordinates) == 1  # exterior only

    def test_polygon_with_hole(self):
        s = ShapelyPolygon(
            [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
            [[(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)]],
        )
        m = shapely_to_microjson(s)
        assert isinstance(m, Polygon)
        assert len(m.coordinates) == 2  # exterior + 1 hole

    def test_multipolygon(self):
        s = ShapelyMultiPolygon(
            [
                ShapelyPolygon([(0, 0), (1, 0), (1, 1), (0, 0)]),
                ShapelyPolygon([(5, 5), (6, 5), (6, 6), (5, 5)]),
            ]
        )
        m = shapely_to_microjson(s)
        assert isinstance(m, MultiPolygon)
        assert len(m.coordinates) == 2

    def test_none(self):
        assert shapely_to_microjson(None) is None

    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            shapely_to_microjson("not a geometry")


# ===== Arrow Round-trip Tests =====


class TestArrowRoundTrip:
    def test_point(self):
        orig = _point_feat(props={"count": 42})
        table = to_arrow_table(orig)
        fc = from_arrow_table(table)
        assert len(fc.features) == 1
        f = fc.features[0]
        assert f.id == "p1"
        assert isinstance(f.geometry, Point)
        assert f.geometry.coordinates[2] == 3.0
        assert f.properties["count"] == 42

    def test_polygon(self):
        orig = _polygon_feat()
        table = to_arrow_table(orig)
        fc = from_arrow_table(table)
        f = fc.features[0]
        assert isinstance(f.geometry, Polygon)
        assert f.properties["area"] == 100

    def test_polygon_with_hole(self):
        orig = _polygon_with_hole_feat()
        table = to_arrow_table(orig)
        fc = from_arrow_table(table)
        f = fc.features[0]
        assert isinstance(f.geometry, Polygon)
        assert len(f.geometry.coordinates) == 2  # exterior + hole

    def test_linestring(self):
        orig = _linestring_feat()
        table = to_arrow_table(orig)
        fc = from_arrow_table(table)
        f = fc.features[0]
        assert isinstance(f.geometry, LineString)
        assert len(f.geometry.coordinates) == 3

    def test_collection(self):
        fc_orig = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _point_feat("a", props={"val": 1}),
                _point_feat("b", props={"val": 2}),
            ],
        )
        table = to_arrow_table(fc_orig)
        fc = from_arrow_table(table)
        assert len(fc.features) == 2
        assert fc.features[0].id == "a"
        assert fc.features[1].id == "b"

    def test_null_geometry(self):
        orig = MuDMFeature(
            type="Feature",
            id="null",
            geometry=None,
            properties={"tag": "x"},
        )
        table = to_arrow_table(orig)
        fc = from_arrow_table(table)
        assert fc.features[0].geometry is None

    def test_feature_class_preserved(self):
        orig = MuDMFeature(
            type="Feature",
            id="fc1",
            geometry=Point(type="Point", coordinates=(0, 0)),
            properties={},
            featureClass="neuron",
        )
        table = to_arrow_table(orig)
        fc = from_arrow_table(table)
        assert fc.features[0].featureClass == "neuron"

    def test_tin_roundtrip(self):
        """TIN -> MultiPolygon WKB -> TIN (triangle-only 3D MultiPolygon reconstructed)."""
        orig = _tin_feat()
        table = to_arrow_table(orig)
        fc = from_arrow_table(table)
        f = fc.features[0]
        # TIN stored as MultiPolygon in WKB — reconstructed as TIN on read
        assert isinstance(f.geometry, TIN)
        assert len(f.geometry.coordinates) == 2


# ===== GeoParquet File Round-trip Tests =====


class TestGeoParquetRoundTrip:
    def test_write_read(self, tmp_path):
        path = tmp_path / "roundtrip.parquet"
        fc_orig = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _point_feat("a", props={"val": 1}),
                _polygon_feat("b"),
            ],
        )
        to_geoparquet(fc_orig, path)
        fc = from_geoparquet(path)
        assert len(fc.features) == 2
        assert fc.features[0].id == "a"

    def test_tin_file_roundtrip(self, tmp_path):
        """TIN -> Parquet -> TIN (triangle-only 3D MultiPolygon reconstructed)."""
        path = tmp_path / "tin.parquet"
        to_geoparquet(_tin_feat(), path)
        fc = from_geoparquet(path)
        f = fc.features[0]
        assert isinstance(f.geometry, TIN)
        assert len(f.geometry.coordinates) == 2

    def test_mixed_geometry_file_roundtrip(self, tmp_path):
        path = tmp_path / "mixed.parquet"
        fc_orig = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _point_feat("p"),
                _tin_feat(),
            ],
        )
        to_geoparquet(fc_orig, path)
        fc = from_geoparquet(path)

        assert len(fc.features) == 2
        geom_types = {type(f.geometry).__name__ for f in fc.features}
        assert "Point" in geom_types
        assert "TIN" in geom_types  # TIN reconstructed from triangle-only 3D MultiPolygon

    def test_custom_geometry_column(self, tmp_path):
        path = tmp_path / "custom_col.parquet"
        config = ArrowConfig(primary_geometry_column="geom")
        to_geoparquet(_point_feat(), path, config)
        fc = from_geoparquet(path)
        assert len(fc.features) == 1
        assert isinstance(fc.features[0].geometry, Point)


# ===== End-to-end: Parquet -> GLB =====


class TestParquetToGlb:
    def test_parquet_to_glb_pipeline(self, tmp_path):
        """Full pipeline: write Parquet, read back, export to GLB."""
        from mudm_tools.gltf import GltfConfig, to_glb

        parquet_path = tmp_path / "tin.parquet"

        # Create a TIN feature and write to parquet
        to_geoparquet(_tin_feat(), parquet_path)

        # Read back
        fc = from_geoparquet(parquet_path)
        assert len(fc.features) == 1
        # TIN reconstructed from triangle-only 3D MultiPolygon
        assert isinstance(fc.features[0].geometry, TIN)

        # Export to GLB (with Draco if available)
        try:
            import DracoPy  # noqa: F401

            config = GltfConfig(draco=True)
        except ImportError:
            config = GltfConfig(draco=False)

        glb_bytes = to_glb(fc, config=config)
        assert len(glb_bytes) > 0
