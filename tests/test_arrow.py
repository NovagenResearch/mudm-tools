"""Unit tests for Arrow export: geometry conversion, schema inference, table builder."""

from __future__ import annotations

import json

import pyarrow as pa
import pytest
import shapely
from shapely.geometry import (
    LineString as ShapelyLineString,
    MultiLineString as ShapelyMultiLineString,
    MultiPoint as ShapelyMultiPoint,
    MultiPolygon as ShapelyMultiPolygon,
    Point as ShapelyPoint,
    Polygon as ShapelyPolygon,
)

from mudm_tools.arrow._geometry import (
    geometry_type_name,
    linestring_to_shapely,
    multilinestring_to_shapely,
    multipoint_to_shapely,
    multipolygon_to_shapely,
    point_to_shapely,
    polyhedral_to_shapely,
    polygon_to_shapely,
    tin_to_shapely,
    to_shapely,
    to_wkb,
)
from mudm_tools.arrow._table_builder import _infer_pa_type, build_table
from mudm_tools.arrow.models import ArrowConfig
from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    PolyhedralSurface,
    TIN,
)

from geojson_pydantic import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


# ---- Fixtures ----

def _point_feature(x=1.0, y=2.0, z=3.0, fid="f1", props=None):
    return MuDMFeature(
        type="Feature",
        id=fid,
        geometry=Point(type="Point", coordinates=(x, y, z)),
        properties=props or {},
    )


def _linestring_feature():
    return MuDMFeature(
        type="Feature",
        id="ls1",
        geometry=LineString(
            type="LineString",
            coordinates=[(0, 0, 0), (1, 1, 1), (2, 0, 2)],
        ),
        properties={"name": "line1"},
    )


def _polygon_feature():
    return MuDMFeature(
        type="Feature",
        id="pg1",
        geometry=Polygon(
            type="Polygon",
            coordinates=[[(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]],
        ),
        properties={"area": 100},
    )


def _tin_feature():
    return MuDMFeature(
        type="Feature",
        id="tin1",
        geometry=TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]],
                [[(1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 0, 0)]],
            ],
        ),
        properties={},
    )


def _polyhedral_feature():
    return MuDMFeature(
        type="Feature",
        id="ph1",
        geometry=PolyhedralSurface(
            type="PolyhedralSurface",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)]],
            ],
        ),
        properties={},
    )


# ===== Geometry Conversion Tests =====


class TestGeometryConversion:
    def test_point_2d(self):
        geom = Point(type="Point", coordinates=(1, 2))
        s = point_to_shapely(geom)
        assert isinstance(s, ShapelyPoint)
        assert s.x == 1 and s.y == 2

    def test_point_3d(self):
        geom = Point(type="Point", coordinates=(1, 2, 3))
        s = point_to_shapely(geom)
        assert s.has_z
        assert s.z == 3

    def test_multipoint(self):
        geom = MultiPoint(
            type="MultiPoint", coordinates=[(0, 0, 1), (1, 1, 2)]
        )
        s = multipoint_to_shapely(geom)
        assert isinstance(s, ShapelyMultiPoint)
        assert len(s.geoms) == 2

    def test_linestring(self):
        geom = LineString(
            type="LineString", coordinates=[(0, 0), (1, 1), (2, 0)]
        )
        s = linestring_to_shapely(geom)
        assert isinstance(s, ShapelyLineString)
        assert len(s.coords) == 3

    def test_multilinestring(self):
        geom = MultiLineString(
            type="MultiLineString",
            coordinates=[[(0, 0), (1, 1)], [(2, 2), (3, 3)]],
        )
        s = multilinestring_to_shapely(geom)
        assert isinstance(s, ShapelyMultiLineString)
        assert len(s.geoms) == 2

    def test_polygon(self):
        geom = Polygon(
            type="Polygon",
            coordinates=[[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
        )
        s = polygon_to_shapely(geom)
        assert isinstance(s, ShapelyPolygon)
        assert not s.is_empty

    def test_polygon_with_hole(self):
        geom = Polygon(
            type="Polygon",
            coordinates=[
                [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)],
                [(2, 2), (8, 2), (8, 8), (2, 8), (2, 2)],
            ],
        )
        s = polygon_to_shapely(geom)
        assert len(list(s.interiors)) == 1

    def test_multipolygon(self):
        geom = MultiPolygon(
            type="MultiPolygon",
            coordinates=[
                [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
                [[(2, 2), (3, 2), (3, 3), (2, 3), (2, 2)]],
            ],
        )
        s = multipolygon_to_shapely(geom)
        assert isinstance(s, ShapelyMultiPolygon)
        assert len(s.geoms) == 2

    def test_tin(self):
        feat = _tin_feature()
        s = tin_to_shapely(feat.geometry)
        assert isinstance(s, ShapelyMultiPolygon)
        assert len(s.geoms) == 2

    def test_polyhedral_surface(self):
        feat = _polyhedral_feature()
        s = polyhedral_to_shapely(feat.geometry)
        assert isinstance(s, ShapelyMultiPolygon)
        assert len(s.geoms) == 1

    def test_none_geometry(self):
        assert to_shapely(None) is None

    def test_unsupported_raises(self):
        with pytest.raises(TypeError, match="Unsupported"):
            to_shapely("not a geometry")


# ===== WKB Tests =====


class TestWKB:
    def test_to_wkb_point(self):
        s = ShapelyPoint(1, 2, 3)
        wkb = to_wkb(s)
        assert isinstance(wkb, bytes)
        assert len(wkb) > 0
        # Round-trip
        restored = shapely.from_wkb(wkb)
        assert restored.equals(s)

    def test_to_wkb_none(self):
        assert to_wkb(None) is None


# ===== Geometry Type Name Tests =====


class TestGeometryTypeName:
    def test_2d_point(self):
        s = ShapelyPoint(1, 2)
        assert geometry_type_name(s) == "Point"

    def test_3d_point(self):
        s = ShapelyPoint(1, 2, 3)
        assert geometry_type_name(s) == "Point Z"

    def test_3d_multilinestring(self):
        s = ShapelyMultiLineString([[(0, 0, 0), (1, 1, 1)]])
        assert geometry_type_name(s) == "MultiLineString Z"

    def test_none(self):
        assert geometry_type_name(None) is None


# ===== Type Inference Tests =====


class TestTypeInference:
    def test_empty_values(self):
        assert _infer_pa_type([]) == pa.string()

    def test_all_int(self):
        assert _infer_pa_type([1, 2, 3]) == pa.int64()

    def test_all_float(self):
        assert _infer_pa_type([1.0, 2.5]) == pa.float64()

    def test_mixed_int_float(self):
        assert _infer_pa_type([1, 2.5, 3]) == pa.float64()

    def test_all_str(self):
        assert _infer_pa_type(["a", "b"]) == pa.string()

    def test_all_bool(self):
        assert _infer_pa_type([True, False]) == pa.bool_()

    def test_mixed_types(self):
        """Mixed types -> JSON string."""
        assert _infer_pa_type([1, "a"]) == pa.string()

    def test_dict_values(self):
        assert _infer_pa_type([{"a": 1}]) == pa.string()


# ===== Table Builder Tests =====


class TestTableBuilder:
    def test_single_point_feature(self):
        feat = _point_feature(props={"count": 42})
        table = build_table(feat)
        assert isinstance(table, pa.Table)
        assert len(table) == 1
        assert "id" in table.column_names
        assert "geometry" in table.column_names
        assert "count" in table.column_names
        assert table["id"][0].as_py() == "f1"
        assert table["count"][0].as_py() == 42

    def test_collection(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _point_feature(fid="a", props={"x": 1}),
                _point_feature(fid="b", props={"x": 2}),
            ],
        )
        table = build_table(fc)
        assert len(table) == 2
        assert table["id"][0].as_py() == "a"
        assert table["id"][1].as_py() == "b"

    def test_tin_geometry(self):
        table = build_table(_tin_feature())
        assert len(table) == 1
        wkb = table["geometry"][0].as_py()
        geom = shapely.from_wkb(wkb)
        assert geom.geom_type == "MultiPolygon"

    def test_polyhedral_geometry(self):
        table = build_table(_polyhedral_feature())
        assert len(table) == 1
        wkb = table["geometry"][0].as_py()
        geom = shapely.from_wkb(wkb)
        assert geom.geom_type == "MultiPolygon"

    def test_none_geometry(self):
        feat = MuDMFeature(
            type="Feature",
            id="null",
            geometry=None,
            properties={"tag": "empty"},
        )
        table = build_table(feat)
        assert len(table) == 1
        assert table["geometry"][0].as_py() is None

    def test_geoparquet_metadata(self):
        table = build_table(_point_feature())
        meta = json.loads(table.schema.metadata[b"geo"])
        assert meta["version"] == "1.1.0"
        assert meta["primary_column"] == "geometry"
        assert "Point Z" in meta["columns"]["geometry"]["geometry_types"]
        assert len(meta["columns"]["geometry"]["bbox"]) == 4

    def test_mixed_property_types(self):
        """Properties with mixed types -> JSON-serialized strings."""
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _point_feature(fid="a", props={"val": 1}),
                _point_feature(fid="b", props={"val": "str"}),
            ],
        )
        table = build_table(fc)
        # Mixed int/str -> string column
        assert table.schema.field("val").type == pa.string()

    def test_feature_class_column(self):
        feat = MuDMFeature(
            type="Feature",
            id="fc1",
            geometry=Point(type="Point", coordinates=(0, 0)),
            properties={},
            featureClass="neuron",
        )
        table = build_table(feat)
        assert table["featureClass"][0].as_py() == "neuron"

    def test_nullable_id(self):
        feat = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=(0, 0)),
            properties={},
        )
        table = build_table(feat)
        assert table["id"][0].as_py() is None

    def test_wkb_roundtrip(self):
        feat = _linestring_feature()
        table = build_table(feat)
        wkb = table["geometry"][0].as_py()
        geom = shapely.from_wkb(wkb)
        assert geom.geom_type == "LineString"
        coords = list(geom.coords)
        assert coords[0] == (0.0, 0.0, 0.0)
        assert coords[2] == (2.0, 0.0, 2.0)

    def test_property_with_dict_value(self):
        """Dict properties -> JSON-serialized string."""
        feat = _point_feature(props={"meta": {"key": "val"}})
        table = build_table(feat)
        val = table["meta"][0].as_py()
        assert json.loads(val) == {"key": "val"}

    def test_bool_property(self):
        feat = _point_feature(props={"flag": True})
        table = build_table(feat)
        assert table.schema.field("flag").type == pa.bool_()
        assert table["flag"][0].as_py() is True
