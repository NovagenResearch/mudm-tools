"""Integration tests for Arrow/GeoParquet export: writer API, round-trip, mixed geometry."""

from __future__ import annotations

import json

import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from mudm_tools.arrow import (
    ArrowConfig,
    to_arrow_table,
    to_geoparquet,
    from_arrow_table,
    from_geoparquet,
)
from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    OntologyTerm,
    TIN,
    Vocabulary,
)

from geojson_pydantic import (
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

# ---- Helpers ----


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
        properties={"area": 100.0},
    )


def _tin_feat():
    return MuDMFeature(
        type="Feature",
        id="t1",
        geometry=TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0)]],
            ],
        ),
        properties={},
    )


# ===== Writer API Tests =====


class TestToArrowTable:
    def test_single_feature(self):
        table = to_arrow_table(_point_feat())
        assert isinstance(table, pa.Table)
        assert len(table) == 1

    def test_collection(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[_point_feat("a"), _point_feat("b")],
        )
        table = to_arrow_table(fc)
        assert len(table) == 2

    def test_custom_config(self):
        config = ArrowConfig(primary_geometry_column="geom")
        table = to_arrow_table(_point_feat(), config)
        assert "geom" in table.column_names
        meta = json.loads(table.schema.metadata[b"geo"])
        assert meta["primary_column"] == "geom"

    def test_returns_table_type(self):
        table = to_arrow_table(_point_feat())
        assert isinstance(table, pa.Table)


# ===== GeoParquet Round-trip Tests =====


class TestGeoParquetRoundTrip:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "test.parquet"
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _point_feat("a", props={"val": 1}),
                _point_feat("b", props={"val": 2}),
            ],
        )
        to_geoparquet(fc, path)

        # Verify file exists
        assert path.exists()
        assert path.stat().st_size > 0

        # Read back with pyarrow
        read_table = pq.read_table(str(path))
        assert len(read_table) == 2
        assert "geometry" in read_table.column_names

        # Verify GeoParquet metadata preserved
        meta = json.loads(read_table.schema.metadata[b"geo"])
        assert meta["version"] == "1.1.0"

    def test_geopandas_read(self, tmp_path):
        """Verify geopandas can read the GeoParquet file."""
        path = tmp_path / "geopandas_test.parquet"
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _polygon_feat("p1"),
                _polygon_feat("p2"),
            ],
        )
        to_geoparquet(fc, path)

        gdf = gpd.read_parquet(str(path))
        assert len(gdf) == 2
        assert "geometry" in gdf.columns
        assert gdf.geometry.iloc[0].geom_type == "Polygon"

    def test_3d_point_roundtrip(self, tmp_path):
        path = tmp_path / "3d_points.parquet"
        to_geoparquet(_point_feat(), path)

        read_table = pq.read_table(str(path))
        wkb = read_table["geometry"][0].as_py()
        geom = shapely.from_wkb(wkb)
        assert geom.has_z
        assert geom.z == 3.0

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "test.parquet"
        to_geoparquet(_point_feat(), path)
        assert path.exists()


# ===== Mixed Geometry Tests =====


class TestMixedGeometry:
    def test_point_and_polygon(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[_point_feat(), _polygon_feat()],
        )
        table = to_arrow_table(fc)
        assert len(table) == 2
        meta = json.loads(table.schema.metadata[b"geo"])
        types = meta["columns"]["geometry"]["geometry_types"]
        assert "Point Z" in types
        assert "Polygon" in types

    def test_tin_and_point(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[_tin_feat(), _point_feat()],
        )
        table = to_arrow_table(fc)
        assert len(table) == 2
        meta = json.loads(table.schema.metadata[b"geo"])
        types = meta["columns"]["geometry"]["geometry_types"]
        assert "Point Z" in types
        assert "MultiPolygon Z" in types  # TIN stored as MultiPolygon

    def test_all_3d_types(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                _point_feat(),
                _tin_feat(),
            ],
        )
        table = to_arrow_table(fc)
        assert len(table) == 2
        meta = json.loads(table.schema.metadata[b"geo"])
        types = meta["columns"]["geometry"]["geometry_types"]
        assert "Point Z" in types
        assert "MultiPolygon Z" in types  # TIN stored as MultiPolygon

    def test_mixed_geometry_geoparquet(self, tmp_path):
        """Write mixed geometry types to GeoParquet and verify metadata."""
        path = tmp_path / "mixed.parquet"
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[_point_feat(), _polygon_feat(), _tin_feat()],
        )
        to_geoparquet(fc, path)

        read_table = pq.read_table(str(path))
        meta = json.loads(read_table.schema.metadata[b"geo"])
        types = meta["columns"]["geometry"]["geometry_types"]
        assert len(types) >= 3


# ===== Edge Cases =====


class TestEdgeCases:
    def test_empty_collection(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[],
        )
        table = to_arrow_table(fc)
        assert len(table) == 0

    def test_no_properties(self):
        feat = MuDMFeature(
            type="Feature",
            id="noprops",
            geometry=Point(type="Point", coordinates=(0, 0)),
            properties={},
        )
        table = to_arrow_table(feat)
        # Should have at least id, featureClass, geometry
        assert len(table.column_names) >= 3

    def test_null_geometry_feature(self):
        feat = MuDMFeature(
            type="Feature",
            id="nullgeo",
            geometry=None,
            properties={"tag": "test"},
        )
        table = to_arrow_table(feat)
        assert table["geometry"][0].as_py() is None

    def test_multipoint_3d(self):
        feat = MuDMFeature(
            type="Feature",
            id="mp1",
            geometry=MultiPoint(
                type="MultiPoint",
                coordinates=[(0, 0, 0), (1, 1, 1), (2, 2, 2)],
            ),
            properties={},
        )
        table = to_arrow_table(feat)
        wkb = table["geometry"][0].as_py()
        geom = shapely.from_wkb(wkb)
        assert geom.geom_type == "MultiPoint"
        assert shapely.get_coordinate_dimension(geom) == 3

    def test_multilinestring_geometry(self):
        feat = MuDMFeature(
            type="Feature",
            id="mls1",
            geometry=MultiLineString(
                type="MultiLineString",
                coordinates=[
                    [(0, 0, 0), (1, 1, 1)],
                    [(2, 2, 2), (3, 3, 3)],
                ],
            ),
            properties={},
        )
        table = to_arrow_table(feat)
        wkb = table["geometry"][0].as_py()
        geom = shapely.from_wkb(wkb)
        assert geom.geom_type == "MultiLineString"

    def test_multipolygon_geometry(self):
        feat = MuDMFeature(
            type="Feature",
            id="mpg1",
            geometry=MultiPolygon(
                type="MultiPolygon",
                coordinates=[
                    [[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]],
                    [[(5, 5), (6, 5), (6, 6), (5, 6), (5, 5)]],
                ],
            ),
            properties={},
        )
        table = to_arrow_table(feat)
        wkb = table["geometry"][0].as_py()
        geom = shapely.from_wkb(wkb)
        assert geom.geom_type == "MultiPolygon"
        assert len(geom.geoms) == 2


# ===== Vocabulary Arrow Round-Trip =====


class TestVocabularyArrowRoundTrip:
    def _make_collection_with_vocab(self):
        return MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    id="v1",
                    geometry=Point(type="Point", coordinates=(1.0, 2.0)),
                    properties={"cell_type": "pyramidal"},
                ),
            ],
            vocabularies={
                "cell_type": Vocabulary(
                    namespace="http://purl.obolibrary.org/obo/CL_",
                    terms={
                        "pyramidal": OntologyTerm(
                            uri="http://purl.obolibrary.org/obo/CL_0000598",
                            label="pyramidal neuron",
                        ),
                    },
                ),
            },
        )

    def test_vocabulary_in_table_metadata(self):
        fc = self._make_collection_with_vocab()
        table = to_arrow_table(fc)
        raw = table.schema.metadata.get(b"mudm:vocabularies")
        assert raw is not None
        vocab = json.loads(raw)
        assert "cell_type" in vocab
        assert (
            vocab["cell_type"]["terms"]["pyramidal"]["uri"]
            == "http://purl.obolibrary.org/obo/CL_0000598"
        )

    def test_vocabulary_arrow_roundtrip(self):
        fc = self._make_collection_with_vocab()
        table = to_arrow_table(fc)
        fc2 = from_arrow_table(table)
        assert fc2.vocabularies is not None
        assert "cell_type" in fc2.vocabularies
        assert (
            fc2.vocabularies["cell_type"].terms["pyramidal"].uri
            == "http://purl.obolibrary.org/obo/CL_0000598"
        )
        assert fc2.vocabularies["cell_type"].terms["pyramidal"].label == "pyramidal neuron"

    def test_vocabulary_geoparquet_roundtrip(self, tmp_path):
        fc = self._make_collection_with_vocab()
        path = tmp_path / "vocab.parquet"
        to_geoparquet(fc, path)
        fc2 = from_geoparquet(path)
        assert fc2.vocabularies is not None
        assert "cell_type" in fc2.vocabularies
        assert fc2.vocabularies["cell_type"].namespace == "http://purl.obolibrary.org/obo/CL_"

    def test_uri_reference_roundtrip(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[
                MuDMFeature(
                    type="Feature",
                    id="u1",
                    geometry=Point(type="Point", coordinates=(0.0, 0.0)),
                    properties={},
                ),
            ],
            vocabularies="https://neuromorpho.org/vocab/neuroscience-v1.json",
        )
        table = to_arrow_table(fc)
        fc2 = from_arrow_table(table)
        assert fc2.vocabularies == "https://neuromorpho.org/vocab/neuroscience-v1.json"

    def test_no_vocabularies_metadata_absent(self):
        fc = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[_point_feat()],
        )
        table = to_arrow_table(fc)
        assert b"mudm:vocabularies" not in table.schema.metadata
        fc2 = from_arrow_table(table)
        assert fc2.vocabularies is None
