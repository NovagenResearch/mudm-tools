"""End-to-end integration tests for glTF/GLB export pipeline."""

import struct
from pathlib import Path

from pygltflib import GLTF2

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    TIN,
)
from mudm_tools.swc import swc_to_tin
from mudm_tools.gltf import to_glb, to_gltf

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SWC_FILE = FIXTURE_DIR / "sample_neuron.swc"


class TestSWCToGLB:
    """SWC file -> swc_to_tin -> to_glb -> validate."""

    def test_swc_to_glb_pipeline(self, tmp_path):
        feat = swc_to_tin(str(SWC_FILE))
        assert feat.geometry.type == "TIN"

        glb_bytes = to_glb(feat)
        assert glb_bytes[:4] == b"glTF"
        version = struct.unpack_from("<I", glb_bytes, 4)[0]
        assert version == 2

    def test_swc_to_glb_roundtrip(self, tmp_path):
        feat = swc_to_tin(str(SWC_FILE))
        out = tmp_path / "neuron.glb"
        to_glb(feat, output_path=out)

        loaded = GLTF2.load(str(out))
        assert len(loaded.meshes) > 0
        assert loaded.asset.version == "2.0"
        assert loaded.asset.generator == "mudm-gltf"

    def test_swc_to_gltf_roundtrip(self, tmp_path):
        feat = swc_to_tin(str(SWC_FILE))
        out = tmp_path / "neuron.gltf"
        to_gltf(feat, output_path=out)

        loaded = GLTF2.load(str(out))
        assert len(loaded.meshes) > 0

    def test_swc_collection_to_glb(self, tmp_path):
        feat = swc_to_tin(str(SWC_FILE))
        collection = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat],
            properties={"source": "test"},
        )
        glb = to_glb(collection)
        assert glb[:4] == b"glTF"

        out = tmp_path / "collection.glb"
        to_glb(collection, output_path=out)
        loaded = GLTF2.load(str(out))
        assert loaded.scenes[0].extras["source"] == "test"


class TestMixedGeometryCollection:
    """Test a collection with multiple geometry types."""

    def test_mixed_collection(self, tmp_path):
        from geojson_pydantic import LineString, Point, Polygon

        point = Point(type="Point", coordinates=[1.0, 2.0, 3.0])
        line = LineString(type="LineString", coordinates=[[0, 0, 0], [5, 5, 5]])
        poly = Polygon(
            type="Polygon",
            coordinates=[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
        )
        tin = TIN(
            type="TIN",
            coordinates=[[[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]]]],
        )

        features = [
            MuDMFeature(type="Feature", geometry=point, properties={"kind": "point"}),
            MuDMFeature(type="Feature", geometry=line, properties={"kind": "line"}),
            MuDMFeature(type="Feature", geometry=poly, properties={"kind": "polygon"}),
            MuDMFeature(type="Feature", geometry=tin, properties={"kind": "tin"}),
        ]
        collection = MuDMFeatureCollection(
            type="FeatureCollection",
            features=features,
        )

        out = tmp_path / "mixed.glb"
        to_glb(collection, output_path=out)

        loaded = GLTF2.load(str(out))
        # point(1) + line(1) + polygon(1) + tin(1) = 4 meshes
        assert len(loaded.meshes) == 4
        assert len(loaded.nodes) == 4


class TestPublicAPIImports:
    """Verify the public API is importable from the top-level package."""

    def test_import_from_microjson(self):
        from mudm_tools import to_gltf, to_glb, GltfConfig

        assert callable(to_gltf)
        assert callable(to_glb)
        assert GltfConfig is not None

    def test_import_from_gltf_subpackage(self):
        from mudm_tools.gltf import to_gltf, to_glb

        assert callable(to_gltf)
        assert callable(to_glb)
