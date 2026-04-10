"""Tests for glTF/GLB writer and public API."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from pygltflib import GLTF2

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    TIN,
)
from mudm_tools.gltf import GltfConfig, to_glb, to_gltf


def _tin_feature(**props) -> MuDMFeature:
    """A simple TIN feature (2 triangles)."""
    return MuDMFeature(
        type="Feature",
        geometry=TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (1, 0, 0), (0.5, 1, 0), (0, 0, 0)]],
                [[(1, 0, 0), (1, 1, 0), (0.5, 1, 0), (1, 0, 0)]],
            ],
        ),
        properties=props or None,
    )


class TestToGltf:
    def test_returns_gltf2(self):
        feat = _tin_feature()
        result = to_gltf(feat)
        assert isinstance(result, GLTF2)
        assert len(result.meshes) > 0

    def test_saves_to_file(self, tmp_path):
        feat = _tin_feature()
        out = tmp_path / "test.gltf"
        result = to_gltf(feat, output_path=out)

        assert out.exists()
        assert out.stat().st_size > 0
        # Should be valid JSON
        import json
        data = json.loads(out.read_text())
        assert data["asset"]["version"] == "2.0"

    def test_config_passthrough(self):
        feat = _tin_feature()
        config = GltfConfig(y_up=False)
        result = to_gltf(feat, config=config)
        assert isinstance(result, GLTF2)


class TestToGlb:
    def test_returns_bytes(self):
        feat = _tin_feature()
        result = to_glb(feat)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_glb_magic_header(self):
        feat = _tin_feature()
        result = to_glb(feat)
        # GLB magic: 0x46546C67 = "glTF"
        assert result[:4] == b"glTF"

    def test_glb_version(self):
        feat = _tin_feature()
        result = to_glb(feat)
        import struct
        version = struct.unpack_from("<I", result, 4)[0]
        assert version == 2

    def test_saves_to_file(self, tmp_path):
        feat = _tin_feature()
        out = tmp_path / "test.glb"
        result = to_glb(feat, output_path=out)

        assert out.exists()
        assert out.read_bytes()[:4] == b"glTF"
        assert result == out.read_bytes()

    def test_creates_parent_dirs(self, tmp_path):
        feat = _tin_feature()
        out = tmp_path / "nested" / "dir" / "test.glb"
        to_glb(feat, output_path=out)
        assert out.exists()

    def test_roundtrip_with_pygltflib(self, tmp_path):
        feat = _tin_feature(cell_type="pyramidal")
        out = tmp_path / "roundtrip.glb"
        to_glb(feat, output_path=out)

        loaded = GLTF2.load(str(out))
        assert len(loaded.meshes) > 0
        assert loaded.asset.version == "2.0"

    def test_collection_glb(self):
        from geojson_pydantic import Point

        feat1 = _tin_feature()
        feat2 = MuDMFeature(
            type="Feature",
            geometry=Point(type="Point", coordinates=[5.0, 5.0, 5.0]),
            properties=None,
        )
        collection = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[feat1, feat2],
        )
        result = to_glb(collection)
        assert result[:4] == b"glTF"

    def test_metadata_preserved_in_glb(self, tmp_path):
        feat = _tin_feature(cell_type="pyramidal", layer=5)
        out = tmp_path / "meta.glb"
        to_glb(feat, output_path=out)

        loaded = GLTF2.load(str(out))
        assert loaded.nodes[0].extras["cell_type"] == "pyramidal"
        assert loaded.nodes[0].extras["layer"] == 5
