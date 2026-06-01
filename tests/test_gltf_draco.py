"""Tests for Draco mesh compression (KHR_draco_mesh_compression)."""

import struct

import numpy as np
import pytest

DracoPy = pytest.importorskip("DracoPy")

from pygltflib import LINES, POINTS, TRIANGLES  # noqa: E402

from mudm.model import (  # noqa: E402
    MuDMFeature,
    MuDMFeatureCollection,
    TIN,
)
from mudm_tools.gltf.gltf_assembler import (  # noqa: E402
    collection_to_gltf,
    feature_to_gltf,
)
from mudm_tools.gltf.models import GltfConfig  # noqa: E402
from mudm_tools.gltf.writer import to_glb  # noqa: E402
from mudm_tools.gltf._draco import (  # noqa: E402
    DRACO_EXTENSION,
    encode_draco,
    ensure_draco_extensions,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tin_feature(**props) -> MuDMFeature:
    """A TIN feature with several triangles for meaningful compression tests."""
    return MuDMFeature(
        type="Feature",
        geometry=TIN(
            type="TIN",
            coordinates=[
                [[(0, 0, 0), (10, 0, 0), (5, 10, 5), (0, 0, 0)]],
                [[(10, 0, 0), (20, 0, 0), (15, 10, 5), (10, 0, 0)]],
                [[(0, 0, 0), (5, 10, 5), (0, 10, 0), (0, 0, 0)]],
                [[(10, 0, 0), (15, 10, 5), (5, 10, 5), (10, 0, 0)]],
            ],
        ),
        properties=props or None,
    )


def _line_feature() -> MuDMFeature:
    from geojson_pydantic import LineString

    return MuDMFeature(
        type="Feature",
        geometry=LineString(
            type="LineString",
            coordinates=[[0, 0, 0], [10, 10, 10], [20, 0, 0]],
        ),
        properties=None,
    )


def _point_feature() -> MuDMFeature:
    from geojson_pydantic import Point

    return MuDMFeature(
        type="Feature",
        geometry=Point(type="Point", coordinates=[5.0, 5.0, 5.0]),
        properties=None,
    )


def _polygon_feature() -> MuDMFeature:
    from geojson_pydantic import Polygon

    return MuDMFeature(
        type="Feature",
        geometry=Polygon(
            type="Polygon",
            coordinates=[[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
        ),
        properties=None,
    )


# ---------------------------------------------------------------------------
# TestDracoCompression
# ---------------------------------------------------------------------------


class TestDracoCompression:
    """Compressed GLB is smaller, valid GLB header, vertex count preserved."""

    def test_compressed_glb_smaller(self):
        feat = _tin_feature()
        config_std = GltfConfig(draco=False)
        config_draco = GltfConfig(draco=True)

        glb_std = to_glb(feat, config=config_std)
        glb_draco = to_glb(feat, config=config_draco)

        assert len(glb_draco) < len(glb_std), (
            f"Draco GLB ({len(glb_draco)}) should be smaller than " f"standard GLB ({len(glb_std)})"
        )

    def test_valid_glb_header(self):
        feat = _tin_feature()
        glb = to_glb(feat, config=GltfConfig(draco=True))

        # GLB magic: 0x46546C67 ('glTF')
        magic = struct.unpack_from("<I", glb, 0)[0]
        assert magic == 0x46546C67

        # Version: 2
        version = struct.unpack_from("<I", glb, 4)[0]
        assert version == 2

        # Total length matches actual size
        length = struct.unpack_from("<I", glb, 8)[0]
        assert length == len(glb)

    def test_vertex_count_preserved(self):
        feat = _tin_feature()
        config_std = GltfConfig(draco=False)
        config_draco = GltfConfig(draco=True)

        gltf_std = feature_to_gltf(feat, config=config_std)
        gltf_draco = feature_to_gltf(feat, config=config_draco)

        # Both should have same number of meshes
        assert len(gltf_draco.meshes) == len(gltf_std.meshes)

        # Sum POSITION accessor counts across all meshes
        def _total_verts(gltf):
            total = 0
            for mesh in gltf.meshes:
                for prim in mesh.primitives:
                    pos_idx = prim.attributes.POSITION
                    total += gltf.accessors[pos_idx].count
            return total

        assert _total_verts(gltf_draco) == _total_verts(gltf_std)

    def test_encode_draco_returns_bytes(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        config = GltfConfig(draco=True)

        result = encode_draco(verts, indices, normals, config)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_encode_draco_without_normals(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.uint32)
        config = GltfConfig(draco=True)

        result = encode_draco(verts, indices, None, config)
        assert isinstance(result, bytes)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# TestDracoExtensionStructure
# ---------------------------------------------------------------------------


class TestDracoExtensionStructure:
    """Extension on root + primitives, stub accessors, Draco BufferView."""

    def test_extension_on_root(self):
        feat = _tin_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        assert DRACO_EXTENSION in gltf.extensionsUsed
        assert DRACO_EXTENSION in gltf.extensionsRequired

    def test_extension_on_primitives(self):
        feat = _tin_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.mode == TRIANGLES:
                    assert prim.extensions is not None
                    assert DRACO_EXTENSION in prim.extensions
                    ext = prim.extensions[DRACO_EXTENSION]
                    assert "bufferView" in ext
                    assert "attributes" in ext
                    assert "POSITION" in ext["attributes"]

    def test_stub_accessors_have_no_bufferview(self):
        feat = _tin_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.mode != TRIANGLES:
                    continue
                # POSITION accessor -- no bufferView or byteOffset
                pos_acc = gltf.accessors[prim.attributes.POSITION]
                assert pos_acc.bufferView is None
                assert pos_acc.byteOffset is None

                # Indices accessor
                idx_acc = gltf.accessors[prim.indices]
                assert idx_acc.bufferView is None
                assert idx_acc.byteOffset is None

                # NORMAL accessor (if present)
                if prim.attributes.NORMAL is not None:
                    norm_acc = gltf.accessors[prim.attributes.NORMAL]
                    assert norm_acc.bufferView is None
                    assert norm_acc.byteOffset is None

    def test_position_accessor_has_min_max(self):
        feat = _tin_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.mode != TRIANGLES:
                    continue
                pos_acc = gltf.accessors[prim.attributes.POSITION]
                assert pos_acc.max is not None
                assert pos_acc.min is not None
                assert len(pos_acc.max) == 3
                assert len(pos_acc.min) == 3

    def test_draco_bufferview_has_no_target(self):
        feat = _tin_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.mode != TRIANGLES:
                    continue
                ext = prim.extensions[DRACO_EXTENSION]
                bv_idx = ext["bufferView"]
                bv = gltf.bufferViews[bv_idx]
                assert bv.target is None

    def test_ensure_draco_extensions_idempotent(self):
        from pygltflib import GLTF2

        gltf = GLTF2()
        ensure_draco_extensions(gltf)
        ensure_draco_extensions(gltf)  # call again

        assert gltf.extensionsUsed.count(DRACO_EXTENSION) == 1
        assert gltf.extensionsRequired.count(DRACO_EXTENSION) == 1


# ---------------------------------------------------------------------------
# TestDracoNonTriangle
# ---------------------------------------------------------------------------


class TestDracoNonTriangle:
    """Lines and points are NOT compressed even with draco=True."""

    def test_lines_not_compressed(self):
        feat = _line_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        assert len(gltf.meshes) == 1
        prim = gltf.meshes[0].primitives[0]
        assert prim.mode == LINES
        assert prim.extensions is None or DRACO_EXTENSION not in (prim.extensions or {})

        # Standard accessor should have bufferView
        pos_acc = gltf.accessors[prim.attributes.POSITION]
        assert pos_acc.bufferView is not None

    def test_points_not_compressed(self):
        feat = _point_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        assert len(gltf.meshes) == 1
        prim = gltf.meshes[0].primitives[0]
        assert prim.mode == POINTS
        assert prim.extensions is None or DRACO_EXTENSION not in (prim.extensions or {})

        pos_acc = gltf.accessors[prim.attributes.POSITION]
        assert pos_acc.bufferView is not None


# ---------------------------------------------------------------------------
# TestDracoMixedGeometry
# ---------------------------------------------------------------------------


class TestDracoMixedGeometry:
    """Collection with TINs + lines -- only triangles compressed."""

    def test_mixed_collection(self):
        tin_feat = _tin_feature()
        line_feat = _line_feature()

        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[tin_feat, line_feat],
        )
        config = GltfConfig(draco=True)
        gltf = collection_to_gltf(coll, config=config)

        # Should have meshes for TIN (triangle) + line
        triangle_meshes = []
        line_meshes = []
        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.mode == TRIANGLES:
                    triangle_meshes.append(prim)
                elif prim.mode == LINES:
                    line_meshes.append(prim)

        assert len(triangle_meshes) > 0
        assert len(line_meshes) > 0

        # Triangle primitives have Draco extension
        for prim in triangle_meshes:
            assert prim.extensions is not None
            assert DRACO_EXTENSION in prim.extensions

        # Line primitives do NOT have Draco extension
        for prim in line_meshes:
            assert prim.extensions is None or DRACO_EXTENSION not in (prim.extensions or {})

    def test_mixed_with_points(self):
        tin_feat = _tin_feature()
        point_feat = _point_feature()

        coll = MuDMFeatureCollection(
            type="FeatureCollection",
            features=[tin_feat, point_feat],
        )
        config = GltfConfig(draco=True)
        gltf = collection_to_gltf(coll, config=config)

        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.mode == TRIANGLES:
                    assert prim.extensions is not None
                    assert DRACO_EXTENSION in prim.extensions
                elif prim.mode == POINTS:
                    assert prim.extensions is None or DRACO_EXTENSION not in (prim.extensions or {})

    def test_polygon_compressed(self):
        """Polygon meshes (triangulated) should also be Draco-compressed."""
        feat = _polygon_feature()
        gltf = feature_to_gltf(feat, config=GltfConfig(draco=True))

        assert len(gltf.meshes) >= 1
        prim = gltf.meshes[0].primitives[0]
        assert prim.mode == TRIANGLES
        assert prim.extensions is not None
        assert DRACO_EXTENSION in prim.extensions


# ---------------------------------------------------------------------------
# TestDracoConfig
# ---------------------------------------------------------------------------


class TestDracoConfig:
    """Custom quantization works, draco=False by default, draco=False -> standard GLB."""

    def test_draco_false_by_default(self):
        config = GltfConfig()
        assert config.draco is False

    def test_draco_false_produces_standard_glb(self):
        feat = _tin_feature()
        config = GltfConfig(draco=False)
        gltf = feature_to_gltf(feat, config=config)

        # No Draco extension on root
        assert not gltf.extensionsUsed
        assert not gltf.extensionsRequired

        # All TRIANGLES primitives have standard accessors (with bufferView)
        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.mode == TRIANGLES:
                    assert prim.extensions is None or DRACO_EXTENSION not in (prim.extensions or {})
                    pos_acc = gltf.accessors[prim.attributes.POSITION]
                    assert pos_acc.bufferView is not None

    def test_custom_quantization(self):
        feat = _tin_feature()
        config_low = GltfConfig(draco=True, draco_quantization_position=8)
        config_high = GltfConfig(draco=True, draco_quantization_position=20)

        glb_low = to_glb(feat, config=config_low)
        glb_high = to_glb(feat, config=config_high)

        # Both should still be valid GLBs
        assert struct.unpack_from("<I", glb_low, 0)[0] == 0x46546C67
        assert struct.unpack_from("<I", glb_high, 0)[0] == 0x46546C67

    def test_custom_compression_level(self):
        feat = _tin_feature()
        config = GltfConfig(draco=True, draco_compression_level=10)
        glb = to_glb(feat, config=config)

        # Should produce valid output
        assert struct.unpack_from("<I", glb, 0)[0] == 0x46546C67
        assert len(glb) > 0

    def test_config_defaults(self):
        config = GltfConfig()
        assert config.draco_quantization_position == 14
        assert config.draco_quantization_normal == 10
        assert config.draco_compression_level == 1
