"""Tests for Neuroglancer skeleton writer."""

import json
import struct

import pytest

from mudm_tools.swc import NeuronMorphology, SWCSample
from mudm.transforms import AffineTransform
from mudm_tools.neuroglancer.skeleton_writer import (
    affine_to_ng_transform,
    build_skeleton_info,
    neuron_to_skeleton_binary,
    write_skeleton,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_neuron():
    """A 3-node neuron: root (soma) -> child1 (axon) -> child2 (axon)."""
    return NeuronMorphology(
        type="NeuronMorphology",
        tree=[
            SWCSample(id=1, type=1, x=0.0, y=0.0, z=0.0, r=5.0, parent=-1),
            SWCSample(id=2, type=2, x=10.0, y=0.0, z=0.0, r=2.0, parent=1),
            SWCSample(id=3, type=2, x=20.0, y=5.0, z=1.0, r=1.0, parent=2),
        ],
    )


@pytest.fixture
def single_node_neuron():
    """A neuron with a single root node (edge case)."""
    return NeuronMorphology(
        type="NeuronMorphology",
        tree=[
            SWCSample(id=1, type=1, x=5.0, y=5.0, z=5.0, r=3.0, parent=-1),
        ],
    )


@pytest.fixture
def identity_transform():
    return AffineTransform(
        type="affine",
        matrix=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
    )


# ---------------------------------------------------------------------------
# Tests: binary encoding
# ---------------------------------------------------------------------------


class TestNeuronToSkeletonBinary:
    def test_header(self, simple_neuron):
        data = neuron_to_skeleton_binary(simple_neuron)
        num_verts, num_edges = struct.unpack_from("<II", data, 0)
        assert num_verts == 3
        assert num_edges == 2  # edges: 1->2, 2->3

    def test_vertex_positions(self, simple_neuron):
        data = neuron_to_skeleton_binary(simple_neuron)
        offset = 8  # after header
        positions = struct.unpack_from("<9f", data, offset)
        assert positions == (0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 5.0, 1.0)

    def test_edge_pairs(self, simple_neuron):
        data = neuron_to_skeleton_binary(simple_neuron)
        offset = 8 + 3 * 4 * 3  # header + 9 floats
        edges = struct.unpack_from("<4I", data, offset)
        # Edge 1->2: parent_idx=0, child_idx=1
        # Edge 2->3: parent_idx=1, child_idx=2
        assert edges == (0, 1, 1, 2)

    def test_radius_attribute(self, simple_neuron):
        data = neuron_to_skeleton_binary(simple_neuron, include_radius=True, include_type=False)
        offset = 8 + 3 * 4 * 3 + 2 * 4 * 2  # header + verts + edges
        radii = struct.unpack_from("<3f", data, offset)
        assert radii == (5.0, 2.0, 1.0)

    def test_type_attribute(self, simple_neuron):
        data = neuron_to_skeleton_binary(simple_neuron, include_radius=False, include_type=True)
        offset = 8 + 3 * 4 * 3 + 2 * 4 * 2  # header + verts + edges
        types = struct.unpack_from("<3f", data, offset)
        assert types == (1.0, 2.0, 2.0)

    def test_both_attributes(self, simple_neuron):
        data = neuron_to_skeleton_binary(simple_neuron)
        # Total size: 8 (header) + 36 (verts) + 16 (edges) + 12 (radii) + 12 (types) = 84
        assert len(data) == 84

    def test_no_attributes(self, simple_neuron):
        data = neuron_to_skeleton_binary(simple_neuron, include_radius=False, include_type=False)
        # 8 (header) + 36 (verts) + 16 (edges) = 60
        assert len(data) == 60

    def test_single_node(self, single_node_neuron):
        data = neuron_to_skeleton_binary(single_node_neuron)
        num_verts, num_edges = struct.unpack_from("<II", data, 0)
        assert num_verts == 1
        assert num_edges == 0
        # header + 3 floats (pos) + 0 edges + 1 float (radius) + 1 float (type)
        assert len(data) == 8 + 12 + 0 + 4 + 4


# ---------------------------------------------------------------------------
# Tests: affine transform conversion
# ---------------------------------------------------------------------------


class TestAffineToNgTransform:
    def test_identity(self, identity_transform):
        result = affine_to_ng_transform(identity_transform)
        assert len(result) == 12
        # Row-major 3x4: each row = [Rx, Ry, Rz, Tx]
        expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]
        assert result == expected

    def test_translation(self):
        t = AffineTransform(
            type="affine",
            matrix=[
                [1, 0, 0, 10],
                [0, 1, 0, 20],
                [0, 0, 1, 30],
                [0, 0, 0, 1],
            ],
        )
        result = affine_to_ng_transform(t)
        # Row-major: [1,0,0,10, 0,1,0,20, 0,0,1,30]
        # tx=result[3], ty=result[7], tz=result[11]
        assert result[3] == 10
        assert result[7] == 20
        assert result[11] == 30

    def test_scale(self):
        t = AffineTransform(
            type="affine",
            matrix=[
                [2, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 4, 0],
                [0, 0, 0, 1],
            ],
        )
        result = affine_to_ng_transform(t)
        # Row-major: [2,0,0,0, 0,3,0,0, 0,0,4,0]
        assert result[0] == 2
        assert result[5] == 3
        assert result[10] == 4


# ---------------------------------------------------------------------------
# Tests: build_skeleton_info
# ---------------------------------------------------------------------------


class TestBuildSkeletonInfo:
    def test_defaults(self):
        info = build_skeleton_info()
        d = info.to_info_dict()
        assert d["@type"] == "neuroglancer_skeletons"
        assert len(d["vertex_attributes"]) == 2
        assert d["vertex_attributes"][0]["id"] == "radius"
        assert d["vertex_attributes"][1]["id"] == "type"

    def test_no_attributes(self):
        info = build_skeleton_info(include_radius=False, include_type=False)
        d = info.to_info_dict()
        assert "vertex_attributes" not in d

    def test_with_transform(self, identity_transform):
        info = build_skeleton_info(transform=identity_transform)
        d = info.to_info_dict()
        assert "transform" in d
        assert len(d["transform"]) == 12


# ---------------------------------------------------------------------------
# Tests: write_skeleton (disk I/O)
# ---------------------------------------------------------------------------


class TestWriteSkeleton:
    def test_creates_directory(self, simple_neuron, tmp_path):
        out = tmp_path / "skel"
        write_skeleton(out, 1, simple_neuron)
        assert out.is_dir()
        assert (out / "info").exists()
        assert (out / "1").exists()

    def test_info_json_valid(self, simple_neuron, tmp_path):
        out = tmp_path / "skel"
        write_skeleton(out, 42, simple_neuron)
        info = json.loads((out / "info").read_text())
        assert info["@type"] == "neuroglancer_skeletons"

    def test_binary_matches(self, simple_neuron, tmp_path):
        out = tmp_path / "skel"
        write_skeleton(out, 1, simple_neuron)
        disk_binary = (out / "1").read_bytes()
        expected = neuron_to_skeleton_binary(simple_neuron)
        assert disk_binary == expected

    def test_with_transform(self, simple_neuron, identity_transform, tmp_path):
        out = tmp_path / "skel"
        write_skeleton(out, 1, simple_neuron, transform=identity_transform)
        info = json.loads((out / "info").read_text())
        assert "transform" in info
