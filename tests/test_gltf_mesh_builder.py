"""Tests for mesh generation utilities and neuron tube mesh generation."""

import math

import numpy as np
import pytest

from mudm_tools.swc import (
    NeuronMorphology,
    SWCSample,
    _extract_paths,
    neuron_to_tube_mesh,
    neuron_to_typed_meshes,
)
from mudm_tools.gltf.mesh_builder import (
    _icosphere,
    _tube_along_path,
    smooth_path,
    thin_path,
)


class TestTubeAlongPath:
    def test_basic_tube(self):
        points = [np.array([0, 0, 0.0]), np.array([0, 0, 10.0])]
        radii = [1.0, 1.0]
        v, n, idx = _tube_along_path(points, radii, 8)

        assert v.shape == (16, 3)  # 2 rings x 8 segments
        assert n.shape == (16, 3)
        assert idx.shape == (16, 3)  # 8 quads x 2 tris

    def test_tapered_tube(self):
        points = [np.array([0, 0, 0.0]), np.array([5, 0, 0.0])]
        radii = [2.0, 0.5]
        v, _, _ = _tube_along_path(points, radii, 6)

        # First ring: distance from p0 should be ~2.0
        ring0 = v[:6]
        dists0 = np.linalg.norm(ring0 - points[0], axis=1)
        np.testing.assert_allclose(dists0, 2.0, atol=1e-10)

        # Second ring: distance from p1 should be ~0.5
        ring1 = v[6:]
        dists1 = np.linalg.norm(ring1 - points[1], axis=1)
        np.testing.assert_allclose(dists1, 0.5, atol=1e-10)

    def test_multipoint_path_shared_rings(self):
        """A 3-point path should have shared ring at the middle node."""
        points = [
            np.array([0, 0, 0.0]),
            np.array([5, 0, 0.0]),
            np.array([10, 0, 0.0]),
        ]
        radii = [1.0, 1.0, 1.0]
        v, n, idx = _tube_along_path(points, radii, 8)

        # 3 rings x 8 = 24 verts (shared middle ring, not 32)
        assert v.shape[0] == 24
        assert idx.shape[0] == 32  # 2 segments x 8 quads x 2 tris

    def test_single_point_returns_empty(self):
        v, n, idx = _tube_along_path([np.array([0, 0, 0.0])], [1.0], 8)
        assert v.shape[0] == 0

    def test_all_indices_valid(self):
        points = [np.array([0, 0, 0.0]), np.array([1, 1, 1.0])]
        v, _, idx = _tube_along_path(points, [0.5, 0.3], 12)
        assert idx.max() < v.shape[0]
        assert idx.min() >= 0

    def test_parallel_transport_no_twist(self):
        """Along a straight line, all rings should have the same orientation."""
        points = [
            np.array([0, 0, 0.0]),
            np.array([0, 0, 5.0]),
            np.array([0, 0, 10.0]),
        ]
        radii = [1.0, 1.0, 1.0]
        v, _, _ = _tube_along_path(points, radii, 4)

        # For a straight path along Z, all rings should have same XY pattern
        ring0 = v[:4, :2]  # XY of first ring
        ring1 = v[4:8, :2]  # XY of middle ring
        ring2 = v[8:12, :2]  # XY of last ring
        np.testing.assert_allclose(ring0, ring1, atol=1e-10)
        np.testing.assert_allclose(ring1, ring2, atol=1e-10)


class TestExtractPaths:
    def _make_tree(self, samples):
        return [SWCSample(**s) for s in samples]

    def test_linear_chain(self):
        tree = self._make_tree([
            {"id": 1, "type": 1, "x": 0, "y": 0, "z": 0, "r": 1, "parent": -1},
            {"id": 2, "type": 3, "x": 1, "y": 0, "z": 0, "r": 1, "parent": 1},
            {"id": 3, "type": 3, "x": 2, "y": 0, "z": 0, "r": 1, "parent": 2},
        ])
        id_to_sample = {s.id: s for s in tree}
        paths = _extract_paths(tree, id_to_sample)

        assert len(paths) == 1
        assert paths[0] == [1, 2, 3]

    def test_branching(self):
        tree = self._make_tree([
            {"id": 1, "type": 1, "x": 0, "y": 0, "z": 0, "r": 1, "parent": -1},
            {"id": 2, "type": 3, "x": 1, "y": 0, "z": 0, "r": 1, "parent": 1},
            {"id": 3, "type": 3, "x": 0, "y": 1, "z": 0, "r": 1, "parent": 1},
        ])
        id_to_sample = {s.id: s for s in tree}
        paths = _extract_paths(tree, id_to_sample)

        assert len(paths) == 2
        # Both paths start from root node 1
        starts = {p[0] for p in paths}
        assert starts == {1}
        ends = {p[-1] for p in paths}
        assert ends == {2, 3}

    def test_deep_branching(self):
        """Root -> chain -> branch point -> two branches."""
        tree = self._make_tree([
            {"id": 1, "type": 1, "x": 0, "y": 0, "z": 0, "r": 1, "parent": -1},
            {"id": 2, "type": 3, "x": 1, "y": 0, "z": 0, "r": 1, "parent": 1},
            {"id": 3, "type": 3, "x": 2, "y": 0, "z": 0, "r": 1, "parent": 2},
            {"id": 4, "type": 3, "x": 3, "y": 1, "z": 0, "r": 1, "parent": 3},
            {"id": 5, "type": 3, "x": 3, "y": -1, "z": 0, "r": 1, "parent": 3},
        ])
        id_to_sample = {s.id: s for s in tree}
        paths = _extract_paths(tree, id_to_sample)

        assert len(paths) == 3
        # One long trunk [1,2,3], then two branches from 3
        trunk = [p for p in paths if len(p) == 3]
        assert len(trunk) == 1
        assert trunk[0] == [1, 2, 3]
        branches = [p for p in paths if len(p) == 2]
        assert len(branches) == 2


class TestIcosphere:
    def test_subdivision_0(self):
        v, n, idx = _icosphere(np.zeros(3), 1.0, subdivisions=0)
        assert v.shape[0] == 12  # Icosahedron has 12 vertices
        assert idx.shape[0] == 20  # 20 faces

    def test_subdivision_2(self):
        v, n, idx = _icosphere(np.zeros(3), 1.0, subdivisions=2)
        assert v.shape[0] == 162
        assert idx.shape[0] == 320

    def test_radius_scaling(self):
        center = np.array([5.0, 5.0, 5.0])
        radius = 3.0
        v, _, _ = _icosphere(center, radius, subdivisions=1)
        dists = np.linalg.norm(v - center, axis=1)
        np.testing.assert_allclose(dists, radius, atol=1e-10)

    def test_normals_unit_length(self):
        _, n, _ = _icosphere(np.zeros(3), 2.0, subdivisions=1)
        lengths = np.linalg.norm(n, axis=1)
        np.testing.assert_allclose(lengths, 1.0, atol=1e-10)

    def test_zero_radius_returns_empty(self):
        v, n, idx = _icosphere(np.zeros(3), 0.0)
        assert v.shape[0] == 0


class TestNeuronToTubeMesh:
    def _simple_neuron(self) -> NeuronMorphology:
        """Soma at origin, one dendrite extending along X."""
        return NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
                SWCSample(id=2, type=3, x=10, y=0, z=0, r=2, parent=1),
            ],
        )

    def test_single_edge_neuron(self):
        neuron = self._simple_neuron()
        v, n, idx = neuron_to_tube_mesh(neuron, segments=8)

        # Soma sphere(162) + tube path [1,2] (2 nodes x 8 segments = 16)
        assert v.shape[0] == 162 + 16
        assert n.shape[0] == v.shape[0]
        assert idx.max() < v.shape[0]

    def test_branching_neuron(self):
        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
                SWCSample(id=2, type=3, x=10, y=0, z=0, r=2, parent=1),
                SWCSample(id=3, type=3, x=0, y=10, z=0, r=2, parent=1),
                SWCSample(id=4, type=3, x=15, y=0, z=0, r=1, parent=2),
            ],
        )
        v, n, idx = neuron_to_tube_mesh(neuron, segments=8)

        # Soma sphere(162) + path [1,2,4] (3x8=24) + path [1,3] (2x8=16)
        # No junction sphere at node 1 (it's the soma)
        expected_verts = 162 + 24 + 16
        assert v.shape[0] == expected_verts
        assert idx.max() < v.shape[0]

    def test_junction_sphere_at_branch_point(self):
        """Non-soma branch point should get a junction sphere."""
        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=3, x=0, y=0, z=0, r=2, parent=-1),
                SWCSample(id=2, type=3, x=5, y=0, z=0, r=1.5, parent=1),
                SWCSample(id=3, type=3, x=10, y=1, z=0, r=1, parent=2),
                SWCSample(id=4, type=3, x=10, y=-1, z=0, r=1, parent=2),
            ],
        )
        v, n, idx = neuron_to_tube_mesh(neuron, segments=8)

        # No soma sphere (root is type=3)
        # Junction sphere at node 2 (subdiv=1 -> 42 verts)
        # Path [1,2] (2x8=16) + path [2,3] (2x8=16) + path [2,4] (2x8=16)
        expected = 42 + 16 + 16 + 16
        assert v.shape[0] == expected
        assert idx.max() < v.shape[0]

    def test_soma_only(self):
        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
            ],
        )
        v, n, idx = neuron_to_tube_mesh(neuron)
        assert v.shape[0] == 162  # Just the icosphere
        assert idx.shape[0] == 320

    def test_min_radius_clamp(self):
        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=0.001, parent=-1),
                SWCSample(id=2, type=3, x=10, y=0, z=0, r=0.001, parent=1),
            ],
        )
        v, _, _ = neuron_to_tube_mesh(neuron, min_radius=1.0)
        # Soma sphere should use min_radius=1.0
        soma_verts = v[:162]
        center = np.array([0.0, 0.0, 0.0])
        dists = np.linalg.norm(soma_verts - center, axis=1)
        np.testing.assert_allclose(dists, 1.0, atol=1e-10)

    def test_different_segment_counts(self):
        neuron = self._simple_neuron()
        for segs in [4, 6, 12]:
            v, _, idx = neuron_to_tube_mesh(neuron, segments=segs)
            # Tube path [1,2]: 2 * segs verts
            assert v.shape[0] == 162 + 2 * segs

    def test_no_soma_node(self):
        """Non-soma root should not generate a sphere."""
        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=3, x=0, y=0, z=0, r=2, parent=-1),
                SWCSample(id=2, type=3, x=10, y=0, z=0, r=1, parent=1),
            ],
        )
        v, _, _ = neuron_to_tube_mesh(neuron, segments=8)
        # Only 1 tube path, no sphere
        assert v.shape[0] == 16

    def test_continuous_tube_no_gaps(self):
        """Verify that a multi-segment path has continuous ring vertices."""
        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=3, x=0, y=0, z=0, r=1, parent=-1),
                SWCSample(id=2, type=3, x=5, y=0, z=0, r=1, parent=1),
                SWCSample(id=3, type=3, x=10, y=0, z=0, r=1, parent=2),
                SWCSample(id=4, type=3, x=15, y=0, z=0, r=1, parent=3),
            ],
        )
        v, _, _ = neuron_to_tube_mesh(neuron, segments=8)

        # Single path [1,2,3,4] -> 4 rings x 8 = 32 verts (shared rings)
        assert v.shape[0] == 32

        # Middle ring (at node 2, index=8..15) should be at x=5
        ring1 = v[8:16]
        np.testing.assert_allclose(ring1[:, 0], 5.0, atol=1e-10)


class TestSmoothPath:
    def test_no_op_when_zero_subdivisions(self):
        pts = [np.array([0, 0, 0.0]), np.array([1, 0, 0.0])]
        radii = [1.0, 0.5]
        sp, sr = smooth_path(pts, radii, 0)
        assert len(sp) == 2
        assert len(sr) == 2

    def test_no_op_when_single_point(self):
        pts = [np.array([0, 0, 0.0])]
        radii = [1.0]
        sp, sr = smooth_path(pts, radii, 3)
        assert len(sp) == 1

    def test_output_length(self):
        """N points + M subdivisions -> N + (N-1)*M output points."""
        pts = [np.array([float(i), 0, 0]) for i in range(5)]
        radii = [1.0] * 5
        sp, sr = smooth_path(pts, radii, 3)
        # 5 original + 4 segments x 3 subdivisions = 5 + 12 = 17
        expected = 5 + (5 - 1) * 3
        assert len(sp) == expected
        assert len(sr) == expected

    def test_endpoints_preserved(self):
        pts = [np.array([0, 0, 0.0]), np.array([5, 3, 1.0]), np.array([10, 0, 0.0])]
        radii = [2.0, 1.0, 0.5]
        sp, sr = smooth_path(pts, radii, 4)
        np.testing.assert_allclose(sp[0], [0, 0, 0], atol=1e-10)
        np.testing.assert_allclose(sp[-1], [10, 0, 0], atol=1e-10)
        assert abs(sr[0] - 2.0) < 1e-10
        assert abs(sr[-1] - 0.5) < 1e-10

    def test_radii_interpolated(self):
        pts = [np.array([0, 0, 0.0]), np.array([10, 0, 0.0])]
        radii = [2.0, 4.0]
        sp, sr = smooth_path(pts, radii, 1)
        # 2 original + 1 subdivision = 3 points
        assert len(sr) == 3
        # Midpoint radius should be ~3.0 (linear interp)
        assert abs(sr[1] - 3.0) < 1e-10

    def test_straight_line_stays_straight(self):
        pts = [np.array([float(i), 0, 0]) for i in range(4)]
        radii = [1.0] * 4
        sp, _ = smooth_path(pts, radii, 2)
        # All smoothed points should have Y=0 and Z=0 (straight line along X)
        for p in sp:
            assert abs(p[1]) < 1e-10
            assert abs(p[2]) < 1e-10


class TestSmoothingInNeuronMesh:
    def _simple_neuron(self):
        return NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=3, x=0, y=0, z=0, r=1, parent=-1),
                SWCSample(id=2, type=3, x=5, y=0, z=0, r=1, parent=1),
                SWCSample(id=3, type=3, x=10, y=5, z=0, r=1, parent=2),
            ],
        )

    def test_smoothing_increases_vertex_count(self):
        neuron = self._simple_neuron()
        v0, _, _ = neuron_to_tube_mesh(neuron, segments=8, smooth_subdivisions=0)
        v3, _, _ = neuron_to_tube_mesh(neuron, segments=8, smooth_subdivisions=3)
        # Smoothing adds more ring cross-sections -> more vertices
        assert v3.shape[0] > v0.shape[0]

    def test_smooth_indices_valid(self):
        neuron = self._simple_neuron()
        v, _, idx = neuron_to_tube_mesh(neuron, segments=8, smooth_subdivisions=3)
        assert idx.max() < v.shape[0]
        assert idx.min() >= 0

    def test_typed_meshes_with_smoothing(self):
        neuron = NeuronMorphology(
            type="NeuronMorphology",
            tree=[
                SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1),
                SWCSample(id=2, type=3, x=10, y=0, z=0, r=2, parent=1),
                SWCSample(id=3, type=3, x=20, y=5, z=0, r=1, parent=2),
            ],
        )
        typed_0 = neuron_to_typed_meshes(neuron, segments=8, smooth_subdivisions=0)
        typed_3 = neuron_to_typed_meshes(neuron, segments=8, smooth_subdivisions=3)

        # Both should produce type 1 (soma) and type 3 (dendrite)
        assert set(typed_0.keys()) == set(typed_3.keys())
        # Smoothed dendrite should have more vertices
        assert typed_3[3][0].shape[0] > typed_0[3][0].shape[0]
        # Soma sphere should be the same (smoothing doesn't affect spheres)
        assert typed_3[1][0].shape[0] == typed_0[1][0].shape[0]


class TestThinPath:
    def test_no_op_at_full_quality(self):
        pts = [np.array([float(i), 0, 0]) for i in range(20)]
        radii = [1.0] * 20
        tp, tr = thin_path(pts, radii, 1.0)
        assert len(tp) == 20

    def test_no_op_for_short_path(self):
        pts = [np.array([0, 0, 0.0]), np.array([1, 0, 0.0])]
        radii = [1.0, 1.0]
        tp, tr = thin_path(pts, radii, 0.1)
        assert len(tp) == 2  # always keep at least 2

    def test_reduces_point_count(self):
        pts = [np.array([float(i), 0, 0]) for i in range(100)]
        radii = [1.0] * 100
        tp, tr = thin_path(pts, radii, 0.5)
        assert len(tp) == 50
        assert len(tr) == 50

    def test_preserves_endpoints(self):
        pts = [np.array([0, 0, 0.0])] + [np.array([float(i), 1, 0]) for i in range(1, 9)] + [np.array([9, 0, 0.0])]
        radii = [2.0] + [1.0] * 8 + [3.0]
        tp, tr = thin_path(pts, radii, 0.3)
        np.testing.assert_allclose(tp[0], [0, 0, 0])
        np.testing.assert_allclose(tp[-1], [9, 0, 0])
        assert abs(tr[0] - 2.0) < 1e-10
        assert abs(tr[-1] - 3.0) < 1e-10

    def test_aggressive_thinning_keeps_minimum(self):
        pts = [np.array([float(i), 0, 0]) for i in range(50)]
        radii = [1.0] * 50
        tp, _ = thin_path(pts, radii, 0.01)
        assert len(tp) >= 2


class TestMeshQuality:
    def _long_neuron(self):
        """Neuron with many nodes -- good for testing quality reduction."""
        samples = [SWCSample(id=1, type=1, x=0, y=0, z=0, r=5, parent=-1)]
        for i in range(2, 52):
            samples.append(SWCSample(
                id=i, type=3, x=float(i * 2), y=float(i % 5), z=0, r=1.5, parent=i - 1,
            ))
        return NeuronMorphology(type="NeuronMorphology", tree=samples)

    def test_quality_reduces_vertices(self):
        neuron = self._long_neuron()
        v_full, _, _ = neuron_to_tube_mesh(
            neuron, segments=8, smooth_subdivisions=5, mesh_quality=1.0)
        v_half, _, _ = neuron_to_tube_mesh(
            neuron, segments=8, smooth_subdivisions=5, mesh_quality=0.5)
        # Half quality should have significantly fewer tube vertices
        assert v_half.shape[0] < v_full.shape[0]

    def test_quality_valid_indices(self):
        neuron = self._long_neuron()
        v, _, idx = neuron_to_tube_mesh(
            neuron, segments=8, smooth_subdivisions=5, mesh_quality=0.3)
        assert idx.max() < v.shape[0]
        assert idx.min() >= 0

    def test_typed_meshes_quality(self):
        neuron = self._long_neuron()
        t_full = neuron_to_typed_meshes(
            neuron, segments=8, smooth_subdivisions=5, mesh_quality=1.0)
        t_half = neuron_to_typed_meshes(
            neuron, segments=8, smooth_subdivisions=5, mesh_quality=0.5)
        # Dendrite mesh (type 3) should shrink; soma (type 1) unaffected
        assert t_half[3][0].shape[0] < t_full[3][0].shape[0]
        assert t_half[1][0].shape[0] == t_full[1][0].shape[0]

    def test_full_quality_unchanged(self):
        """mesh_quality=1.0 should produce identical output to no thinning."""
        neuron = self._long_neuron()
        v1, n1, idx1 = neuron_to_tube_mesh(
            neuron, segments=8, smooth_subdivisions=3, mesh_quality=1.0)
        v2, n2, idx2 = neuron_to_tube_mesh(
            neuron, segments=8, smooth_subdivisions=3)
        assert v1.shape == v2.shape
        np.testing.assert_array_equal(idx1, idx2)
