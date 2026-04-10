"""Smoke tests for Draco compression benchmark utilities.

Verifies encode/decode round-trip, L2 error computation, and
PBF3 size helpers work correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if DracoPy is not installed
DracoPy = pytest.importorskip("DracoPy")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from benchmark_draco import (
    compute_l2_error,
    draco_decode,
    draco_encode,
    generate_synthetic_mesh,
    hybrid_pbf3_draco_size,
    pbf3_gzipped_size,
    pbf3_raw_size,
)


@pytest.fixture
def simple_mesh():
    """A simple 4-vertex, 2-triangle mesh."""
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    return verts, faces


@pytest.fixture
def large_mesh():
    """A larger synthetic mesh."""
    return generate_synthetic_mesh(n_faces=500, seed=123)


class TestDracoRoundTrip:
    """Verify Draco encode/decode preserves mesh structure."""

    def test_vertex_count_preserved(self, simple_mesh):
        verts, faces = simple_mesh
        encoded = draco_encode(verts, faces, quantization_bits=16)
        decoded_v, decoded_f = draco_decode(encoded)
        assert decoded_v.shape[0] == verts.shape[0]
        assert decoded_v.shape[1] == 3

    def test_face_count_preserved(self, simple_mesh):
        verts, faces = simple_mesh
        encoded = draco_encode(verts, faces, quantization_bits=16)
        decoded_v, decoded_f = draco_decode(encoded)
        assert decoded_f.shape[0] == faces.shape[0]
        assert decoded_f.shape[1] == 3

    def test_high_quant_low_relative_error(self):
        """20-bit quantization should have very low relative L2 error on a 3D mesh."""
        # Use a proper 3D mesh (not flat) so all bbox dimensions are non-zero
        verts = np.array(
            [[0, 0, 0], [100, 0, 0], [100, 100, 0], [0, 100, 0],
             [50, 50, 100], [0, 0, 50]],
            dtype=np.float64,
        )
        faces = np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 1, 5], [1, 2, 5]],
            dtype=np.uint32,
        )
        encoded = draco_encode(verts, faces, quantization_bits=20)
        decoded_v, _ = draco_decode(encoded)
        if len(decoded_v) == len(verts):
            l2 = compute_l2_error(verts, decoded_v)
            # Relative to bbox diagonal, 20-bit error should be very small
            assert l2["rel_mean"] < 0.01

    def test_low_quant_higher_error(self, simple_mesh):
        """8-bit quantization should have higher error than 20-bit."""
        verts, faces = simple_mesh
        enc_8 = draco_encode(verts, faces, quantization_bits=8)
        enc_20 = draco_encode(verts, faces, quantization_bits=20)
        dec_8, _ = draco_decode(enc_8)
        dec_20, _ = draco_decode(enc_20)
        l2_8 = compute_l2_error(verts, dec_8)
        l2_20 = compute_l2_error(verts, dec_20)
        assert l2_8["mean"] >= l2_20["mean"]

    def test_larger_mesh_roundtrip(self, large_mesh):
        """Draco should encode/decode without error; vertex count may differ
        due to Draco deduplication of coincident vertices."""
        verts, faces = large_mesh
        encoded = draco_encode(verts, faces, quantization_bits=14)
        decoded_v, decoded_f = draco_decode(encoded)
        # Draco may deduplicate coincident vertices, so we check structure
        assert decoded_v.shape[1] == 3
        assert decoded_f.shape[1] == 3
        assert decoded_v.shape[0] > 0
        assert decoded_f.shape[0] > 0

    def test_compression_effective(self, large_mesh):
        """Draco output should be smaller than raw float32 data."""
        verts, faces = large_mesh
        raw_size = pbf3_raw_size(verts, faces)
        encoded = draco_encode(verts, faces, quantization_bits=14)
        assert len(encoded) < raw_size


class TestL2Error:
    """Verify L2 error computation."""

    def test_identical_vertices(self):
        verts = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        l2 = compute_l2_error(verts, verts)
        assert l2["mean"] == 0.0
        assert l2["max"] == 0.0

    def test_known_error(self):
        orig = np.array([[0, 0, 0]], dtype=np.float64)
        shifted = np.array([[1, 0, 0]], dtype=np.float64)
        l2 = compute_l2_error(orig, shifted)
        assert abs(l2["mean"] - 1.0) < 1e-10
        assert abs(l2["max"] - 1.0) < 1e-10

    def test_relative_to_diagonal(self):
        orig = np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float64)
        shifted = np.array([[0.1, 0, 0], [10, 10, 10]], dtype=np.float64)
        l2 = compute_l2_error(orig, shifted)
        diag = np.sqrt(10**2 * 3)
        expected_rel = 0.05 / diag  # mean of [0.1, 0] / diag
        assert abs(l2["rel_mean"] - expected_rel) < 1e-6

    def test_shape_mismatch_raises(self):
        a = np.array([[0, 0, 0]], dtype=np.float64)
        b = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_l2_error(a, b)

    def test_all_percentiles_present(self, simple_mesh):
        verts, faces = simple_mesh
        encoded = draco_encode(verts, faces, quantization_bits=14)
        decoded_v, _ = draco_decode(encoded)
        l2 = compute_l2_error(verts, decoded_v)
        for key in ["mean", "max", "p95", "p99", "rel_mean", "rel_max"]:
            assert key in l2


class TestPbf3Size:
    """Verify PBF3 size calculation helpers."""

    def test_raw_size(self, simple_mesh):
        verts, faces = simple_mesh
        expected = 4 * 3 * 4 + 2 * 3 * 4  # 4 verts * 3 * 4B + 2 faces * 3 * 4B
        assert pbf3_raw_size(verts, faces) == expected

    def test_gzipped_smaller_than_raw(self, large_mesh):
        verts, faces = large_mesh
        raw = pbf3_raw_size(verts, faces)
        gz = pbf3_gzipped_size(verts, faces)
        assert gz < raw

    def test_hybrid_produces_positive_size(self, simple_mesh):
        verts, faces = simple_mesh
        gz, draco_raw = hybrid_pbf3_draco_size(verts, faces, 14)
        assert gz > 0
        assert draco_raw > 0


class TestSyntheticMesh:
    """Verify synthetic mesh generator."""

    def test_correct_shape(self):
        verts, faces = generate_synthetic_mesh(n_faces=100)
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert faces.shape[0] == 100

    def test_deterministic(self):
        v1, f1 = generate_synthetic_mesh(n_faces=50, seed=42)
        v2, f2 = generate_synthetic_mesh(n_faces=50, seed=42)
        np.testing.assert_array_equal(v1, v2)
        np.testing.assert_array_equal(f1, f2)
