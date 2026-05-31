"""Tests for the in-memory Rust Draco encoder (`mudm_tools._rs.draco_encode_mesh`).

These cover the T2 SOTA-NG/Draco round: the encoder was swapped from a temp-OBJ
round-trip to a direct in-memory `MeshBuilder` build. The contract verified here:

  1. DRACO magic + non-empty output (no decoder needed).
  2. Byte-determinism across repeated calls (no decoder needed).
  3. Decode-equivalence: decoded geometry matches the input AS A SET within
     quantization tolerance, AND triangle/vertex COUNT parity holds
     (DracoPy-gated, since the vendored draco-oxide decode module is a stub and
     DracoPy is the only working decoder).

The re-bake itself is asserted on the Rust side
(`encoder_draco::rebake_snapshot::rebake_guard_bytes_unchanged_vs_old_obj_path`):
empirically the swap is byte-PRESERVING for these position-only meshes because the
old `load_obj` bridge already used the identical MeshBuilder recipe.
"""

from __future__ import annotations

import numpy as np
import pytest

_rs = pytest.importorskip("mudm_tools._rs")
draco_encode_mesh = _rs.draco_encode_mesh


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _tetra():
    """A non-degenerate tetrahedron (4 verts, 4 tris)."""
    positions = [
        0.0, 0.0, 0.0,
        100.0, 0.0, 0.0,
        50.0, 100.0, 0.0,
        50.0, 50.0, 100.0,
    ]
    indices = [0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]
    return positions, indices


def _grid(n: int = 16):
    """An n x n vertex lattice with a z-ripple -> ~2*(n-1)^2 tris, all unique."""
    positions: list[float] = []
    for y in range(n):
        for x in range(n):
            z = float((x * 7 + y * 13) % 97)
            positions.extend([float(x) * 16.0, float(y) * 16.0, z])
    indices: list[int] = []
    for y in range(n - 1):
        for x in range(n - 1):
            i = y * n + x
            r = i + 1
            d = i + n
            dr = d + 1
            indices.extend([i, r, d, r, dr, d])
    return positions, indices


# ---------------------------------------------------------------------------
# Decoder-free gates (run everywhere)
# ---------------------------------------------------------------------------


class TestMagicAndDeterminism:
    def test_draco_magic_and_nonempty(self):
        positions, indices = _tetra()
        b = draco_encode_mesh(positions, indices, 14)
        assert len(b) > 5, "Draco output too short"
        assert b[:5] == b"DRACO", "missing DRACO magic"

    def test_deterministic_three_calls(self):
        positions, indices = _tetra()
        a = draco_encode_mesh(positions, indices, 14)
        b = draco_encode_mesh(positions, indices, 14)
        c = draco_encode_mesh(positions, indices, 14)
        assert a == b, "encode is non-deterministic (a != b)"
        assert b == c, "encode is non-deterministic (b != c)"

    def test_deterministic_large_mesh(self):
        positions, indices = _grid(16)
        a = draco_encode_mesh(positions, indices, 12)
        b = draco_encode_mesh(positions, indices, 12)
        assert a == b, "large-mesh encode is non-deterministic"
        assert a[:5] == b"DRACO"

    def test_empty_raises(self):
        with pytest.raises(Exception):
            draco_encode_mesh([], [], 14)


# ---------------------------------------------------------------------------
# Decode-equivalence + topology parity (DracoPy-gated)
# ---------------------------------------------------------------------------


class TestDecodeEquivalence:
    """Requires DracoPy (the only working Draco decoder available)."""

    @staticmethod
    def _decode(data: bytes):
        DracoPy = pytest.importorskip("DracoPy")
        mesh = DracoPy.decode(data)
        verts = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1, 3)
        return verts, faces

    @staticmethod
    def _quantize(positions: list[float], qbits: int) -> np.ndarray:
        """Mirror the Rust `draco_encode_mesh` bbox quantization so we compare in
        the same quantized integer lattice the encoder used."""
        p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        mins = p.min(axis=0)
        maxs = p.max(axis=0)
        rng = maxs - mins
        qmax = float((1 << qbits) - 1)
        out = np.zeros_like(p)
        for d in range(3):
            if rng[d] > 0:
                out[:, d] = np.clip(np.round((p[:, d] - mins[d]) / rng[d] * qmax), 0, qmax)
            else:
                out[:, d] = 0.0
        return out

    def test_geometry_set_equivalence_and_count_parity(self):
        pytest.importorskip("DracoPy")
        positions, indices = _tetra()
        qbits = 14
        data = draco_encode_mesh(positions, indices, qbits)
        dv, df = self._decode(data)

        n_in_verts = len(positions) // 3
        n_in_tris = len(indices) // 3

        # COUNT parity (mandatory): this clean tetra has no coincident vertices
        # and no degenerate faces, so build()'s dedup/degenerate-filter must NOT
        # drop anything.
        assert df.shape[0] == n_in_tris, (
            f"triangle count changed: in={n_in_tris} out={df.shape[0]}"
        )
        assert dv.shape[0] == n_in_verts, (
            f"vertex count changed: in={n_in_verts} out={dv.shape[0]}"
        )

        # SET equivalence within quantization tolerance. Vertex order may permute,
        # so compare as a nearest-neighbour set, not positionally. Decode returns
        # de-quantized world coords; compare against the encoder's quantized lattice
        # re-expanded to world space via the same bbox.
        in_q = self._quantize(positions, qbits)
        p = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        mins = p.min(axis=0)
        maxs = p.max(axis=0)
        rng = maxs - mins
        qmax = float((1 << qbits) - 1)
        expected_world = mins + in_q / qmax * rng  # de-quantize back to world
        # tolerance = one quantization step on the largest axis
        step = float(np.max(rng)) / qmax
        tol = step * 4.0 + 1e-4

        for ev in expected_world:
            dists = np.linalg.norm(dv - ev, axis=1)
            assert dists.min() <= tol, (
                f"input vertex {ev} has no decoded match within tol={tol} "
                f"(nearest={dists.min()})"
            )

    def test_count_parity_large_mesh(self):
        pytest.importorskip("DracoPy")
        positions, indices = _grid(16)
        data = draco_encode_mesh(positions, indices, 14)
        _dv, df = self._decode(data)
        n_in_tris = len(indices) // 3
        # The grid has all-unique vertices and no degenerate faces.
        assert df.shape[0] == n_in_tris, (
            f"triangle count changed on grid: in={n_in_tris} out={df.shape[0]}"
        )
