"""Tests for the in-memory Rust Draco encoder (`mudm_tools._rs.draco_encode_mesh`).

These cover the T2 SOTA-NG/Draco round: the encoder was swapped from a temp-OBJ
round-trip to a direct in-memory `MeshBuilder` build. The contract verified here:

  1. DRACO magic + non-empty output (no decoder needed).
  2. Byte-determinism across repeated calls (no decoder needed).
  3. Decode-equivalence: decoded geometry matches the input AS A SET within
     quantization tolerance, AND triangle-count + UNIQUE-vertex-count parity holds
     (DracoPy-gated, since the vendored draco-oxide decode module is a stub and
     DracoPy is the only working decoder). The raw decoded point count is NOT
     asserted: libdraco's decode-side corner-table split inflates it (the benign
     4->8 artifact) on any QuantizationCoordinateWise stream.

Note: `draco_encode_mesh` shares the NG u32 encoder path, which (A1) uses the
lossless grid-identity QuantizationCoordinateWise transform
(`encode::Config::with_ng_lossless(qbits)`). The re-bake guards live on the Rust
side: `rebake_guard_f32_byte_identical` (GLB/f32, frozen byte-identical) and
`rebake_guard_u32_ng_lossless` (NG u32, re-baked for A1).
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

        # FACE-count parity (mandatory): this clean tetra has no coincident
        # vertices and no degenerate faces, so build()'s dedup/degenerate-filter
        # must NOT drop any triangle.
        assert df.shape[0] == n_in_tris, (
            f"triangle count changed: in={n_in_tris} out={df.shape[0]}"
        )
        # UNIQUE-decoded-vertex-count parity (NOT raw point count): libdraco's
        # decode-side attribute-corner-table split inflates the reported point
        # count (the benign 4->8 artifact, present on ANY QuantizationCoordinateWise
        # stream that decodes — independent of the data-bbox vs grid-identity
        # transform). Deduplicate before comparing.
        n_unique_out = len(set(map(tuple, np.round(dv).tolist())))
        assert n_unique_out == n_in_verts, (
            f"unique vertex count changed: in={n_in_verts} out={n_unique_out} "
            f"(raw out={dv.shape[0]})"
        )

        # SET equivalence — EXACT. A1: the encoder pre-quantizes the world coords
        # to the integer grid [0, 2^qbits - 1] (the encoder's `_quantize` mirror),
        # then stores them under the LOSSLESS grid-identity QuantizationCoordinateWise
        # transform. So Draco/DracoPy decodes back the EXACT grid integers (the
        # viewer, not Draco, re-expands them to world space via the info transform).
        # Compare the unique decoded position SET against the grid integers with
        # zero tolerance.
        in_q = self._quantize(positions, qbits)
        in_set = np.array(sorted(set(map(tuple, in_q.tolist()))))
        out_set = np.array(sorted(set(map(tuple, np.round(dv).tolist()))))
        assert np.allclose(dv, np.round(dv), atol=0.0), (
            "decoded positions are not exact integers (lossless transform broken)"
        )
        np.testing.assert_array_equal(out_set, in_set)

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
