"""A1: lossless, libdraco-conformant Neuroglancer (u32) Draco path.

The NG (u32) Draco path emits LOSSLESS positions while staying on the
libdraco-conformant `QuantizationCoordinateWise` portabilization (id=2, which the
Neuroglancer WASM viewer / libdraco / DracoPy decode today). The trick (A1): the
NG assembler pre-quantizes positions to the integer grid [0, 2^qbits - 1], and the
encoder forces the QuantizationCoordinateWise transform to be the IDENTITY over
that same grid — min=0, range=(2^qbits - 1), bits=qbits. Encode is then
round((v-0)/(2^qbits-1) * (2^qbits-1)) = v and libdraco dequant is
v * (2^qbits-1)/(2^qbits-1) + 0 = v — EXACT. Integer 500 decodes as 500.0
(not the near-lossless 500.244 the data-bbox transform produced).

`qbits` is threaded from `generate_neuroglancer_multilod`'s
`vertex_quantization_bits` (default 10) into the encoder, so the grid range always
matches the pre-quantization (no hardcoded duplicate).

Earlier (T2b) the `ToBits` portabilization (id=1) was tried for losslessness but
its bitstream is NOT libdraco-decodable; A1 supersedes it by keeping the
conformant id=2 transform and just making it exact.

The GLB/f32 path (`encode_draco_mesh_f32`) is UNCHANGED — it keeps the quantizing
`Config::default()` and is unaffected by this change (by construction: it never
sets the grid-identity override).
"""

from __future__ import annotations

import numpy as np
import pytest

_rs = pytest.importorskip("mudm_tools._rs")

# Direct entry into the NG u32 path (pre-quantized integers in, raw .drc out).
draco_encode_ng_u32 = getattr(_rs, "draco_encode_ng_u32", None)


def _tetra_u32():
    positions = [
        0,
        0,
        0,
        1000,
        0,
        0,
        500,
        1000,
        0,
        500,
        500,
        1000,
    ]
    indices = [0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]
    return positions, indices


# qbits the NG path pre-quantizes with by default (generate_neuroglancer_multilod
# vertex_quantization_bits=10). The grid range is then [0, 2^qbits - 1].
_NG_DEFAULT_QBITS = 10


def _tetra_u32_near_max():
    """A tetra whose coords approach the top of the NG grid (2^qbits - 1).

    At the default qbits=10 the grid max is 1023; place vertices at/near it so
    the lossless guarantee is exercised at the upper end of the integer range,
    not just near the origin.
    """
    qmax = (1 << _NG_DEFAULT_QBITS) - 1  # 1023
    positions = [
        0,
        0,
        0,
        qmax,
        0,
        0,
        qmax // 2,
        qmax,
        0,
        qmax // 2,
        qmax // 2,
        qmax,
    ]
    indices = [0, 1, 2, 0, 1, 3, 1, 2, 3, 0, 2, 3]
    return positions, indices


@pytest.mark.skipif(draco_encode_ng_u32 is None, reason="NG u32 entry not built")
class TestNgPath:
    @staticmethod
    def _decode(data: bytes):
        DracoPy = pytest.importorskip("DracoPy")
        mesh = DracoPy.decode(data)
        verts = np.asarray(mesh.points, dtype=np.float64).reshape(-1, 3)
        faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1, 3)
        return verts, faces

    def test_magic_and_nonempty(self):
        positions, indices = _tetra_u32()
        b = draco_encode_ng_u32(positions, indices)
        assert len(b) > 5
        assert b[:5] == b"DRACO"

    def test_deterministic_three_calls(self):
        positions, indices = _tetra_u32()
        a = draco_encode_ng_u32(positions, indices)
        b = draco_encode_ng_u32(positions, indices)
        c = draco_encode_ng_u32(positions, indices)
        assert a == b
        assert b == c

    def test_ng_stream_is_decodable_by_libdraco(self):
        """The NG path MUST stay on a libdraco-decodable portabilization
        (QuantizationCoordinateWise). If this ever breaks, the NG viewer breaks."""
        pytest.importorskip("DracoPy")
        positions, indices = _tetra_u32()
        data = draco_encode_ng_u32(positions, indices)
        dv, df = self._decode(data)
        assert dv.shape[1] == 3 and df.shape[1] == 3
        assert dv.shape[0] > 0 and df.shape[0] > 0

    @pytest.mark.parametrize(
        "mesh_factory",
        [_tetra_u32, _tetra_u32_near_max],
        ids=["origin_tetra", "near_grid_max_tetra"],
    )
    def test_ng_lossless_exact_integers(self, mesh_factory):
        """Lossless NG round-trip (A1). The NG path keeps the
        libdraco-conformant QuantizationCoordinateWise transform (portabilization
        id=2) but forces it to be the IDENTITY over the NG integer grid
        [0, 2^qbits - 1] (min=0, range=2^qbits-1). Encode/dequant is then v->v
        EXACT, so libdraco/DracoPy decodes the pre-quantized grid integers with
        zero quantization error (500 -> 500.0, not 500.244).

        Losslessness is asserted on the SET of unique decoded positions (round +
        deduplicate) == the input position set, plus FACE-count parity. The raw
        decoded point count is NOT asserted == n_verts: libdraco's decode-side
        attribute-corner-table split can inflate the reported point count (the
        4->8 artifact), which is benign and orthogonal to position losslessness.
        """
        pytest.importorskip("DracoPy")
        positions, indices = mesh_factory()
        data = draco_encode_ng_u32(positions, indices)
        dv, df = self._decode(data)

        n_in_tris = len(indices) // 3
        # face-count parity (connectivity is preserved exactly)
        assert df.shape[0] == n_in_tris
        # exact integers, no quantization error
        assert np.allclose(dv, np.round(dv), atol=0.0)
        # unique-decoded-position-set == input-position-set (handles the benign
        # decode-side vertex inflation by deduplicating before comparison)
        in_pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        in_sorted = np.array(sorted(set(map(tuple, in_pts.tolist()))))
        out_sorted = np.array(sorted(set(map(tuple, np.round(dv).tolist()))))
        np.testing.assert_array_equal(out_sorted, in_sorted)
