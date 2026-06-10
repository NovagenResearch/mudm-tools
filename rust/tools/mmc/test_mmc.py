"""Round-trip tests for the MMC reference codec, mirroring the NG pathologies
exercised by encoder_draco.rs tests."""

import numpy as np
import mmc


def canon(verts, tris):
    """Canonical mesh equality independent of vertex labels and triangle order:
    set of triangles as cyclically-minimal position triples (winding preserved)."""
    t = np.asarray(verts)[np.asarray(tris).reshape(-1, 3)]
    out = set()
    for tri in t:
        rots = [tuple(map(tuple, np.roll(tri, -k, axis=0))) for k in range(3)]
        out.add(min(rots))
    return out


def nondegenerate(pos, idx):
    p = pos.reshape(-1, 3)
    out = set()
    for t in idx.reshape(-1, 3):
        a, b, c = tuple(p[t[0]]), tuple(p[t[1]]), tuple(p[t[2]])
        if a != b and b != c and a != c:
            out.add(min([(a, b, c), (b, c, a), (c, a, b)]))
    return out


def rt(pos, idx, qbits=10):
    blob = mmc.encode_u32(pos, idx, qbits)
    dpos, didx = mmc.decode(blob)
    return blob, dpos, didx


def test_single_triangle():
    pos = np.array([0, 0, 0, 100, 0, 0, 50, 100, 0], dtype=np.uint32)
    idx = np.array([0, 1, 2], dtype=np.uint32)
    blob, dpos, didx = rt(pos, idx)
    assert blob[:4] == b"MMC1"
    assert canon(pos.reshape(-1, 3), idx) == canon(dpos, didx)
    print("single_triangle ok,", len(blob), "bytes")


def test_random_soup_lossless():
    rng = np.random.RandomState(7)
    n = 5000
    pos = rng.randint(0, 1 << 10, size=n * 3).astype(np.uint32)
    idx = rng.randint(0, n, size=3 * 4000).astype(np.uint32)
    blob, dpos, didx = rt(pos, idx)
    assert canon(dpos, didx) == nondegenerate(pos, idx)
    print("random_soup_lossless ok,", len(blob), "bytes for", len(didx), "tris")


def test_duplicate_and_orphan():
    # encoder_draco.rs pathologies: coincident dup + orphan + quant-degenerate face
    pos = np.array(
        [10, 10, 10,    # v0
         10, 10, 10,    # v1 == v0 (coincident dup)
         100, 0, 0,     # v2
         50, 100, 0,    # v3
         200, 50, 0,    # v4
         500, 500, 500  # v5 orphan
         ], dtype=np.uint32)
    idx = np.array([0, 1, 2, 2, 3, 4], dtype=np.uint32)  # face0 degenerate after dedup
    blob, dpos, didx = rt(pos, idx)
    assert len(didx) == 1, "degenerate face must drop"
    assert len(dpos) == 3, "orphans + dups must compact"
    assert canon(dpos, didx) == nondegenerate(pos, idx)
    print("duplicate_and_orphan ok")


def test_lattice_lossless():
    N = 64
    x, y = np.meshgrid(np.arange(N, dtype=np.uint32), np.arange(N, dtype=np.uint32))
    z = ((x * 7 + y * 13) % 97).astype(np.uint32)
    pos = np.stack([x * 16, y * 16, z], axis=-1).reshape(-1, 3).astype(np.uint32)
    idx = []
    for yy in range(N - 1):
        for xx in range(N - 1):
            i = yy * N + xx
            idx += [i, i + 1, i + N, i + 1, i + N + 1, i + N]
    idx = np.array(idx, dtype=np.uint32)
    blob, dpos, didx = rt(pos.reshape(-1), idx)
    assert canon(pos, idx) == canon(dpos, didx)
    assert set(map(tuple, dpos)) == set(map(tuple, pos[np.unique(idx)]))
    print("lattice_lossless ok,", len(blob), "bytes for", len(idx) // 3, "tris")


def test_scattered_connectivity():
    # long-range references (used to be the escape path; now meshopt's problem)
    n = 600
    pos = (np.arange(n * 3, dtype=np.uint32) * 37 % 1024).astype(np.uint32)
    idx = []
    for i in range(0, n - 3, 3):
        idx += [i, i + 1, i + 2]
    idx += [0, 5, n - 2]
    idx = np.array(idx, dtype=np.uint32)
    blob, dpos, didx = rt(pos.reshape(-1), idx)
    assert canon(pos.reshape(-1, 3), idx) == canon(dpos, didx)
    print("scattered_connectivity ok")


def test_qbits_16():
    rng = np.random.RandomState(11)
    n = 3000
    pos = rng.randint(0, 1 << 16, size=n * 3).astype(np.uint32)
    idx = rng.randint(0, n, size=3 * 2500).astype(np.uint32)
    blob, dpos, didx = rt(pos, idx, qbits=16)
    assert canon(dpos, didx) == nondegenerate(pos, idx)
    print("qbits_16 ok")


def test_f32_dequant():
    rng = np.random.RandomState(3)
    n = 2000
    pos = (rng.rand(n, 3) * np.array([2000.0, 900.0, 80.0]) - 400.0).astype(np.float32)
    idx = rng.randint(0, n, size=3 * 1500).astype(np.uint32)
    qbits = 14
    blob = mmc.encode_f32(pos.reshape(-1), idx, qbits)
    dpos, didx = mmc.decode(blob)
    assert dpos.dtype == np.float32 and didx.dtype == np.uint32
    # invert the stored dequant transform: decoded verts must sit on integer
    # grid points of the codec's own transform (mins + q*scale)
    mins = pos.min(0)
    qmax = float((1 << qbits) - 1)
    rngs = pos.max(0).astype(np.float64) - mins
    scale = (rngs / qmax).astype(np.float32)
    q_dec = (dpos.astype(np.float64) - mins) / scale.astype(np.float64)
    q_round = np.round(q_dec)
    assert np.abs(q_dec - q_round).max() < 1e-2, "decoded verts off-grid"
    assert q_round.min() >= 0 and q_round.max() <= qmax
    # quantized integers must match the encoder's quantization of the source
    q_src = np.floor((pos.astype(np.float64) - mins) / rngs * qmax + 0.5).clip(0, qmax)
    assert set(map(tuple, q_round.astype(np.int64))) <= set(map(tuple, q_src.astype(np.int64)))
    # geometric error bound: half a quantization step per axis (+fp slack)
    step = rngs / qmax
    recon = mins + q_src * step
    assert np.abs(recon - pos).max() <= (step.max() / 2) * 1.001 + 1e-4
    print("f32_dequant ok,", len(blob), "bytes; max quant step", step.round(5))


if __name__ == "__main__":
    test_single_triangle()
    test_random_soup_lossless()
    test_duplicate_and_orphan()
    test_lattice_lossless()
    test_scattered_connectivity()
    test_qbits_16()
    test_f32_dequant()
    print("ALL TESTS PASSED")
