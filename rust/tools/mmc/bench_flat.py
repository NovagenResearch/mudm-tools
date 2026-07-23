"""Can anything decode FASTER than meshopt at otherwise-similar performance?

meshopt's decoder per output byte = control-stream parse + variable-width
unpack + vertical delta accumulation. The only way to beat it is to do LESS
work per byte. Two candidates, both pushing entropy coding into the transport
layer (Content-Encoding zstd/gzip — decoded by the browser's native network
stack, off the app thread, i.e. "free" for a viewer; timed separately here for
non-browser consumers):

  FLAT-RAW   positions as GPU-uploadable quantized u16 (stride 8, KHR_mesh_
             quantization-compatible) + u16/u32 indices. In-app decode = ZERO
             (pointer cast / memcpy straight to the GPU upload).
  FLAT-DELTA positions as fixed-width u16 zigzag deltas per component (planar),
             indices as u32 zigzag deltas. In-app decode = one vectorized
             prefix-sum per stream. No control streams, no branches; valid for
             qbits <= 15 (delta zigzag must fit u16).

Both are lossy-by-quantization exactly like the doc's meshopt-u16 quick win
(lossless on an NG grid input). Baselines: the shipped meshopt pipeline
(raw f32) and the meshopt-u16 quick-win variant, plus libdraco decode for
context. Native kernels on both sides: meshopt = C SIMD via bindings;
flat = numpy C kernels (frombuffer/cumsum/astype) — a LOWER bound on a tuned
Rust/WASM SIMD implementation (numpy cumsum is scalar).

A third candidate emerged from the measurements (see §7 of
docs/mesh_codec_review.md):

  FLAT-HYBRID first-use vertex remap (encoder-side, decoder-free), then
              FLAT-DELTA positions + per-mesh index mode (1 header bit):
              FLAT-DELTA indices (raster/NG meshes) or the meshopt index
              codec (irregular/organic meshes). Beats meshopt-u16+zstd on
              wire size on every measured mesh while decoding faster.

The native decode race (this file's kernels vs meshopt's C decoder) needs
flatdec.c compiled next to this file:  cc -O3 -shared -fPIC flatdec.c -o flatdec.so

Usage: python bench_flat.py
"""

import ctypes
import os
import time
import numpy as np
import meshoptimizer as mo
import zstandard

try:
    import DracoPy
    HAVE_DRACO = True
except Exception:
    HAVE_DRACO = False

ZC = zstandard.ZstdCompressor(level=3)
ZD = zstandard.ZstdDecompressor()


# ---- meshes (same generators as bench.py) ----------------------------------

def terrain(N=256, qbits=10, seed=1):
    rng = np.random.RandomState(seed)
    z = np.zeros((N, N))
    for octv in range(1, 6):
        k = 2 ** octv
        small = rng.rand(k, k)
        idx = np.linspace(0, k - 1, N)
        xi, yi = np.meshgrid(idx, idx)
        x0 = np.floor(xi).astype(int); y0 = np.floor(yi).astype(int)
        x1 = np.minimum(x0 + 1, k - 1); y1 = np.minimum(y0 + 1, k - 1)
        fx = xi - x0; fy = yi - y0
        up = (small[y0, x0] * (1 - fx) * (1 - fy) + small[y0, x1] * fx * (1 - fy)
              + small[y1, x0] * (1 - fx) * fy + small[y1, x1] * fx * fy)
        z += up / k
    z = (z - z.min()) / (z.max() - z.min())
    qmax = (1 << qbits) - 1
    xs, ys = np.meshgrid(np.arange(N), np.arange(N))
    pos = np.stack([(xs / (N - 1) * qmax).round(), (ys / (N - 1) * qmax).round(),
                    (z * qmax).round()], axis=-1).reshape(-1, 3).astype(np.uint32)
    idx = []
    for y in range(N - 1):
        for x in range(N - 1):
            i = y * N + x
            idx += [i, i + 1, i + N, i + 1, i + N + 1, i + N]
    return pos, np.array(idx, dtype=np.uint32)


def icosphere(subdiv=5, qbits=10):
    t = (1 + 5 ** 0.5) / 2
    verts = [(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),(0,-1,t),(0,1,t),(0,-1,-t),(0,1,-t),(t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)]
    faces = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
             (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    verts = [np.array(v) / np.linalg.norm(v) for v in verts]
    cache = {}
    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = verts[a] + verts[b]
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
        return cache[key]
    for _ in range(subdiv):
        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    v = np.array(verts)
    qmax = (1 << qbits) - 1
    q = ((v - v.min(0)) / (v.max(0) - v.min(0)) * qmax + 0.5).astype(np.uint32).clip(0, qmax)
    return q.astype(np.uint32), np.array(faces, dtype=np.uint32).reshape(-1)


# ---- helpers ----------------------------------------------------------------

def run(fn, reps=7):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def zz16(d):  # zigzag i32 -> u16 (caller guarantees range)
    return (((d << 1) ^ (d >> 31)) & 0xFFFF).astype(np.uint16)


def unzz(z):  # u16/u32 -> i32/i64
    z = z.astype(np.int64)
    return (z >> 1) ^ -(z & 1)


# ---- candidates -------------------------------------------------------------

def flat_raw_encode(pos_q, idx):
    """GPU-ready: u16 positions padded to stride 8 + u16/u32 indices."""
    n = len(pos_q)
    p = np.zeros((n, 4), dtype=np.uint16)
    p[:, :3] = pos_q.astype(np.uint16)
    i = idx.astype(np.uint16) if n < 65536 else idx
    pos_b, idx_b = p.tobytes(), i.tobytes()
    return pos_b, idx_b, ZC.compress(pos_b), ZC.compress(idx_b)


def flat_raw_decode(pos_b, idx_b, n_verts, idx_u16):
    # in-app decode: pointer cast only (frombuffer is zero-copy)
    p = np.frombuffer(pos_b, dtype=np.uint16).reshape(n_verts, 4)
    i = np.frombuffer(idx_b, dtype=np.uint16 if idx_u16 else np.uint32)
    return p, i


def flat_delta_encode(pos_q, idx):
    """u16 zigzag deltas (planar per component) + u32 zigzag index deltas."""
    p = pos_q.astype(np.int32)
    d = np.diff(p, axis=0, prepend=np.zeros((1, 3), np.int32))
    pos_b = zz16(d).T.copy().tobytes()                      # planar: xxx..yyy..zzz
    di = np.diff(idx.astype(np.int64), prepend=np.int64(0))
    idx_b = (((di << 1) ^ (di >> 63)) & 0xFFFFFFFF).astype(np.uint32).tobytes()
    return pos_b, idx_b, ZC.compress(pos_b), ZC.compress(idx_b)


def flat_delta_decode(pos_b, idx_b, n_verts):
    z = np.frombuffer(pos_b, dtype=np.uint16).reshape(3, n_verts)
    p = np.cumsum(unzz(z), axis=1, dtype=np.int64).T.astype(np.uint16)
    zi = np.frombuffer(idx_b, dtype=np.uint32)
    i = np.cumsum(unzz(zi), dtype=np.int64).astype(np.uint32)
    return p, i


def first_use_remap(pos_q, idx):
    """Encoder-side remap: vertex ids in first-use order of the index stream.
    Free for the decoder; required for the meshopt index codec to perform,
    and harmless for FLAT-DELTA indices."""
    used, fp, inv = np.unique(idx, return_index=True, return_inverse=True)
    order = np.argsort(fp, kind="stable")
    nid = np.empty_like(order)
    nid[order] = np.arange(len(order))
    return pos_q[used[order]], nid[inv].astype(np.uint32)


def flat_hybrid_encode(pos_q, idx, zc=None):
    """FLAT-DELTA positions + smaller-of(FLAT-DELTA, meshopt) index stream."""
    zc = zc or ZC
    p2, i2 = first_use_remap(pos_q, idx)
    pos_b, idx_b, pos_z, idx_flat_z = flat_delta_encode(p2, i2)
    ib_mo = mo.encode_index_buffer(i2, len(i2), len(p2))
    idx_mo_z = zc.compress(ib_mo)
    if len(idx_mo_z) < len(idx_flat_z):
        return p2, i2, pos_b, pos_z, ib_mo, idx_mo_z, "meshopt"
    return p2, i2, pos_b, pos_z, idx_b, idx_flat_z, "flat"


# ---- native decode kernels (flatdec.so) --------------------------------------

def load_flatdec():
    here = os.path.dirname(os.path.abspath(__file__))
    so = os.path.join(here, "flatdec.so")
    if not os.path.exists(so):
        src = os.path.join(here, "flatdec.c")
        if os.system(f"cc -O3 -shared -fPIC {src} -o {so}") != 0:
            return None
    lib = ctypes.CDLL(so)
    for f in ("decode_pos_scalar", "decode_pos_neon", "decode_idx_scalar", "decode_idx_neon"):
        if hasattr(lib, f):
            getattr(lib, f).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    return lib


# ---- baselines --------------------------------------------------------------

def meshopt_pipeline(pos_f32_or_u16, idx, stride):
    n = len(pos_f32_or_u16)
    n_idx = len(idx)
    opt_idx = np.zeros(n_idx, dtype=np.uint32)
    mo.optimize_vertex_cache(opt_idx, idx, n_idx, n)
    opt_v = np.zeros_like(pos_f32_or_u16)
    nu = mo.optimize_vertex_fetch(opt_v, opt_idx, pos_f32_or_u16)
    vb = mo.encode_vertex_buffer(opt_v[:nu], nu, stride)
    ib = mo.encode_index_buffer(opt_idx, n_idx, nu)
    return vb, ib, nu


# ---- benchmark --------------------------------------------------------------

def bench(name, pos_q, idx, qbits):
    n_verts = len(pos_q)
    n_idx = len(idx)
    n_tris = n_idx // 3
    out_bytes = n_verts * 12 + n_idx * 4
    print(f"\n=== {name}: {n_verts:,} verts, {n_tris:,} tris ===")
    print(f"{'codec':30s} {'wire(z)':>10s} {'B/tri':>6s} {'enc':>8s} {'app-dec':>9s} {'+zstd':>9s}")

    def row(name, wire, t_enc, t_app, t_z):
        print(f"{name:30s} {wire:>10,d} {wire/n_tris:6.2f} {t_enc*1e3:7.2f}m {t_app*1e6:8.1f}u {((t_app+t_z))*1e6:8.1f}u")

    # meshopt raw f32 (current shipped config)
    pf = pos_q.astype(np.float32)
    (vb, ib, nu), te = run(lambda: meshopt_pipeline(pf, idx, 12))
    _, td = run(lambda: (mo.decode_vertex_buffer(nu, 12, vb, np.dtype((np.float32, 3))),
                         mo.decode_index_buffer(n_idx, 4, ib)))
    vz, iz = ZC.compress(vb), ZC.compress(ib)
    _, tz = run(lambda: (ZD.decompress(vz, max_output_size=len(vb)),
                         ZD.decompress(iz, max_output_size=len(ib))))
    row("meshopt raw f32 (+transport z)", len(vz) + len(iz), te, td, tz)

    # meshopt u16-quant (quick win)
    p16 = np.zeros((n_verts, 4), dtype=np.uint16)
    p16[:, :3] = pos_q.astype(np.uint16)
    (vb2, ib2, nu2), te2 = run(lambda: meshopt_pipeline(p16, idx, 8))
    _, td2 = run(lambda: (mo.decode_vertex_buffer(nu2, 8, vb2, np.dtype((np.uint16, 4))),
                          mo.decode_index_buffer(n_idx, 4, ib2)))
    vz2, iz2 = ZC.compress(vb2), ZC.compress(ib2)
    _, tz2 = run(lambda: (ZD.decompress(vz2, max_output_size=len(vb2)),
                          ZD.decompress(iz2, max_output_size=len(ib2))))
    row("meshopt u16 (+transport z)", len(vz2) + len(iz2), te2, td2, tz2)

    # FLAT-RAW
    (pb, ibb, pz, iz3), te3 = run(lambda: flat_raw_encode(pos_q, idx))
    idx_u16 = n_verts < 65536
    _, td3 = run(lambda: flat_raw_decode(pb, ibb, n_verts, idx_u16))
    _, tz3 = run(lambda: (ZD.decompress(pz, max_output_size=len(pb)),
                          ZD.decompress(iz3, max_output_size=len(ibb))))
    row("FLAT-RAW u16 (zero decode)", len(pz) + len(iz3), te3, td3, tz3)

    # FLAT-DELTA
    (pb4, ib4, pz4, iz4), te4 = run(lambda: flat_delta_encode(pos_q, idx))
    _, td4 = run(lambda: flat_delta_decode(pb4, ib4, n_verts))
    _, tz4 = run(lambda: (ZD.decompress(pz4, max_output_size=len(pb4)),
                          ZD.decompress(iz4, max_output_size=len(ib4))))
    row("FLAT-DELTA u16 (prefix-sum)", len(pz4) + len(iz4), te4, td4, tz4)

    # correctness of FLAT-DELTA roundtrip
    p_dec, i_dec = flat_delta_decode(pb4, ib4, n_verts)
    assert np.array_equal(p_dec.astype(np.uint32), pos_q.astype(np.uint32) & 0xFFFF)
    assert np.array_equal(i_dec, idx)

    # FLAT-HYBRID (first-use remap + flat positions + best index mode)
    lib = load_flatdec()
    (p2, i2, pb5, pz5, ib5, iz5, mode), te5 = run(lambda: flat_hybrid_encode(pos_q, idx))
    if lib is not None and hasattr(lib, "decode_pos_neon"):
        zp = np.frombuffer(pb5, dtype=np.uint16).copy()
        op = np.zeros(n_verts * 4, dtype=np.uint16)
        _, t_pos = run(lambda: lib.decode_pos_neon(zp.ctypes.data, n_verts, op.ctypes.data))
        if mode == "meshopt":
            _, t_idx = run(lambda: mo.decode_index_buffer(n_idx, 4, ib5))
        else:
            zi = np.frombuffer(ib5, dtype=np.uint32).copy()
            oi = np.zeros(n_idx, dtype=np.uint32)
            _, t_idx = run(lambda: lib.decode_idx_neon(zi.ctypes.data, n_idx, oi.ctypes.data))
        _, tz5 = run(lambda: (ZD.decompress(pz5, max_output_size=len(pb5)),
                              ZD.decompress(iz5, max_output_size=len(ib5) + 16)))
        row(f"FLAT-HYBRID ({mode} idx)", len(pz5) + len(iz5), te5, t_pos + t_idx, tz5)
        # correctness
        assert np.array_equal(op.reshape(-1, 4)[:, :3].astype(np.uint32),
                              p2.astype(np.uint32) & 0xFFFF)

    # libdraco decode for context
    if HAVE_DRACO:
        blob = DracoPy.encode(pos_q.astype(np.float32), idx.reshape(-1, 3),
                              quantization_bits=qbits, compression_level=7,
                              quantization_range=float((1 << qbits) - 1),
                              quantization_origin=np.zeros(3, dtype=np.float32))
        _, tdd = run(lambda: DracoPy.decode(blob))
        row("libdraco cl7 (context)", len(blob), 0, tdd, 0)

    # throughput summary
    print(f"  output payload {out_bytes/1e6:.1f} MB; app-decode GB/s: "
          f"meshopt {out_bytes/td/1e9:.1f}, flat-delta {out_bytes/td4/1e9:.1f}, "
          f"flat-raw inf (zero-copy)")


if __name__ == "__main__":
    pos, idx = terrain(256, 10)
    bench("terrain 256x256 q10", pos, idx, 10)
    pos, idx = icosphere(5, 10)
    bench("icosphere subdiv5 q10", pos.reshape(-1, 3), idx, 10)
    pos, idx = terrain(512, 14, seed=2)
    bench("terrain 512x512 q14", pos, idx, 14)
