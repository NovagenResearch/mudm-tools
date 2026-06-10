"""Codec benchmark: MMC (reference impl) vs Draco (native libdraco via DracoPy)
vs meshopt (native via meshoptimizer bindings, mirroring encoder_meshopt.rs).

Times: native C/C++ for Draco + meshopt. MMC times are the Python reference
(numpy + native zstd) — an upper bound on the Rust implementation's time.
"""

import time
import numpy as np
import zstandard
import DracoPy
import meshoptimizer as mo
import mmc

ZL = 3  # zstd level used for MMC and the meshopt+zstd variant


# ---------------- meshes ----------------

def terrain(N=256, qbits=10, seed=1):
    """fbm-ish TIN terrain, pre-quantized to the NG grid."""
    rng = np.random.RandomState(seed)
    z = np.zeros((N, N))
    for octv in range(1, 6):
        k = 2 ** octv
        small = rng.rand(k, k)
        # bilinear upsample
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
    pos = np.stack([
        (xs / (N - 1) * qmax).round(),
        (ys / (N - 1) * qmax).round(),
        (z * qmax).round(),
    ], axis=-1).reshape(-1, 3).astype(np.uint32)
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


def dup_heavy(N=128, qbits=10):
    """NG pathology from encoder_draco.rs rebake_guard_u32_duplicate_heavy."""
    base = []
    for y in range(N):
        for x in range(N):
            base.append([x * 7 % (1 << qbits), y * 7 % (1 << qbits), (x * 7 + y * 13) % 97])
    base = np.array(base, dtype=np.uint32)
    pos = np.concatenate([base, base])  # coincident twins
    k = len(base)
    idx, t = [], 0
    for y in range(N - 1):
        for x in range(N - 1):
            i = y * N + x; r = i + 1; d = i + N; dr = d + 1
            tw = (lambda v: v + k) if t % 2 == 0 else (lambda v: v)
            idx += [tw(i), r, tw(d), r, dr, tw(d)]
            t += 1
    return pos.reshape(-1), np.array(idx, dtype=np.uint32)


# ---------------- codecs ----------------

def run(fn, reps=3):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def bench_draco(pos_q, idx, qbits):
    pts = pos_q.reshape(-1, 3).astype(np.float32)
    faces = idx.reshape(-1, 3)
    blob, t_enc = run(lambda: DracoPy.encode(
        pts, faces, quantization_bits=qbits, compression_level=7,
        quantization_range=float((1 << qbits) - 1), quantization_origin=np.zeros(3, dtype=np.float32)))
    dec, t_dec = run(lambda: DracoPy.decode(blob))
    n_tri_dec = len(dec.faces)
    return len(blob), t_enc, t_dec, n_tri_dec


def bench_meshopt_current(pos_q, idx):
    """Mirrors encoder_meshopt.rs: cache-opt + fetch-opt + raw-f32 vertex codec."""
    verts = pos_q.reshape(-1, 3).astype(np.float32).copy()
    n_idx = len(idx)

    def enc():
        opt_idx = np.zeros(n_idx, dtype=np.uint32)
        mo.optimize_vertex_cache(opt_idx, idx, n_idx, len(verts))
        opt_verts = np.zeros_like(verts)
        n_unique = mo.optimize_vertex_fetch(opt_verts, opt_idx, verts)
        opt_verts = opt_verts[:n_unique]
        vb = mo.encode_vertex_buffer(opt_verts, n_unique, 12)
        ib = mo.encode_index_buffer(opt_idx, n_idx, n_unique)
        return vb, ib, n_unique

    (vb, ib, n_unique), t_enc = run(enc)

    def dec():
        # per-vertex structured dtype so the wrapper allocates n*12 bytes
        v = mo.decode_vertex_buffer(n_unique, 12, vb, np.dtype((np.float32, 3)))
        i = mo.decode_index_buffer(n_idx, 4, ib)
        return v, i

    _, t_dec = run(dec)
    size = len(vb) + len(ib)
    z = zstandard.ZstdCompressor(level=ZL)
    size_z = len(z.compress(vb)) + len(z.compress(ib))
    return size, size_z, t_enc, t_dec


def bench_meshopt_quantized(pos_q, idx, qbits):
    """Quick-win variant: u16 quantized positions (KHR_mesh_quantization style)."""
    assert qbits <= 16
    verts16 = pos_q.reshape(-1, 3).astype(np.uint16)
    # pad to 8-byte stride (meshopt codec wants stride % 4 == 0)
    n = len(verts16)
    padded = np.zeros((n, 4), dtype=np.uint16)
    padded[:, :3] = verts16
    n_idx = len(idx)

    def enc():
        opt_idx = np.zeros(n_idx, dtype=np.uint32)
        mo.optimize_vertex_cache(opt_idx, idx, n_idx, n)
        opt_verts = np.zeros_like(padded)
        n_unique = mo.optimize_vertex_fetch(opt_verts, opt_idx, padded)
        vb = mo.encode_vertex_buffer(opt_verts[:n_unique], n_unique, 8)
        ib = mo.encode_index_buffer(opt_idx, n_idx, n_unique)
        return vb, ib, n_unique

    (vb, ib, n_unique), t_enc = run(enc)

    def dec():
        v = mo.decode_vertex_buffer(n_unique, 8, vb, np.dtype((np.uint16, 4)))
        i = mo.decode_index_buffer(n_idx, 4, ib)
        return v, i

    _, t_dec = run(dec)
    size = len(vb) + len(ib)
    z = zstandard.ZstdCompressor(level=ZL)
    size_z = len(z.compress(vb)) + len(z.compress(ib))
    return size, size_z, t_enc, t_dec


def bench_mmc(pos_q, idx, qbits):
    blob, t_enc = run(lambda: mmc.encode_u32(pos_q, idx, qbits))
    _, t_dec = run(lambda: mmc.decode(blob))
    return len(blob), t_enc, t_dec


# ---------------- report ----------------

def fmt_row(name, n_tris, size, t_enc, t_dec, extra=""):
    bpt = size / n_tris
    print(f"{name:34s} {size:>10,d} B  {bpt:6.2f} B/tri  enc {t_enc*1e3:8.2f} ms  dec {t_dec*1e3:8.2f} ms {extra}")


def main():
    meshes = [
        ("terrain 256x256 q10 (NG)", *terrain(256, 10), 10),
        ("icosphere subdiv5 q10", *icosphere(5, 10), 10),
        ("dup-heavy 128x128 q10 (NG)", *dup_heavy(128, 10), 10),
        ("terrain 512x512 q14 (GLB-ish)", *terrain(512, 14, seed=2), 14),
    ]
    for name, pos, idx, qbits in meshes:
        n_tris = len(idx) // 3
        n_verts = len(pos.reshape(-1)) // 3
        raw = n_verts * 12 + n_tris * 12
        print(f"\n=== {name}: {n_verts:,} verts, {n_tris:,} tris, raw {raw:,} B ===")

        s, ze, te, td = bench_meshopt_current(pos, idx)
        fmt_row("meshopt current (raw f32, =crate)", n_tris, s, te, td, f"(+zstd: {ze:,} B)")

        s, ze, te, td = bench_meshopt_quantized(pos, idx, qbits)
        fmt_row("meshopt u16-quantized (quick win)", n_tris, s, te, td, f"(+zstd: {ze:,} B)")

        s, te, td, ntd = bench_draco(pos, idx, qbits)
        fmt_row("Draco native cl=7 (libdraco)", n_tris, s, te, td, f"(dec tris {ntd:,})")

        s, te, td = bench_mmc(pos, idx, qbits)
        fmt_row("MMC v1 (python ref, zstd-3)", n_tris, s, te, td)


if __name__ == "__main__":
    main()
