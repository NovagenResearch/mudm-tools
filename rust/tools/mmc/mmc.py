"""MMC v1 ("mudm mesh codec") — reference implementation.

Bit-exact reference for the Rust implementation in mudm-tools/rust/src/codec_mmc.rs.
Position-only triangle meshes, optimized for the two mudm-tools paths:

  * NG path:  pre-quantized u32 grid positions in [0, 2^qbits) — LOSSLESS.
  * GLB path: f32 world positions, bbox-quantized to qbits with the dequant
              transform stored in the header (decoder returns f32).

Architecture
============
Connectivity: the meshopt index codec (proven, ~1 B/tri pre-LZ, edge-FIFO based,
already a mudm-tools dependency, WASM decoder ships in viewers), wrapped in zstd
(0.00–0.27 B/tri post-zstd on our benches).

Positions: parallelogram prediction driven by the *decoded* (canonical) triangle
stream — no edgebreaker, no corner table. Vertices are emitted in first-seen
order of the canonical stream. Each new vertex is predicted as
    pred = pos[o1] + pos[o2] - pos[w]
where (o1, o2) are the already-known vertices of its triangle and w the vertex
opposite the shared edge (o2, o1) recorded from an earlier triangle. Fallbacks:
known neighbor -> last decoded -> 0. Residuals are zigzagged and stored as
byte-planes (component-major), then zstd.

Layout (little-endian)
======================
  offset size  field
  0      4     magic "MMC1"
  4      1     version (1)
  5      1     flags: bit0 = HAS_DEQUANT
  6      1     qbits (1..24)
  7      1     zstd level used (informational)
  8      4     n_verts  u32   (post dedup/compaction)
  12     4     n_tris   u32   (post degenerate-drop)
  [HAS_DEQUANT: 24 bytes = min xyz f32le, scale xyz f32le; world = min + q*scale]
  4            idx_zstd_len u32
  4            pos_zstd_len u32
  ...          idx zstd frame (meshopt index codec) | pos zstd frame (planes)

Residual planes: B = ceil((qbits+2)/8) bytes per component (prediction residual
spans [-(2^qbits-1), 2^qbits-1] after clamping pred to the grid; zigzag needs
qbits+1 bits, +1 spare). Stored as 3*B planes of n_verts bytes:
component-major, plane-minor, vertex-major within a plane.
"""

import struct
import numpy as np
import zstandard
import meshoptimizer as mo

MAGIC = b"MMC1"
VERSION = 1
FLAG_DEQUANT = 1
ZSTD_LEVEL = 3


def _sanitize(positions_q: np.ndarray, indices: np.ndarray):
    """Dedup identical quantized triples, drop degenerate tris, compact unused.

    Mirrors the semantics of the existing Draco NG path (encoder_draco.rs):
    coincident quantized vertices collapse, triangles that become lines/points
    drop, unused vertices are compacted away. Vertex order: ascending original
    dedup-rank (deterministic, vectorized).

    Returns (verts_q [V,3] u32, tris [T,3] u32).
    """
    pos = positions_q.reshape(-1, 3)
    tris = indices.reshape(-1, 3).astype(np.int64)

    # collapse coincident quantized positions to first-occurrence rank
    _, first_idx, inv = np.unique(pos, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first_idx, kind="stable")
    rank_of_unique = np.empty_like(order)
    rank_of_unique[order] = np.arange(len(order))
    vert_rank = rank_of_unique[inv]
    uniq_pos = pos[np.sort(first_idx)]

    t = vert_rank[tris]
    good = (t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])
    t = t[good]
    if len(t) == 0:
        raise ValueError("mesh has no non-degenerate faces after quantization")

    # compact unused vertices in FIRST-USE order of the index stream — the
    # meshopt index codec encodes ascending-first-use ids most cheaply (this is
    # what optimize_vertex_fetch produces in the meshopt pipeline)
    flat = t.reshape(-1)
    used, first_pos, inv2 = np.unique(flat, return_index=True, return_inverse=True)
    order = np.argsort(first_pos, kind="stable")      # unique-rank -> first-use rank
    new_id = np.empty_like(order)
    new_id[order] = np.arange(len(order))
    verts = uniq_pos[used[order]]
    return verts.astype(np.uint32), new_id[inv2].astype(np.uint32).reshape(-1, 3)


def _predict_walk(canon_tris: np.ndarray, n_verts: int, qbits: int,
                  verts=None, residual_planes=None):
    """Shared encoder/decoder walk over the canonical triangle stream.

    encode mode: pass verts [V,3] -> returns residual byte-planes (bytes)
    decode mode: pass residual_planes (bytes) -> returns verts [V,3] u32

    Walks triangles in order; the k-th distinct vertex id seen corresponds to
    the k-th residual slot. Prediction: parallelogram via the edge map, with
    known-neighbor / last-vertex / zero fallbacks. All arithmetic in i64;
    predictions clamped to [0, 2^qbits - 1].
    """
    qmax = (1 << qbits) - 1
    B = (qbits + 2 + 7) // 8
    encode = verts is not None
    if encode:
        v = verts.astype(np.int64)
        res = np.zeros((n_verts, 3), dtype=np.int64)
    else:
        a = np.frombuffer(residual_planes, dtype=np.uint8).reshape(3, B, n_verts)
        z = np.zeros((n_verts, 3), dtype=np.uint64)
        for c in range(3):
            for b in range(B):
                z[:, c] |= a[c, b].astype(np.uint64) << np.uint64(8 * b)
        zi = z.astype(np.int64)
        res = (zi >> 1) ^ -(zi & 1)          # un-zigzag, slot-ordered
        v = np.zeros((n_verts, 3), dtype=np.int64)

    slot_of = np.full(n_verts, -1, dtype=np.int64)   # vertex id -> emission slot
    known = np.zeros(n_verts, dtype=bool)
    edge = {}
    last = np.zeros(3, dtype=np.int64)
    n_seen = 0

    for t in canon_tris:
        ta, tb, tc = int(t[0]), int(t[1]), int(t[2])
        for (x, o1, o2) in ((ta, tb, tc), (tb, tc, ta), (tc, ta, tb)):
            if not known[x]:
                if known[o1] and known[o2]:
                    w = edge.get((o2, o1))
                    if w is not None and known[w]:
                        p = v[o1] + v[o2] - v[w]
                    else:
                        p = v[o1].copy()
                elif known[o1]:
                    p = v[o1].copy()
                elif known[o2]:
                    p = v[o2].copy()
                else:
                    p = last.copy()
                np.clip(p, 0, qmax, out=p)
                if encode:
                    res[n_seen] = v[x] - p
                else:
                    v[x] = p + res[n_seen]
                slot_of[x] = n_seen
                known[x] = True
                last = v[x].copy()
                n_seen += 1
        edge.setdefault((ta, tb), tc)
        edge.setdefault((tb, tc), ta)
        edge.setdefault((tc, ta), tb)

    if n_seen != n_verts:
        raise ValueError("index stream does not reference all vertices")

    if encode:
        z = ((res << 1) ^ (res >> 63)).astype(np.uint64)
        # reorder slot-major: slot k holds residual of k-th seen vertex
        # res was already filled slot-major above
        planes = []
        for c in range(3):
            for b in range(B):
                planes.append(((z[:, c] >> np.uint64(8 * b)) & np.uint64(0xFF)).astype(np.uint8))
        return np.concatenate(planes).tobytes()
    return v.astype(np.uint32)


def _encode_core(verts: np.ndarray, tris: np.ndarray, qbits: int, dequant, level=ZSTD_LEVEL) -> bytes:
    n_verts, n_tris = len(verts), len(tris)
    flat = np.ascontiguousarray(tris.reshape(-1), dtype=np.uint32)
    ib = mo.encode_index_buffer(flat, len(flat), n_verts)
    # canonical stream: what every decoder will see (meshopt may rotate tris)
    canon = np.asarray(mo.decode_index_buffer(len(flat), 4, ib), dtype=np.uint32).reshape(-1, 3)
    pos_raw = _predict_walk(canon, n_verts, qbits, verts=verts)

    c = zstandard.ZstdCompressor(level=level)
    idx_z = c.compress(ib)
    pos_z = c.compress(pos_raw)

    out = bytearray()
    out += MAGIC
    out += struct.pack("<BBBB", VERSION, FLAG_DEQUANT if dequant is not None else 0, qbits, level)
    out += struct.pack("<II", n_verts, n_tris)
    if dequant is not None:
        mins, scale = dequant
        out += mins.tobytes() + scale.tobytes()
    out += struct.pack("<II", len(idx_z), len(pos_z))
    out += idx_z
    out += pos_z
    return bytes(out)


def encode_u32(positions_q, indices, qbits: int, level: int = ZSTD_LEVEL) -> bytes:
    """NG path: lossless encode of pre-quantized u32 grid positions."""
    positions_q = np.ascontiguousarray(positions_q, dtype=np.uint32)
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    if positions_q.size == 0 or indices.size == 0:
        raise ValueError("empty mesh")
    if positions_q.max() >= (1 << qbits):
        raise ValueError("position exceeds 2^qbits")
    verts, tris = _sanitize(positions_q, indices)
    return _encode_core(verts, tris, qbits, None, level)


def encode_f32(positions, indices, qbits: int = 14, level: int = ZSTD_LEVEL) -> bytes:
    """GLB path: bbox-quantize f32 world positions to qbits (lossy, like Draco).

    Quantization: q = clamp(floor((v - min)/range * (2^qbits - 1) + 0.5), 0, qmax)
    (round-half-away-from-zero on non-negative values == f64::round in Rust).
    """
    positions = np.ascontiguousarray(positions, dtype=np.float32).reshape(-1, 3)
    indices = np.ascontiguousarray(indices, dtype=np.uint32)
    if positions.size == 0 or indices.size == 0:
        raise ValueError("empty mesh")
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    qmax = float((1 << qbits) - 1)
    rng = (maxs.astype(np.float64) - mins.astype(np.float64))
    safe = np.where(rng > 0, rng, 1.0)
    q = np.floor((positions.astype(np.float64) - mins) / safe * qmax + 0.5)
    q = np.where(rng > 0, q, 0.0)
    q = np.clip(q, 0, qmax).astype(np.uint32)
    scale = np.where(rng > 0, rng / qmax, 0.0).astype(np.float32)
    verts, tris = _sanitize(q, indices)
    return _encode_core(verts, tris, qbits, (mins.astype(np.float32), scale), level)


def decode(data: bytes):
    """Returns (positions, indices): u32 grid (NG) or f32 world (dequant mode)."""
    if data[:4] != MAGIC:
        raise ValueError("bad magic")
    version, flags, qbits, _level = struct.unpack_from("<BBBB", data, 4)
    if version != VERSION:
        raise ValueError("unsupported version")
    n_verts, n_tris = struct.unpack_from("<II", data, 8)
    off = 16
    dequant = None
    if flags & FLAG_DEQUANT:
        mins = np.frombuffer(data, dtype=np.float32, count=3, offset=off).copy()
        scale = np.frombuffer(data, dtype=np.float32, count=3, offset=off + 12).copy()
        dequant = (mins, scale)
        off += 24
    idx_len, pos_len = struct.unpack_from("<II", data, off)
    off += 8
    d = zstandard.ZstdDecompressor()
    B = (qbits + 2 + 7) // 8
    ib = d.decompress(data[off : off + idx_len], max_output_size=1 << 30)
    off += idx_len
    pos_raw = d.decompress(data[off : off + pos_len], max_output_size=3 * B * n_verts)

    canon = np.asarray(mo.decode_index_buffer(3 * n_tris, 4, ib), dtype=np.uint32).reshape(-1, 3)
    verts = _predict_walk(canon, n_verts, qbits, residual_planes=pos_raw)
    if dequant is not None:
        mins, scale = dequant
        pos = mins.astype(np.float64) + verts.astype(np.float64) * scale.astype(np.float64)
        return pos.astype(np.float32), canon
    return verts, canon
