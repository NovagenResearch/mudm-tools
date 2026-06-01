"""T6: opt-in `neuroglancer_uint64_sharded_v1` output equivalence.

Proves the sharded NG output is the loose per-segment output, merely
re-containerized: decoding the emitted `.shard` files (with a minimal,
spec-conformant reader) recovers every segment's manifest (`.index`) bytes and
concatenated Draco fragment bytes EXACTLY, matching the loose `{fid}` /
`{fid}.index` files. The loose Draco bytes are themselves locked by the Rust
rebake guards, so this closes the loop: sharding changes only the container.

Reader mirrors the neuroglancer spec: shard index (2^minishard_bits x (start,end)
u64le, relative to shard_index_end) -> per-minishard [3,n] u64le index (row0
delta labels, row1 start deltas, row2 sizes) -> manifest at the reconstructed
offset; the mesh fragment data sits immediately before the manifest (length =
sum of the manifest's fragment_offsets).
"""

import json
import struct
from pathlib import Path

import pytest

try:
    from mudm_tools._rs import StreamingTileGenerator

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extensions not compiled")

WORLD_BOUNDS = (0.0, 0.0, 0.0, 100.0, 200.0, 300.0)


def _make_dense_tin_feature(n_triangles=20, seed=42, tags=None):
    import random

    rng = random.Random(seed)
    xy, z, ring_lengths = [], [], []
    for _ in range(n_triangles):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        cz = rng.uniform(0.15, 0.85)
        d = 0.05
        xy.extend([cx - d, cy - d, cx + d, cy - d, cx, cy + d])
        z.extend([cz - d, cz + d, cz])
        ring_lengths.append(3)
    n = len(z)
    return {
        "geometry": xy,
        "geometry_z": z,
        "ring_lengths": ring_lengths,
        "type": 5,
        "tags": tags or {"name": "dense"},
        "minX": min(xy[i * 2] for i in range(n)),
        "minY": min(xy[i * 2 + 1] for i in range(n)),
        "minZ": min(z),
        "maxX": max(xy[i * 2] for i in range(n)),
        "maxY": max(xy[i * 2 + 1] for i in range(n)),
        "maxZ": max(z),
    }


def _corpus(n=8):
    return [
        _make_dense_tin_feature(
            12 + i, seed=100 + i, tags={"name": f"neuron_{i}", "volume": float(10 * i + 5)}
        )
        for i in range(n)
    ]


def _build(features, min_zoom=0, max_zoom=2):
    gen = StreamingTileGenerator(min_zoom=min_zoom, max_zoom=max_zoom)
    for f in features:
        gen.add_feature(f)
    return gen


def _loose_geometry(out_dir: Path) -> dict:
    """fid(str) -> (manifest_bytes, fragment_bytes) from loose {fid}/{fid}.index."""
    out = {}
    for idx in sorted(out_dir.glob("*.index")):
        fid = idx.stem
        data = out_dir / fid
        if data.exists():
            out[fid] = (idx.read_bytes(), data.read_bytes())
    return out


def _manifest_total_fragment_size(man: bytes) -> int:
    """Sum of all fragment_offsets across all LODs (= total fragment-data bytes)."""
    off = 3 * 4 + 3 * 4  # chunk_shape + grid_origin (f32 x3 each)
    (num_lods,) = struct.unpack_from("<I", man, off)
    off += 4
    off += num_lods * 4  # lod_scales
    off += num_lods * 3 * 4  # vertex_offsets
    num_frags = struct.unpack_from("<%dI" % num_lods, man, off)
    off += num_lods * 4
    total = 0
    for lod in range(num_lods):
        nf = num_frags[lod]
        off += nf * 3 * 4  # fragment_positions [3, nf]
        offs = struct.unpack_from("<%dI" % nf, man, off)
        off += nf * 4  # fragment_offsets [nf]
        total += sum(offs)
    return total


def _decode_sharded(out_dir: Path) -> dict:
    """Decode every .shard per the neuroglancer spec -> fid(str) -> (manifest, fragment)."""
    info = json.loads((out_dir / "info").read_text())
    sh = info["sharding"]
    assert sh["@type"] == "neuroglancer_uint64_sharded_v1"
    assert sh["data_encoding"] == "raw" and sh["minishard_index_encoding"] == "raw"
    num_minishards = 1 << sh["minishard_bits"]
    shard_index_end = num_minishards * 16

    decoded = {}
    for shard_file in sorted(out_dir.glob("*.shard")):
        b = shard_file.read_bytes()
        for m in range(num_minishards):
            start, end = struct.unpack_from("<QQ", b, m * 16)
            if start == end:
                continue
            mi = b[shard_index_end + start : shard_index_end + end]
            n = len(mi) // 24
            u = struct.unpack_from("<%dQ" % (3 * n), mi, 0)
            row0, row1, row2 = u[0:n], u[n : 2 * n], u[2 * n : 3 * n]
            labels, acc = [], 0
            for i in range(n):
                acc = row0[i] if i == 0 else acc + row0[i]
                labels.append(acc)
            sizes = list(row2)
            starts = []
            for i in range(n):
                starts.append(
                    shard_index_end + row1[0] if i == 0 else starts[-1] + sizes[i - 1] + row1[i]
                )
            for i in range(n):
                man = b[starts[i] : starts[i] + sizes[i]]
                total_frag = _manifest_total_fragment_size(man)
                frag = b[starts[i] - total_frag : starts[i]]
                decoded[str(labels[i])] = (man, frag)
    return decoded


@pytest.mark.parametrize("minishard_bits,shard_bits", [(4, 0), (3, 2), (6, 0)])
def test_sharded_equivalent_to_loose(tmp_path, minishard_bits, shard_bits):
    feats = _corpus(8)

    out_loose = tmp_path / "loose"
    n_loose = _build(feats).generate_neuroglancer_multilod(str(out_loose), WORLD_BOUNDS)
    loose = _loose_geometry(out_loose)
    assert loose, "loose run produced no geometry"

    out_sharded = tmp_path / "sharded"
    n_sharded = _build(feats).generate_neuroglancer_multilod(
        str(out_sharded),
        WORLD_BOUNDS,
        sharded=True,
        minishard_bits=minishard_bits,
        shard_bits=shard_bits,
    )

    assert n_loose == n_sharded
    # Sharded dir must NOT contain loose per-segment files, and MUST contain shards.
    assert not list(out_sharded.glob("*.index")), "sharded mode must not write loose .index files"
    assert list(out_sharded.glob("*.shard")), "sharded mode must write .shard files"

    info = json.loads((out_sharded / "info").read_text())
    assert info["sharding"]["@type"] == "neuroglancer_uint64_sharded_v1"
    assert info["sharding"]["hash"] == "murmurhash3_x86_128"

    decoded = _decode_sharded(out_sharded)
    assert set(decoded) == set(loose), f"segment id sets differ: {set(decoded) ^ set(loose)}"
    for fid, (man_l, frag_l) in loose.items():
        man_s, frag_s = decoded[fid]
        assert man_s == man_l, f"manifest bytes differ for segment {fid}"
        assert frag_s == frag_l, f"fragment (Draco) bytes differ for segment {fid}"

    # segment_properties is emitted identically in both modes.
    sp_loose = (out_loose / "segment_properties" / "info").read_text()
    sp_sharded = (out_sharded / "segment_properties" / "info").read_text()
    assert sp_loose == sp_sharded, "segment_properties must be identical loose vs sharded"


def test_sharded_is_deterministic(tmp_path):
    feats = _corpus(6)
    out1 = tmp_path / "s1"
    out2 = tmp_path / "s2"
    _build(feats).generate_neuroglancer_multilod(
        str(out1), WORLD_BOUNDS, sharded=True, minishard_bits=4
    )
    _build(feats).generate_neuroglancer_multilod(
        str(out2), WORLD_BOUNDS, sharded=True, minishard_bits=4
    )
    shards1 = sorted(out1.glob("*.shard"))
    shards2 = sorted(out2.glob("*.shard"))
    assert shards1 and [p.name for p in shards1] == [p.name for p in shards2]
    for a, b in zip(shards1, shards2):
        assert a.read_bytes() == b.read_bytes(), f"shard {a.name} not deterministic across runs"
