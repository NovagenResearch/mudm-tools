#!/usr/bin/env python3
"""Benchmark per-feature mesh extraction across output formats.

Measures two operations:
  A. Bulk extraction — extract ALL features' meshes at finest LOD. Total time.
  B. Single-feature random access — extract one feature by ID. Median over N samples.

Formats compared:
  - Feature PBF3 (multilod): one .pbf3 per feature, protobuf decode
  - Parquet ZSTD: shared partitioned table, filter by feature_id
  - Arrow IPC: shared partitioned table (mmap), filter by feature_id
  - Neuroglancer multilod: .index + data per segment, Draco decode

Usage:
    # Generate missing formats first (re-ingests OBJs, ~30s)
    uv run python scripts/benchmark_feature_read.py --brain 2016-10-31 --generate

    # Run benchmark
    uv run python scripts/benchmark_feature_read.py --brain 2016-10-31 --zoom 3 --iters 5
"""

from __future__ import annotations

import argparse
import gc
import random
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f}us"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def _median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


def _parse_index_file(data: bytes) -> dict:
    """Parse a neuroglancer multilod .index binary manifest file."""
    off = 0

    def read_f32(n=1):
        nonlocal off
        vals = struct.unpack_from(f"<{n}f", data, off)
        off += 4 * n
        return vals if n > 1 else vals[0]

    def read_u32(n=1):
        nonlocal off
        vals = struct.unpack_from(f"<{n}I", data, off)
        off += 4 * n
        return vals if n > 1 else vals[0]

    chunk_shape = read_f32(3)
    grid_origin = read_f32(3)
    num_lods = read_u32()

    lod_scales = read_f32(num_lods)
    if num_lods == 1:
        lod_scales = (lod_scales,) if isinstance(lod_scales, float) else lod_scales

    vertex_offsets = []
    for _ in range(num_lods):
        vertex_offsets.append(read_f32(3))

    num_fragments_per_lod = read_u32(num_lods)
    if num_lods == 1:
        num_fragments_per_lod = (
            (num_fragments_per_lod,)
            if isinstance(num_fragments_per_lod, int)
            else num_fragments_per_lod
        )

    lods = []
    for lod_idx in range(num_lods):
        nf = num_fragments_per_lod[lod_idx]
        x_vals = read_u32(nf) if nf > 0 else ()
        y_vals = read_u32(nf) if nf > 0 else ()
        z_vals = read_u32(nf) if nf > 0 else ()
        if nf == 1:
            x_vals = (x_vals,) if isinstance(x_vals, int) else x_vals
            y_vals = (y_vals,) if isinstance(y_vals, int) else y_vals
            z_vals = (z_vals,) if isinstance(z_vals, int) else z_vals
        positions = list(zip(x_vals, y_vals, z_vals))
        frag_offsets = read_u32(nf) if nf > 0 else ()
        if nf == 1:
            frag_offsets = (
                (frag_offsets,) if isinstance(frag_offsets, int) else frag_offsets
            )
        lods.append({
            "positions": positions,
            "offsets": list(frag_offsets),
        })

    return {
        "chunk_shape": chunk_shape,
        "grid_origin": grid_origin,
        "num_lods": num_lods,
        "lod_scales": list(lod_scales),
        "vertex_offsets": vertex_offsets,
        "num_fragments_per_lod": list(num_fragments_per_lod),
        "lods": lods,
    }


def _merge_feature_fragments(fragments):
    """Merge (positions, indices) from multiple tile fragments into one mesh."""
    if not fragments:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint32)
    if len(fragments) == 1:
        return fragments[0]
    all_pos = []
    all_idx = []
    vertex_offset = 0
    for positions, indices in fragments:
        all_pos.append(positions)
        if len(indices) > 0:
            all_idx.append(indices + vertex_offset)
        vertex_offset += positions.shape[0]
    merged_pos = np.concatenate(all_pos)
    merged_idx = np.concatenate(all_idx) if all_idx else np.empty(0, dtype=np.uint32)
    return merged_pos, merged_idx


# ---------------------------------------------------------------------------
# Feature PBF3 benchmark
# ---------------------------------------------------------------------------

def bench_feature_pbf3(feat_pbf3_dir: Path, *, n_warmup: int, n_iters: int,
                      n_samples: int, seed: int):
    """Benchmark Feature PBF3 decode — per-feature .pbf3 files."""
    from mudm_tools.tiling3d.reader3d import decode_tile

    pbf3_files = sorted(feat_pbf3_dir.glob("*.pbf3"))
    if not pbf3_files:
        return None

    # Map feature_id -> bytes (pre-load)
    feature_ids = []
    feature_bytes: dict[int, bytes] = {}
    disk_size = 0
    for f in pbf3_files:
        fid = int(f.stem)
        data = f.read_bytes()
        feature_bytes[fid] = data
        feature_ids.append(fid)
        disk_size += len(data)

    def extract_feature(data: bytes):
        """Decode .pbf3, extract finest LOD, merge fragments."""
        layers = decode_tile(data)
        if not layers:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.uint32)
        # lod_0 = finest LOD (layer[0] for multilod, only layer for v1)
        finest = layers[0]
        fragments = []
        for feat in finest["features"]:
            pos_bytes = feat.get("mesh_positions", b"")
            idx_bytes = feat.get("mesh_indices", b"")
            if pos_bytes:
                positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
                indices = (
                    np.frombuffer(idx_bytes, dtype=np.uint32)
                    if idx_bytes
                    else np.empty(0, dtype=np.uint32)
                )
                fragments.append((positions, indices))
        return _merge_feature_fragments(fragments)

    def run_bulk():
        results = {}
        for fid, data in feature_bytes.items():
            results[fid] = extract_feature(data)
        return results

    # Warmup
    for _ in range(n_warmup):
        run_bulk()

    # Bulk timing
    gc.collect()
    bulk_times = []
    bulk_result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        bulk_result = run_bulk()
        bulk_times.append((time.perf_counter() - t0) * 1000)

    n_features = len(bulk_result)
    total_verts = sum(m[0].shape[0] for m in bulk_result.values())
    total_idx = sum(len(m[1]) for m in bulk_result.values())
    del bulk_result
    gc.collect()

    # Single-feature timing
    rng = random.Random(seed)
    sample_ids = [rng.choice(feature_ids) for _ in range(n_samples)]

    for fid in sample_ids[:min(3, len(sample_ids))]:
        extract_feature(feature_bytes[fid])

    single_times = []
    for fid in sample_ids:
        t0 = time.perf_counter()
        extract_feature(feature_bytes[fid])
        single_times.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "Feature PBF3",
        "files": len(pbf3_files),
        "disk_bytes": disk_size,
        "features": n_features,
        "vertices": total_verts,
        "indices": total_idx,
        "bulk_median_ms": _median(bulk_times),
        "bulk_times_ms": bulk_times,
        "single_median_ms": _median(single_times),
        "single_times_ms": single_times,
        "note": "bytes pre-loaded",
    }


# ---------------------------------------------------------------------------
# Parquet feature benchmark
# ---------------------------------------------------------------------------

def bench_parquet_feature(pq_dir: Path, zoom: int, *, n_warmup: int, n_iters: int,
                          n_samples: int, seed: int):
    """Benchmark Parquet per-feature extraction (bulk includes disk I/O)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    zoom_dir = pq_dir / f"zoom={zoom}"
    if not zoom_dir.exists():
        return None
    pq_files = sorted(str(f) for f in zoom_dir.glob("*.parquet"))
    if not pq_files:
        return None
    disk_size = sum(Path(f).stat().st_size for f in pq_files)

    def load_table():
        partitioning = ds.HivePartitioning(pa.schema([("zoom", pa.int32())]))
        dataset = ds.dataset(
            pq_files, format="parquet",
            partitioning=partitioning,
            partition_base_dir=str(pq_dir),
        )
        return dataset.to_table()

    def extract_all_features(table):
        """Group by feature_id, merge fragments per feature."""
        fids = table.column("feature_id").to_pylist()
        positions_list = table.column("positions").to_pylist()
        indices_list = table.column("indices").to_pylist()

        groups: dict[int, list] = {}
        for i in range(len(fids)):
            fid = fids[i]
            pos = np.frombuffer(positions_list[i], dtype=np.float32).reshape(-1, 3)
            idx = np.frombuffer(indices_list[i], dtype=np.uint32)
            if fid not in groups:
                groups[fid] = []
            groups[fid].append((pos, idx))

        results = {}
        for fid, frags in groups.items():
            results[fid] = _merge_feature_fragments(frags)
        return results

    def extract_single_feature(table, target_fid):
        """Filter table for one feature_id using Arrow compute."""
        filtered = table.filter(pc.field("feature_id") == target_fid)
        pos_list = filtered.column("positions").to_pylist()
        idx_list = filtered.column("indices").to_pylist()
        fragments = []
        for i in range(len(pos_list)):
            pos = np.frombuffer(pos_list[i], dtype=np.float32).reshape(-1, 3)
            idx = np.frombuffer(idx_list[i], dtype=np.uint32)
            fragments.append((pos, idx))
        return _merge_feature_fragments(fragments)

    # Warmup
    for _ in range(n_warmup):
        t = load_table()
        extract_all_features(t)
        del t

    # Bulk timing (includes disk I/O)
    gc.collect()
    bulk_times = []
    bulk_result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        table = load_table()
        bulk_result = extract_all_features(table)
        bulk_times.append((time.perf_counter() - t0) * 1000)
        del table

    n_features = len(bulk_result)
    total_verts = sum(m[0].shape[0] for m in bulk_result.values())
    total_idx = sum(len(m[1]) for m in bulk_result.values())
    feature_ids = list(bulk_result.keys())
    del bulk_result
    gc.collect()

    # Single-feature timing (pre-load table, time only extraction)
    table = load_table()
    rng = random.Random(seed)
    sample_ids = [rng.choice(feature_ids) for _ in range(n_samples)]

    for fid in sample_ids[:min(3, len(sample_ids))]:
        extract_single_feature(table, fid)

    single_times = []
    for fid in sample_ids:
        t0 = time.perf_counter()
        extract_single_feature(table, fid)
        single_times.append((time.perf_counter() - t0) * 1000)

    del table
    gc.collect()

    return {
        "name": "Parquet (ZSTD)",
        "files": len(pq_files),
        "disk_bytes": disk_size,
        "features": n_features,
        "vertices": total_verts,
        "indices": total_idx,
        "bulk_median_ms": _median(bulk_times),
        "bulk_times_ms": bulk_times,
        "single_median_ms": _median(single_times),
        "single_times_ms": single_times,
        "note": "bulk=disk I/O, single=pre-loaded",
    }


# ---------------------------------------------------------------------------
# Arrow IPC feature benchmark
# ---------------------------------------------------------------------------

def bench_arrow_feature(pq_dir: Path, zoom: int, *, n_warmup: int, n_iters: int,
                        n_samples: int, seed: int):
    """Benchmark Arrow IPC per-feature extraction (bulk includes disk I/O)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    zoom_dir = pq_dir / f"zoom={zoom}"
    if not zoom_dir.exists():
        return None
    arrow_files = sorted(str(f) for f in zoom_dir.glob("*.arrow"))
    if not arrow_files:
        return None
    disk_size = sum(Path(f).stat().st_size for f in arrow_files)

    def load_table():
        partitioning = ds.HivePartitioning(pa.schema([("zoom", pa.int32())]))
        dataset = ds.dataset(
            arrow_files, format="ipc",
            partitioning=partitioning,
            partition_base_dir=str(pq_dir),
        )
        return dataset.to_table()

    def extract_all_features(table):
        fids = table.column("feature_id").to_pylist()
        positions_list = table.column("positions").to_pylist()
        indices_list = table.column("indices").to_pylist()

        groups: dict[int, list] = {}
        for i in range(len(fids)):
            fid = fids[i]
            pos = np.frombuffer(positions_list[i], dtype=np.float32).reshape(-1, 3)
            idx = np.frombuffer(indices_list[i], dtype=np.uint32)
            if fid not in groups:
                groups[fid] = []
            groups[fid].append((pos, idx))

        results = {}
        for fid, frags in groups.items():
            results[fid] = _merge_feature_fragments(frags)
        return results

    def extract_single_feature(table, target_fid):
        filtered = table.filter(pc.field("feature_id") == target_fid)
        pos_list = filtered.column("positions").to_pylist()
        idx_list = filtered.column("indices").to_pylist()
        fragments = []
        for i in range(len(pos_list)):
            pos = np.frombuffer(pos_list[i], dtype=np.float32).reshape(-1, 3)
            idx = np.frombuffer(idx_list[i], dtype=np.uint32)
            fragments.append((pos, idx))
        return _merge_feature_fragments(fragments)

    # Warmup
    for _ in range(n_warmup):
        t = load_table()
        extract_all_features(t)
        del t

    # Bulk timing
    gc.collect()
    bulk_times = []
    bulk_result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        table = load_table()
        bulk_result = extract_all_features(table)
        bulk_times.append((time.perf_counter() - t0) * 1000)
        del table

    n_features = len(bulk_result)
    total_verts = sum(m[0].shape[0] for m in bulk_result.values())
    total_idx = sum(len(m[1]) for m in bulk_result.values())
    feature_ids = list(bulk_result.keys())
    del bulk_result
    gc.collect()

    # Single-feature timing
    table = load_table()
    rng = random.Random(seed)
    sample_ids = [rng.choice(feature_ids) for _ in range(n_samples)]

    for fid in sample_ids[:min(3, len(sample_ids))]:
        extract_single_feature(table, fid)

    single_times = []
    for fid in sample_ids:
        t0 = time.perf_counter()
        extract_single_feature(table, fid)
        single_times.append((time.perf_counter() - t0) * 1000)

    del table
    gc.collect()

    return {
        "name": "Arrow IPC (mmap)",
        "files": len(arrow_files),
        "disk_bytes": disk_size,
        "features": n_features,
        "vertices": total_verts,
        "indices": total_idx,
        "bulk_median_ms": _median(bulk_times),
        "bulk_times_ms": bulk_times,
        "single_median_ms": _median(single_times),
        "single_times_ms": single_times,
        "note": "bulk=disk I/O, single=pre-loaded",
    }


# ---------------------------------------------------------------------------
# Neuroglancer multilod benchmark
# ---------------------------------------------------------------------------

def bench_neuroglancer(ng_dir: Path, *, n_warmup: int, n_iters: int,
                       n_samples: int, seed: int):
    """Benchmark Neuroglancer multilod per-feature extraction."""
    try:
        import DracoPy
    except ImportError:
        print("  (DracoPy not installed, skipping Neuroglancer)")
        return None

    # Check for multilod format (.index files)
    index_files = sorted(ng_dir.glob("*.index"))
    if not index_files:
        print("  (no .index files — legacy format? Use --generate to create multilod)")
        return None

    # Pre-load all segment data
    segment_ids = []
    segment_data: dict[int, tuple[bytes, bytes]] = {}
    disk_size = 0
    for idx_file in index_files:
        seg_id = int(idx_file.stem)
        data_file = ng_dir / str(seg_id)
        if not data_file.exists():
            continue
        idx_bytes = idx_file.read_bytes()
        mesh_bytes = data_file.read_bytes()
        segment_data[seg_id] = (idx_bytes, mesh_bytes)
        segment_ids.append(seg_id)
        disk_size += len(idx_bytes) + len(mesh_bytes)

    if not segment_ids:
        return None

    def extract_feature(index_bytes: bytes, mesh_bytes: bytes):
        """Parse .index, decode finest LOD Draco fragments, merge."""
        parsed = _parse_index_file(index_bytes)
        # LOD 0 = finest
        lod0 = parsed["lods"][0]
        n_frags = parsed["num_fragments_per_lod"][0]

        fragments = []
        offset = 0
        for i in range(n_frags):
            frag_size = lod0["offsets"][i]
            if frag_size == 0:
                continue
            frag_data = mesh_bytes[offset:offset + frag_size]
            offset += frag_size
            decoded = DracoPy.decode(frag_data)
            positions = np.array(decoded.points, dtype=np.float32).reshape(-1, 3)
            faces = np.array(decoded.faces, dtype=np.uint32).ravel()
            fragments.append((positions, faces))

        return _merge_feature_fragments(fragments)

    def run_bulk():
        results = {}
        for seg_id, (idx_bytes, mesh_bytes) in segment_data.items():
            results[seg_id] = extract_feature(idx_bytes, mesh_bytes)
        return results

    # Warmup
    for _ in range(n_warmup):
        run_bulk()

    # Bulk timing
    gc.collect()
    bulk_times = []
    bulk_result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        bulk_result = run_bulk()
        bulk_times.append((time.perf_counter() - t0) * 1000)

    n_features = len(bulk_result)
    total_verts = sum(m[0].shape[0] for m in bulk_result.values())
    total_idx = sum(len(m[1]) for m in bulk_result.values())
    del bulk_result
    gc.collect()

    # Single-feature timing
    rng = random.Random(seed)
    sample_ids = [rng.choice(segment_ids) for _ in range(n_samples)]

    for seg_id in sample_ids[:min(3, len(sample_ids))]:
        idx_b, mesh_b = segment_data[seg_id]
        extract_feature(idx_b, mesh_b)

    single_times = []
    for seg_id in sample_ids:
        idx_b, mesh_b = segment_data[seg_id]
        t0 = time.perf_counter()
        extract_feature(idx_b, mesh_b)
        single_times.append((time.perf_counter() - t0) * 1000)

    return {
        "name": "Neuroglancer (Draco)",
        "files": len(segment_ids) * 2,
        "disk_bytes": disk_size,
        "features": n_features,
        "vertices": total_verts,
        "indices": total_idx,
        "bulk_median_ms": _median(bulk_times),
        "bulk_times_ms": bulk_times,
        "single_median_ms": _median(single_times),
        "single_times_ms": single_times,
        "note": "bytes pre-loaded",
    }


# ---------------------------------------------------------------------------
# Generate formats (--generate)
# ---------------------------------------------------------------------------

def generate_formats(brain_id: str, max_zoom: int = 3):
    """Re-ingest OBJs and generate Feature PBF3 v2 + Neuroglancer multilod."""
    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

    sys.path.insert(0, str(_ROOT / "scripts"))
    from obj_to_microjson import fetch_allen_ontology, match_region

    obj_dir = _ROOT / "data" / "mouselight" / brain_id
    output_dir = _ROOT / "data" / "mouselight" / "tiles" / brain_id

    obj_paths = sorted(obj_dir.glob("*.obj"))
    if not obj_paths:
        print(f"ERROR: No .obj files found in {obj_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(obj_paths)} OBJ files in {obj_dir}")

    # Fetch ontology for tags
    print("Fetching Allen CCF ontology...")
    t0 = time.perf_counter()
    ontology = fetch_allen_ontology()
    print(f"  Loaded {len(ontology)} entries ({_fmt_time((time.perf_counter() - t0) * 1000)})")

    def _build_tags(obj_path: Path) -> dict:
        stem = obj_path.stem
        tags: dict = {"mesh_name": stem, "source": obj_path.name}
        region = match_region(stem, ontology)
        if region:
            tags["ccf_id"] = region["id"]
            tags["name"] = region["name"]
            tags["acronym"] = region["acronym"]
            if region["parent_structure_id"] is not None:
                tags["parent_id"] = region["parent_structure_id"]
            if region["color_hex_triplet"]:
                tags["color"] = f"#{region['color_hex_triplet']}"
        return tags

    tags_list = [_build_tags(p) for p in obj_paths]
    path_strs = [str(p) for p in obj_paths]

    # Scan bounds
    print("Scanning bounds...")
    t0 = time.perf_counter()
    bounds = scan_obj_bounds(path_strs)
    print(f"  Bounds scanned ({_fmt_time((time.perf_counter() - t0) * 1000)})")

    # --- Feature PBF3 v2 (multilod=True) ---
    feat_pbf3_dir = output_dir / "mudm_feature_pbf3"
    if feat_pbf3_dir.exists():
        shutil.rmtree(feat_pbf3_dir)
    feat_pbf3_dir.mkdir(parents=True, exist_ok=True)

    gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    print(f"\nIngesting OBJs for Feature PBF3 (parallel Rust, zoom 0-{max_zoom})...")
    t0 = time.perf_counter()
    gen.add_obj_files(path_strs, bounds, tags_list)
    t_ingest = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest * 1000)}")

    print("Generating Feature PBF3 v2 (multilod=True)...")
    t0 = time.perf_counter()
    n_feat = gen.generate_feature_pbf3(str(feat_pbf3_dir), bounds)
    t_gen = time.perf_counter() - t0
    print(f"  {n_feat} features in {_fmt_time(t_gen * 1000)}")
    del gen

    # --- Neuroglancer multilod ---
    ng_dir = output_dir / "neuroglancer"
    if ng_dir.exists():
        shutil.rmtree(ng_dir)
    ng_dir.mkdir(parents=True, exist_ok=True)

    gen_ng = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    print(f"\nIngesting OBJs for Neuroglancer multilod...")
    t0 = time.perf_counter()
    gen_ng.add_obj_files(path_strs, bounds, tags_list)
    t_ingest = time.perf_counter() - t0
    print(f"  Ingest: {_fmt_time(t_ingest * 1000)}")

    print("Generating Neuroglancer multilod_draco...")
    t0 = time.perf_counter()
    n_seg = gen_ng.generate_neuroglancer_multilod(str(ng_dir), bounds)
    t_gen = time.perf_counter() - t0
    print(f"  {n_seg} segments in {_fmt_time(t_gen * 1000)}")
    del gen_ng

    print("\nGeneration complete.")


# ---------------------------------------------------------------------------
# CLI + Summary
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark per-feature mesh extraction across formats",
    )
    parser.add_argument(
        "--brain", type=str, default="2016-10-31",
        help="Brain date folder (default: 2016-10-31)",
    )
    parser.add_argument(
        "--zoom", type=int, default=3,
        help="Zoom level for Parquet/Arrow extraction (default: 3)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--iters", type=int, default=5,
        help="Benchmark iterations for bulk (default: 5)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Random features for single-feature timing (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Regenerate Feature PBF3 v2 + Neuroglancer multilod before benchmarking",
    )
    args = parser.parse_args()

    base = _ROOT / "data" / "mouselight" / "tiles" / args.brain

    # Generate if requested
    if args.generate:
        generate_formats(args.brain, max_zoom=args.zoom)

    if not base.exists():
        print(f"ERROR: {base} not found", file=sys.stderr)
        sys.exit(1)

    zoom = args.zoom
    print(f"\nBenchmark: per-feature mesh extraction at zoom={zoom} for {args.brain}")
    print(f"Warmup: {args.warmup}, Bulk iters: {args.iters}, "
          f"Single samples: {args.n_samples}, Seed: {args.seed}")
    print()

    bench_kwargs = dict(
        n_warmup=args.warmup, n_iters=args.iters,
        n_samples=args.n_samples, seed=args.seed,
    )

    results = []

    # Feature PBF3
    feat_pbf3_dir = base / "mudm_feature_pbf3"
    print("--- Feature PBF3 ---")
    r = bench_feature_pbf3(feat_pbf3_dir, **bench_kwargs)
    if r:
        results.append(r)
        print(f"  {r['files']} files, {_fmt_bytes(r['disk_bytes'])} on disk")
        print(f"  Features: {r['features']:,}  Vertices: {r['vertices']:,}  "
              f"Indices: {r['indices']:,}")
        print(f"  Bulk median: {_fmt_time(r['bulk_median_ms'])}  ({r['note']})")
        print(f"  Single median: {_fmt_time(r['single_median_ms'])}")
    else:
        print("  (not found)")
    print()

    # Parquet
    pq_dir = base / "parquet_partitioned"
    print("--- Parquet (ZSTD) ---")
    r = bench_parquet_feature(pq_dir, zoom, **bench_kwargs)
    if r:
        results.append(r)
        print(f"  {r['files']} file(s), {_fmt_bytes(r['disk_bytes'])} on disk")
        print(f"  Features: {r['features']:,}  Vertices: {r['vertices']:,}  "
              f"Indices: {r['indices']:,}")
        print(f"  Bulk median: {_fmt_time(r['bulk_median_ms'])}  ({r['note']})")
        print(f"  Single median: {_fmt_time(r['single_median_ms'])}")
    else:
        print("  (not found)")
    print()

    # Arrow IPC
    print("--- Arrow IPC (mmap) ---")
    r = bench_arrow_feature(pq_dir, zoom, **bench_kwargs)
    if r:
        results.append(r)
        print(f"  {r['files']} file(s), {_fmt_bytes(r['disk_bytes'])} on disk")
        print(f"  Features: {r['features']:,}  Vertices: {r['vertices']:,}  "
              f"Indices: {r['indices']:,}")
        print(f"  Bulk median: {_fmt_time(r['bulk_median_ms'])}  ({r['note']})")
        print(f"  Single median: {_fmt_time(r['single_median_ms'])}")
    else:
        print("  (not found)")
    print()

    # Neuroglancer
    ng_dir = base / "neuroglancer"
    print("--- Neuroglancer multilod (Draco) ---")
    r = bench_neuroglancer(ng_dir, **bench_kwargs)
    if r:
        results.append(r)
        print(f"  {r['files']} files, {_fmt_bytes(r['disk_bytes'])} on disk")
        print(f"  Features: {r['features']:,}  Vertices: {r['vertices']:,}  "
              f"Indices: {r['indices']:,}")
        print(f"  Bulk median: {_fmt_time(r['bulk_median_ms'])}  ({r['note']})")
        print(f"  Single median: {_fmt_time(r['single_median_ms'])}")
    else:
        print("  (not found)")
    print()

    # Summary tables
    if not results:
        print("No formats found to benchmark.")
        return

    # Table A: Bulk extraction
    results.sort(key=lambda x: x["bulk_median_ms"])
    fastest_bulk = results[0]["bulk_median_ms"]

    print("=" * 90)
    print(f"TABLE A — Bulk extraction (all features, finest LOD), "
          f"zoom={zoom}, brain {args.brain}")
    print("=" * 90)
    print(f"  {'Format':<25s} {'Disk':>10s} {'Features':>10s} "
          f"{'Vertices':>10s} {'Time':>10s} {'vs best':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        ratio = r["bulk_median_ms"] / fastest_bulk
        print(
            f"  {r['name']:<25s} "
            f"{_fmt_bytes(r['disk_bytes']):>10s} "
            f"{r['features']:>10,} "
            f"{r['vertices']:>10,} "
            f"{_fmt_time(r['bulk_median_ms']):>10s} "
            f"{ratio:>7.1f}x"
        )
    print()

    # Table B: Single-feature random access
    results.sort(key=lambda x: x["single_median_ms"])
    fastest_single = results[0]["single_median_ms"]

    print("=" * 65)
    print(f"TABLE B — Single-feature random access "
          f"(median of {args.n_samples} samples)")
    print("=" * 65)
    print(f"  {'Format':<25s} {'Median':>12s} {'vs best':>8s}")
    print(f"  {'-'*25} {'-'*12} {'-'*8}")
    for r in results:
        ratio = r["single_median_ms"] / fastest_single
        print(
            f"  {r['name']:<25s} "
            f"{_fmt_time(r['single_median_ms']):>12s} "
            f"{ratio:>7.1f}x"
        )
    print()

    print("Note: Feature PBF3/Neuroglancer bytes pre-loaded to RAM (I/O excluded).")
    print("      Parquet/Arrow bulk includes disk I/O; single-feature uses pre-loaded table.")


if __name__ == "__main__":
    main()
