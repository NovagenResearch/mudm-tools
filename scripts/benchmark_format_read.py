#!/usr/bin/env python3
"""Benchmark mesh read + recreate across all output formats.

Reads all meshes at a given zoom level from PBF3, Parquet, Arrow IPC, and
3D Tiles (GLB), reconstructs numpy arrays, and reports median timings.

Usage:
    uv run python scripts/benchmark_format_read.py
    uv run python scripts/benchmark_format_read.py --brain 2021-09-16 --zoom 2
    uv run python scripts/benchmark_format_read.py --iters 10
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))


def _fmt_time(ms: float) -> str:
    if ms < 1:
        return f"{ms * 1000:.0f}us"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms / 1000:.2f}s"


def _median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


def bench_pbf3(pbf3_dir: Path, *, n_warmup: int, n_iters: int):
    """Benchmark PBF3 protobuf decode (bytes pre-loaded to RAM)."""
    from mudm_tools.tiling3d.reader3d import decode_tile

    pbf3_files = sorted(pbf3_dir.rglob("*.pbf3"))
    if not pbf3_files:
        return None
    pbf3_bytes_list = [f.read_bytes() for f in pbf3_files]
    disk_size = sum(len(b) for b in pbf3_bytes_list)

    def run():
        meshes = []
        for data in pbf3_bytes_list:
            layers = decode_tile(data)
            for layer in layers:
                for feat in layer["features"]:
                    pos_bytes = feat.get("mesh_positions", b"")
                    idx_bytes = feat.get("mesh_indices", b"")
                    if pos_bytes:
                        positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
                        indices = np.frombuffer(idx_bytes, dtype=np.uint32) if idx_bytes else np.array([], dtype=np.uint32)
                        meshes.append((positions, indices))
        return meshes

    for _ in range(n_warmup):
        run()

    gc.collect()
    times = []
    result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = run()
        times.append((time.perf_counter() - t0) * 1000)

    n_meshes = len(result)
    total_verts = sum(m[0].shape[0] for m in result)
    total_idx = sum(len(m[1]) for m in result)
    del result
    gc.collect()

    return {
        "name": "PBF3 (protobuf)",
        "files": len(pbf3_files),
        "disk_bytes": disk_size,
        "meshes": n_meshes,
        "vertices": total_verts,
        "indices": total_idx,
        "median_ms": _median(times),
        "times_ms": times,
        "note": "bytes pre-loaded",
    }


def bench_parquet(pq_dir: Path, zoom: int, *, n_warmup: int, n_iters: int):
    """Benchmark Parquet ZSTD read (includes disk I/O + decompress)."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    zoom_dir = pq_dir / f"zoom={zoom}"
    if not zoom_dir.exists():
        return None
    pq_files = sorted(str(f) for f in zoom_dir.glob("*.parquet"))
    if not pq_files:
        return None
    disk_size = sum(Path(f).stat().st_size for f in pq_files)

    def run():
        partitioning = ds.HivePartitioning(pa.schema([("zoom", pa.int32())]))
        dataset = ds.dataset(
            pq_files, format="parquet",
            partitioning=partitioning,
            partition_base_dir=str(pq_dir),
        )
        table = dataset.to_table()
        meshes = []
        pos_col = table.column("positions")
        idx_col = table.column("indices")
        for i in range(table.num_rows):
            pos_bytes = pos_col[i].as_py()
            idx_bytes = idx_col[i].as_py()
            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            indices = np.frombuffer(idx_bytes, dtype=np.uint32)
            meshes.append((positions, indices))
        return meshes

    for _ in range(n_warmup):
        run()

    gc.collect()
    times = []
    result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = run()
        times.append((time.perf_counter() - t0) * 1000)

    n_meshes = len(result)
    total_verts = sum(m[0].shape[0] for m in result)
    total_idx = sum(len(m[1]) for m in result)
    del result
    gc.collect()

    return {
        "name": "Parquet (ZSTD)",
        "files": len(pq_files),
        "disk_bytes": disk_size,
        "meshes": n_meshes,
        "vertices": total_verts,
        "indices": total_idx,
        "median_ms": _median(times),
        "times_ms": times,
        "note": "disk I/O included",
    }


def bench_arrow(pq_dir: Path, zoom: int, *, n_warmup: int, n_iters: int):
    """Benchmark Arrow IPC read (includes disk I/O, zero-copy mmap)."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    zoom_dir = pq_dir / f"zoom={zoom}"
    if not zoom_dir.exists():
        return None
    arrow_files = sorted(str(f) for f in zoom_dir.glob("*.arrow"))
    if not arrow_files:
        return None
    disk_size = sum(Path(f).stat().st_size for f in arrow_files)

    def run():
        partitioning = ds.HivePartitioning(pa.schema([("zoom", pa.int32())]))
        dataset = ds.dataset(
            arrow_files, format="ipc",
            partitioning=partitioning,
            partition_base_dir=str(pq_dir),
        )
        table = dataset.to_table()
        meshes = []
        pos_col = table.column("positions")
        idx_col = table.column("indices")
        for i in range(table.num_rows):
            pos_bytes = pos_col[i].as_py()
            idx_bytes = idx_col[i].as_py()
            positions = np.frombuffer(pos_bytes, dtype=np.float32).reshape(-1, 3)
            indices = np.frombuffer(idx_bytes, dtype=np.uint32)
            meshes.append((positions, indices))
        return meshes

    for _ in range(n_warmup):
        run()

    gc.collect()
    times = []
    result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = run()
        times.append((time.perf_counter() - t0) * 1000)

    n_meshes = len(result)
    total_verts = sum(m[0].shape[0] for m in result)
    del result
    gc.collect()

    return {
        "name": "Arrow IPC (mmap)",
        "files": len(arrow_files),
        "disk_bytes": disk_size,
        "meshes": n_meshes,
        "vertices": total_verts,
        "indices": 0,
        "median_ms": _median(times),
        "times_ms": times,
        "note": "disk I/O included",
    }


def bench_3dtiles(tiles3d_dir: Path, *, n_warmup: int, n_iters: int):
    """Benchmark 3D Tiles GLB parse (bytes pre-loaded to RAM)."""
    try:
        import pygltflib
    except ImportError:
        print("  (pygltflib not installed, skipping 3D Tiles)")
        return None

    glb_files = sorted(tiles3d_dir.rglob("*.glb"))
    if not glb_files:
        return None
    glb_bytes_list = [f.read_bytes() for f in glb_files]
    disk_size = sum(len(b) for b in glb_bytes_list)

    def run():
        meshes = []
        for data in glb_bytes_list:
            gltf = pygltflib.GLTF2.load_from_bytes(data)
            blob = bytes(gltf.binary_blob())
            for mesh in gltf.meshes:
                for prim in mesh.primitives:
                    pos_acc = gltf.accessors[prim.attributes.POSITION]
                    pos_bv = gltf.bufferViews[pos_acc.bufferView]
                    pos_start = (pos_bv.byteOffset or 0) + (pos_acc.byteOffset or 0)
                    pos_end = pos_start + pos_acc.count * 12
                    positions = np.frombuffer(
                        blob[pos_start:pos_end], dtype=np.float32,
                    ).reshape(-1, 3)

                    if prim.indices is not None:
                        idx_acc = gltf.accessors[prim.indices]
                        idx_bv = gltf.bufferViews[idx_acc.bufferView]
                        idx_start = (idx_bv.byteOffset or 0) + (idx_acc.byteOffset or 0)
                        if idx_acc.componentType == 5123:  # uint16
                            idx_end = idx_start + idx_acc.count * 2
                            indices = np.frombuffer(
                                blob[idx_start:idx_end], dtype=np.uint16,
                            ).astype(np.uint32)
                        else:  # uint32
                            idx_end = idx_start + idx_acc.count * 4
                            indices = np.frombuffer(
                                blob[idx_start:idx_end], dtype=np.uint32,
                            )
                    else:
                        indices = np.array([], dtype=np.uint32)

                    meshes.append((positions, indices))
        return meshes

    for _ in range(n_warmup):
        run()

    gc.collect()
    times = []
    result = None
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = run()
        times.append((time.perf_counter() - t0) * 1000)

    n_meshes = len(result)
    total_verts = sum(m[0].shape[0] for m in result)
    total_idx = sum(len(m[1]) for m in result)
    del result
    gc.collect()

    return {
        "name": "3D Tiles (GLB)",
        "files": len(glb_files),
        "disk_bytes": disk_size,
        "meshes": n_meshes,
        "vertices": total_verts,
        "indices": total_idx,
        "median_ms": _median(times),
        "times_ms": times,
        "note": "bytes pre-loaded",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark mesh read+recreate across PBF3, Parquet, Arrow, 3D Tiles",
    )
    parser.add_argument(
        "--brain", type=str, default="2016-10-31",
        help="Brain date folder (default: 2016-10-31)",
    )
    parser.add_argument(
        "--zoom", type=int, default=3,
        help="Zoom level to benchmark (default: 3)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--iters", type=int, default=5,
        help="Benchmark iterations (default: 5)",
    )
    args = parser.parse_args()

    base = _ROOT / "data" / "mouselight" / "tiles" / args.brain
    if not base.exists():
        print(f"ERROR: {base} not found", file=sys.stderr)
        sys.exit(1)

    zoom = args.zoom
    print(f"Benchmark: read + recreate all meshes at zoom={zoom} for {args.brain}")
    print(f"Warmup: {args.warmup}, Iterations: {args.iters}")
    print()

    results = []

    # PBF3
    pbf3_dir = base / "pbf3" / str(zoom)
    print(f"--- PBF3 (protobuf) ---")
    r = bench_pbf3(pbf3_dir, n_warmup=args.warmup, n_iters=args.iters)
    if r:
        results.append(r)
        print(f"  {r['files']} tiles, {r['disk_bytes']/1e6:.1f} MB on disk")
        print(f"  Meshes: {r['meshes']:,}  Vertices: {r['vertices']:,}  Indices: {r['indices']:,}")
        print(f"  Median: {_fmt_time(r['median_ms'])}  ({r['note']})")
        print(f"  Runs: {[f'{t:.1f}' for t in r['times_ms']]} ms")
    else:
        print("  (not found)")
    print()

    # Parquet
    pq_dir = base / "parquet_partitioned"
    print(f"--- Parquet (ZSTD) ---")
    r = bench_parquet(pq_dir, zoom, n_warmup=args.warmup, n_iters=args.iters)
    if r:
        results.append(r)
        print(f"  {r['files']} file(s), {r['disk_bytes']/1e6:.1f} MB on disk")
        print(f"  Meshes: {r['meshes']:,}  Vertices: {r['vertices']:,}  Indices: {r['indices']:,}")
        print(f"  Median: {_fmt_time(r['median_ms'])}  ({r['note']})")
        print(f"  Runs: {[f'{t:.1f}' for t in r['times_ms']]} ms")
    else:
        print("  (not found)")
    print()

    # Arrow IPC
    print(f"--- Arrow IPC (mmap) ---")
    r = bench_arrow(pq_dir, zoom, n_warmup=args.warmup, n_iters=args.iters)
    if r:
        results.append(r)
        print(f"  {r['files']} file(s), {r['disk_bytes']/1e6:.1f} MB on disk")
        print(f"  Meshes: {r['meshes']:,}  Vertices: {r['vertices']:,}")
        print(f"  Median: {_fmt_time(r['median_ms'])}  ({r['note']})")
        print(f"  Runs: {[f'{t:.1f}' for t in r['times_ms']]} ms")
    else:
        print("  (not found)")
    print()

    # 3D Tiles
    tiles3d_dir = base / "3dtiles" / str(zoom)
    print(f"--- 3D Tiles (GLB) ---")
    r = bench_3dtiles(tiles3d_dir, n_warmup=args.warmup, n_iters=args.iters)
    if r:
        results.append(r)
        print(f"  {r['files']} tiles, {r['disk_bytes']/1e6:.1f} MB on disk")
        print(f"  Meshes: {r['meshes']:,}  Vertices: {r['vertices']:,}  Indices: {r['indices']:,}")
        print(f"  Median: {_fmt_time(r['median_ms'])}  ({r['note']})")
        print(f"  Runs: {[f'{t:.1f}' for t in r['times_ms']]} ms")
    else:
        print("  (not found)")
    print()

    # Summary
    if not results:
        print("No formats found to benchmark.")
        return

    results.sort(key=lambda r: r["median_ms"])
    fastest = results[0]["median_ms"]

    print("=" * 78)
    print(f"SUMMARY — zoom={zoom}, brain {args.brain}")
    print("=" * 78)
    print(f"  {'Format':<25s} {'Disk':>10s} {'Meshes':>8s} {'Vertices':>10s} {'Time':>10s} {'vs best':>8s}")
    print(f"  {'-'*25} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for r in results:
        ratio = r["median_ms"] / fastest
        print(
            f"  {r['name']:<25s} "
            f"{r['disk_bytes']/1e6:>8.1f}MB "
            f"{r['meshes']:>8,} "
            f"{r['vertices']:>10,} "
            f"{_fmt_time(r['median_ms']):>10s} "
            f"{ratio:>7.1f}x"
        )
    print()
    print("Note: PBF3/3D Tiles bytes pre-loaded to RAM (I/O excluded).")
    print("      Parquet/Arrow read from disk each iteration (I/O included).")


if __name__ == "__main__":
    main()
