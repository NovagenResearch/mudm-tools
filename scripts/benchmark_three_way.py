#!/usr/bin/env python3
"""Three-way benchmark: PBF3 vs OGC 3D Tiles (GLB) vs Neuroglancer precomputed mesh.

Usage::

    uv run python scripts/benchmark_three_way.py
    uv run python scripts/benchmark_three_way.py --data-dir data/mouselight/2021-09-16/HortaObj/
    uv run python scripts/benchmark_three_way.py --scales 100 1000 5000
    uv run python scripts/benchmark_three_way.py --csv three_way_results.csv

Generates all three formats from the same StreamingTileGenerator run and
measures: raw size, gzipped size, decode latency, metadata query time,
generation time, peak memory, and DataLoader throughput.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import statistics
import struct
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds
from mudm_tools.tiling3d.reader3d import decode_tile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _dir_size_gzipped(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        total += len(gzip.compress(f.read_bytes(), compresslevel=6))
    return total


def _file_count(path: Path, suffix: str) -> int:
    return len(list(path.rglob(f"*{suffix}")))


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Format generation
# ---------------------------------------------------------------------------


def generate_all_formats_streaming(
    obj_paths: list[str],
    bounds: tuple[float, float, float, float, float, float],
    tags_list: list[dict],
    max_zoom: int,
    work_dir: Path,
) -> tuple[Path, Path, Path, dict[str, float]]:
    """Generate PBF3, 3D Tiles, and Neuroglancer from same streaming pipeline.

    Returns (pbf3_dir, tiles3d_dir, ng_dir, timing_dict).
    """
    pbf3_dir = work_dir / "pbf3"
    tiles3d_dir = work_dir / "3dtiles"
    ng_dir = work_dir / "neuroglancer"

    timings: dict[str, float] = {}

    # --- PBF3 ---
    t0 = time.perf_counter()
    gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    gen.add_obj_files(obj_paths, bounds, tags_list)
    gen.generate_pbf3(str(pbf3_dir))
    timings["pbf3_gen_sec"] = time.perf_counter() - t0

    # --- 3D Tiles (GLB) ---
    t0 = time.perf_counter()
    gen2 = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    gen2.add_obj_files(obj_paths, bounds, tags_list)
    gen2.generate_3dtiles(str(tiles3d_dir), bounds)
    timings["3dt_gen_sec"] = time.perf_counter() - t0

    # --- Neuroglancer ---
    t0 = time.perf_counter()
    gen3 = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    gen3.add_obj_files(obj_paths, bounds, tags_list)
    gen3.generate_neuroglancer_multilod(str(ng_dir), bounds)
    timings["ng_gen_sec"] = time.perf_counter() - t0

    return pbf3_dir, tiles3d_dir, ng_dir, timings


def _make_synthetic_features(n_features: int, seed: int = 42) -> list[dict]:
    """Generate synthetic TIN features in normalized [0,1]³ space."""
    rng = np.random.default_rng(seed)
    features = []
    for i in range(n_features):
        # Random triangle in [0,1]³
        cx, cy, cz = rng.random(3) * 0.8 + 0.1
        size = rng.random() * 0.1 + 0.01
        v0x, v0y = cx - size, cy - size
        v1x, v1y = cx + size, cy - size
        v2x, v2y = cx, cy + size
        z0 = cz
        z1 = cz + rng.random() * 0.05
        z2 = cz - rng.random() * 0.05

        feat = {
            "geometry": [v0x, v0y, v1x, v1y, v2x, v2y, v0x, v0y],
            "geometry_z": [z0, z1, z2, z0],
            "type": 5,  # TIN
            "ring_lengths": [4],
            "minX": min(v0x, v1x, v2x),
            "minY": min(v0y, v1y, v2y),
            "minZ": min(z0, z1, z2),
            "maxX": max(v0x, v1x, v2x),
            "maxY": max(v0y, v1y, v2y),
            "maxZ": max(z0, z1, z2),
            "tags": {"name": f"feature_{i}", "index": i},
        }
        features.append(feat)
    return features


def generate_all_formats_synthetic(
    n_features: int,
    max_zoom: int,
    work_dir: Path,
) -> tuple[Path, Path, Path, dict[str, float]]:
    """Generate all three formats from synthetic data using streaming generator."""
    pbf3_dir = work_dir / "pbf3"
    tiles3d_dir = work_dir / "3dtiles"
    ng_dir = work_dir / "neuroglancer"
    bounds = (0.0, 0.0, 0.0, 100.0, 100.0, 100.0)

    features = _make_synthetic_features(n_features)
    timings: dict[str, float] = {}

    # --- PBF3 ---
    t0 = time.perf_counter()
    gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    for feat in features:
        gen.add_feature(feat)
    gen.generate_pbf3(str(pbf3_dir))
    timings["pbf3_gen_sec"] = time.perf_counter() - t0

    # --- 3D Tiles (GLB) ---
    t0 = time.perf_counter()
    gen2 = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    for feat in features:
        gen2.add_feature(feat)
    gen2.generate_3dtiles(str(tiles3d_dir), bounds)
    timings["3dt_gen_sec"] = time.perf_counter() - t0

    # --- Neuroglancer ---
    t0 = time.perf_counter()
    try:
        gen3 = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
        for feat in features:
            gen3.add_feature(feat)
        gen3.generate_neuroglancer_multilod(str(ng_dir), bounds)
        timings["ng_gen_sec"] = time.perf_counter() - t0
    except Exception as e:
        print(f"    [WARN] Neuroglancer generation failed: {e}", file=sys.stderr)
        ng_dir.mkdir(parents=True, exist_ok=True)
        timings["ng_gen_sec"] = 0.0

    return pbf3_dir, tiles3d_dir, ng_dir, timings


# ---------------------------------------------------------------------------
# Decode benchmarks
# ---------------------------------------------------------------------------


def bench_decode_latency_pbf3(pbf3_dir: Path, n_iters: int = 50, max_sample: int = 30) -> dict[str, float]:
    """PBF3 decode latency."""
    import random as rng

    files = sorted(pbf3_dir.rglob("*.pbf3"))
    if not files:
        return {"median_ms": 0, "p95_ms": 0}
    if len(files) > max_sample:
        files = rng.sample(files, max_sample)

    data_list = [f.read_bytes() for f in files]
    times: list[float] = []
    for _ in range(n_iters):
        for data in data_list:
            t0 = time.perf_counter()
            decode_tile(data)
            times.append(time.perf_counter() - t0)

    sorted_t = sorted(times)
    return {
        "median_ms": statistics.median(sorted_t) * 1000,
        "p95_ms": sorted_t[int(len(sorted_t) * 0.95)] * 1000 if sorted_t else 0,
    }


def bench_decode_latency_glb(tiles3d_dir: Path, n_iters: int = 50, max_sample: int = 30) -> dict[str, float]:
    """GLB/3D Tiles decode latency."""
    import random as rng

    import pygltflib

    files = sorted(tiles3d_dir.rglob("*.glb"))
    if not files:
        return {"median_ms": 0, "p95_ms": 0}
    if len(files) > max_sample:
        files = rng.sample(files, max_sample)

    data_list = [f.read_bytes() for f in files]
    times: list[float] = []
    for _ in range(n_iters):
        for data in data_list:
            t0 = time.perf_counter()
            pygltflib.GLTF2.load_from_bytes(data)
            times.append(time.perf_counter() - t0)

    sorted_t = sorted(times)
    return {
        "median_ms": statistics.median(sorted_t) * 1000,
        "p95_ms": sorted_t[int(len(sorted_t) * 0.95)] * 1000 if sorted_t else 0,
    }


def bench_decode_latency_ng(ng_dir: Path, n_iters: int = 50, max_sample: int = 30) -> dict[str, float]:
    """Neuroglancer mesh decode latency."""
    import random as rng

    # Find segment binary files (numeric names, no extension)
    files = [
        f for f in sorted(ng_dir.iterdir())
        if f.is_file() and f.name.isdigit()
    ]
    if not files:
        return {"median_ms": 0, "p95_ms": 0}
    if len(files) > max_sample:
        files = rng.sample(files, max_sample)

    data_list = [f.read_bytes() for f in files]
    times: list[float] = []
    for _ in range(n_iters):
        for data in data_list:
            t0 = time.perf_counter()
            # Decode: uint32 num_verts + float32[N*3] + uint32[M]
            num_verts = struct.unpack_from("<I", data, 0)[0]
            n_floats = num_verts * 3
            needed = 4 + n_floats * 4
            if needed > len(data):
                continue  # skip malformed/empty segment
            np.frombuffer(data, dtype=np.float32, count=n_floats, offset=4)
            remaining = (len(data) - needed) // 4
            if remaining > 0:
                np.frombuffer(data, dtype=np.uint32, count=remaining, offset=needed)
            times.append(time.perf_counter() - t0)

    sorted_t = sorted(times)
    if not sorted_t:
        return {"median_ms": 0, "p95_ms": 0}
    return {
        "median_ms": statistics.median(sorted_t) * 1000,
        "p95_ms": sorted_t[int(len(sorted_t) * 0.95)] * 1000 if sorted_t else 0,
    }


def bench_metadata_query_ng(ng_dir: Path, n_iters: int = 100) -> dict[str, float]:
    """Neuroglancer segment_properties query time."""
    sp_path = ng_dir / "segment_properties" / "info"
    if not sp_path.exists():
        return {"median_ms": 0}

    data = sp_path.read_text()
    times: list[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        props = json.loads(data)
        # Simulate querying all properties
        if "inline" in props:
            _ = props["inline"].get("ids", [])
            for p in props["inline"].get("properties", []):
                _ = p.get("values", [])
        times.append(time.perf_counter() - t0)

    return {"median_ms": statistics.median(times) * 1000}


def bench_memory_ng(ng_dir: Path) -> dict[str, float]:
    """Peak memory during full Neuroglancer decode."""
    files = [
        f for f in sorted(ng_dir.iterdir())
        if f.is_file() and f.name.isdigit()
    ]
    data_list = [f.read_bytes() for f in files]

    tracemalloc.start()
    for data in data_list:
        num_verts = struct.unpack_from("<I", data, 0)[0]
        n_floats = num_verts * 3
        np.frombuffer(data, dtype=np.float32, count=n_floats, offset=4)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {"peak_kb": peak / 1024}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_three_way_report(label: str, r: dict[str, Any]) -> None:
    """Print formatted three-column comparison table."""
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")

    print(f"\n  {'Metric':30s} {'PBF3':>15s} {'3D Tiles':>15s} {'Neuroglancer':>15s}")
    print(f"  {'─' * 75}")

    # Size
    if "pbf3_raw" in r:
        print(f"  {'Raw size':30s} {_fmt_bytes(r['pbf3_raw']):>15s} {_fmt_bytes(r['3dt_raw']):>15s} {_fmt_bytes(r['ng_raw']):>15s}")
        print(f"  {'Gzipped size':30s} {_fmt_bytes(r['pbf3_gz']):>15s} {_fmt_bytes(r['3dt_gz']):>15s} {_fmt_bytes(r['ng_gz']):>15s}")
        print(f"  {'File/segment count':30s} {r['pbf3_count']:>15d} {r['3dt_count']:>15d} {r['ng_count']:>15d}")

    # Generation time
    if "pbf3_gen_sec" in r:
        print(f"  {'Generation time':30s} {r['pbf3_gen_sec']:>14.2f}s {r['3dt_gen_sec']:>14.2f}s {r['ng_gen_sec']:>14.2f}s")

    # Decode latency
    if "pbf3_dec_median" in r:
        print(f"  {'Decode median':30s} {r['pbf3_dec_median']:>13.3f}ms {r['3dt_dec_median']:>13.3f}ms {r['ng_dec_median']:>13.3f}ms")
        print(f"  {'Decode P95':30s} {r['pbf3_dec_p95']:>13.3f}ms {r['3dt_dec_p95']:>13.3f}ms {r['ng_dec_p95']:>13.3f}ms")

    # Speedups
    if r.get("3dt_dec_median", 0) > 0 and r.get("pbf3_dec_median", 0) > 0:
        pbf3_vs_3dt = r["3dt_dec_median"] / r["pbf3_dec_median"]
        print(f"  {'PBF3 vs 3D Tiles speedup':30s} {pbf3_vs_3dt:>15.1f}x")
    if r.get("ng_dec_median", 0) > 0 and r.get("pbf3_dec_median", 0) > 0:
        pbf3_vs_ng = r["pbf3_dec_median"] / r["ng_dec_median"]
        ng_vs_pbf3 = r["ng_dec_median"] / r["pbf3_dec_median"]
        if pbf3_vs_ng > 1:
            print(f"  {'Neuroglancer vs PBF3':30s} {pbf3_vs_ng:>14.1f}x faster")
        else:
            print(f"  {'PBF3 vs Neuroglancer':30s} {1/pbf3_vs_ng:>14.1f}x faster")

    # Metadata
    if "ng_meta_median" in r:
        print(f"  {'Metadata query':30s} {'N/A':>15s} {'N/A':>15s} {r['ng_meta_median']:>13.3f}ms")

    # Memory
    if "ng_mem_peak" in r:
        print(f"  {'Peak memory':30s} {'':>15s} {'':>15s} {r['ng_mem_peak']:>13.1f}KB")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-way benchmark: PBF3 vs 3D Tiles vs Neuroglancer",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing .obj files (MouseLight HortaObj)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Max OBJ files to load (0=all)",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        default=[100, 1000],
        help="Feature counts for synthetic benchmarks (default: 100 1000)",
    )
    parser.add_argument(
        "--max-zoom",
        type=int,
        default=3,
        help="Max zoom level (default: 3)",
    )
    parser.add_argument(
        "--decode-iters",
        type=int,
        default=50,
        help="Decode latency iterations (default: 50)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write results to CSV file",
    )
    args = parser.parse_args()

    csv_rows: list[dict[str, Any]] = []

    # --- Real OBJ dataset ---
    if args.data_dir and args.data_dir.exists():
        obj_files = sorted(args.data_dir.glob("*.obj"))
        if args.max_files > 0:
            obj_files = obj_files[:args.max_files]

        if obj_files:
            label = f"obj_{args.data_dir.name}_{len(obj_files)}"
            print(f"\n>>> {label}: Scanning {len(obj_files)} OBJ files...")
            paths = [str(f) for f in obj_files]
            bounds = scan_obj_bounds(paths)
            tags_list = [{"filename": f.stem, "index": i} for i, f in enumerate(obj_files)]

            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                print("    Generating all 3 formats...")
                pbf3_dir, t3d_dir, ng_dir, timings = generate_all_formats_streaming(
                    paths, bounds, tags_list, args.max_zoom, work,
                )

                r: dict[str, Any] = {}
                r.update(timings)

                print("    Measuring sizes...")
                r["pbf3_raw"] = _dir_size(pbf3_dir)
                r["pbf3_gz"] = _dir_size_gzipped(pbf3_dir)
                r["pbf3_count"] = _file_count(pbf3_dir, ".pbf3")
                r["3dt_raw"] = _dir_size(t3d_dir)
                r["3dt_gz"] = _dir_size_gzipped(t3d_dir)
                r["3dt_count"] = _file_count(t3d_dir, ".glb")
                r["ng_raw"] = _dir_size(ng_dir)
                r["ng_gz"] = _dir_size_gzipped(ng_dir)
                ng_segs = [f for f in ng_dir.iterdir() if f.is_file() and f.name.isdigit()]
                r["ng_count"] = len(ng_segs)

                print("    Measuring decode latency...")
                pbf3_dec = bench_decode_latency_pbf3(pbf3_dir, args.decode_iters)
                r["pbf3_dec_median"] = pbf3_dec["median_ms"]
                r["pbf3_dec_p95"] = pbf3_dec["p95_ms"]
                glb_dec = bench_decode_latency_glb(t3d_dir, args.decode_iters)
                r["3dt_dec_median"] = glb_dec["median_ms"]
                r["3dt_dec_p95"] = glb_dec["p95_ms"]
                ng_dec = bench_decode_latency_ng(ng_dir, args.decode_iters)
                r["ng_dec_median"] = ng_dec["median_ms"]
                r["ng_dec_p95"] = ng_dec["p95_ms"]

                print("    Measuring metadata query...")
                ng_meta = bench_metadata_query_ng(ng_dir)
                r["ng_meta_median"] = ng_meta["median_ms"]

                print("    Measuring memory...")
                ng_mem = bench_memory_ng(ng_dir)
                r["ng_mem_peak"] = ng_mem["peak_kb"]

                print_three_way_report(label, r)
                r["dataset"] = label
                csv_rows.append(r)

    # --- Synthetic benchmarks ---
    for n in args.scales:
        label = f"synthetic_{n}"
        print(f"\n>>> {label}: Generating {n} features (zoom 0-{args.max_zoom})...")

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            pbf3_dir, t3d_dir, ng_dir, timings = generate_all_formats_synthetic(
                n, args.max_zoom, work,
            )

            r = {}
            r.update(timings)

            print("    Measuring sizes...")
            r["pbf3_raw"] = _dir_size(pbf3_dir)
            r["pbf3_gz"] = _dir_size_gzipped(pbf3_dir)
            r["pbf3_count"] = _file_count(pbf3_dir, ".pbf3")
            r["3dt_raw"] = _dir_size(t3d_dir)
            r["3dt_gz"] = _dir_size_gzipped(t3d_dir)
            r["3dt_count"] = _file_count(t3d_dir, ".glb")
            r["ng_raw"] = _dir_size(ng_dir)
            r["ng_gz"] = _dir_size_gzipped(ng_dir)
            ng_segs = [f for f in ng_dir.iterdir() if f.is_file() and f.name.isdigit()]
            r["ng_count"] = len(ng_segs)

            print("    Measuring decode latency...")
            pbf3_dec = bench_decode_latency_pbf3(pbf3_dir, args.decode_iters)
            r["pbf3_dec_median"] = pbf3_dec["median_ms"]
            r["pbf3_dec_p95"] = pbf3_dec["p95_ms"]
            glb_dec = bench_decode_latency_glb(t3d_dir, args.decode_iters)
            r["3dt_dec_median"] = glb_dec["median_ms"]
            r["3dt_dec_p95"] = glb_dec["p95_ms"]
            ng_dec = bench_decode_latency_ng(ng_dir, args.decode_iters)
            r["ng_dec_median"] = ng_dec["median_ms"]
            r["ng_dec_p95"] = ng_dec["p95_ms"]

            print("    Measuring metadata query...")
            ng_meta = bench_metadata_query_ng(ng_dir)
            r["ng_meta_median"] = ng_meta["median_ms"]

            print_three_way_report(label, r)
            r["dataset"] = label
            csv_rows.append(r)

    # --- CSV export ---
    if args.csv and csv_rows:
        import csv

        all_keys = list(dict.fromkeys(k for row in csv_rows for k in row.keys()))
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Results written to {args.csv}")

    print("Done.")


if __name__ == "__main__":
    main()
