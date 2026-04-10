#!/usr/bin/env python3
"""Per-feature retrieval benchmark: tile-centric PBF3 vs feature-centric PBF3 vs Neuroglancer.

Measures the query "give me all geometry for feature X" across three access patterns.

Usage::

    uv run python scripts/benchmark_feature_retrieval.py
    uv run python scripts/benchmark_feature_retrieval.py --scales 100 1000
    uv run python scripts/benchmark_feature_retrieval.py --csv results.csv
    uv run python scripts/benchmark_feature_retrieval.py --data-dir data/mouselight/2021-09-16/HortaObj/

Access patterns:
  1. Tile-centric PBF3: Scan all max-zoom tiles, decode each, filter for feature X
  2. Feature-centric PBF3: Read features/{X}.pbf3, decode one file
  3. Neuroglancer: Read {X} binary, np.frombuffer (trivial decode)

Metrics: wall-clock retrieval time (median/P95), bytes read, decode time.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import struct
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds
from mudm_tools.tiling3d.reader3d import decode_tile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _fmt_time(secs: float) -> str:
    if secs < 0.001:
        return f"{secs * 1_000_000:.0f} us"
    if secs < 1.0:
        return f"{secs * 1000:.2f} ms"
    return f"{secs:.3f} s"


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_features(n_features: int, seed: int = 42) -> list[dict]:
    """Generate N synthetic TIN features in [0,1]³ space."""
    rng = np.random.default_rng(seed)
    features = []
    for i in range(n_features):
        # Random triangle in a small region
        cx = rng.uniform(0.05, 0.95)
        cy = rng.uniform(0.05, 0.95)
        cz = rng.uniform(0.05, 0.95)
        size = rng.uniform(0.01, 0.05)

        # 2-4 triangles per feature
        n_tris = rng.integers(2, 5)
        xy = []
        z = []
        ring_lengths = []

        for _ in range(n_tris):
            # Random triangle near center
            for v in range(3):
                x = cx + rng.uniform(-size, size)
                y = cy + rng.uniform(-size, size)
                zv = cz + rng.uniform(-size, size)
                xy.extend([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])
                z.append(max(0.0, min(1.0, zv)))
            # Closing vertex
            xy.extend([xy[-6], xy[-5]])
            z.append(z[-3])
            ring_lengths.append(4)

        n = len(z)
        xs = [xy[j * 2] for j in range(n)]
        ys = [xy[j * 2 + 1] for j in range(n)]

        features.append({
            "geometry": xy,
            "geometry_z": z,
            "ring_lengths": ring_lengths,
            "type": 5,  # TIN
            "tags": {"name": f"feature_{i}", "id": i, "volume": float(rng.uniform(10, 1000))},
            "minX": min(xs), "minY": min(ys), "minZ": min(z),
            "maxX": max(xs), "maxY": max(ys), "maxZ": max(z),
        })
    return features


def generate_from_obj_dir(data_dir: Path, max_features: int = 0) -> tuple[list[str], tuple]:
    """Load OBJ files from a directory and return (paths, bounds)."""
    obj_paths = sorted(str(p) for p in data_dir.rglob("*.obj"))
    if max_features > 0:
        obj_paths = obj_paths[:max_features]
    if not obj_paths:
        raise FileNotFoundError(f"No .obj files found in {data_dir}")
    bounds = scan_obj_bounds(obj_paths)
    return obj_paths, bounds


# ---------------------------------------------------------------------------
# Format generation
# ---------------------------------------------------------------------------

def generate_all_formats(
    features: list[dict],
    world_bounds: tuple[float, ...],
    output_base: Path,
    max_zoom: int = 2,
) -> dict[str, Path]:
    """Generate tile-centric PBF3, feature-centric PBF3, and Neuroglancer from same features."""
    paths = {}

    # Tile-centric PBF3
    pbf3_dir = output_base / "tile_pbf3"
    gen1 = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    for feat in features:
        gen1.add_feature(feat)
    gen1.generate_pbf3(str(pbf3_dir))
    paths["tile_pbf3"] = pbf3_dir

    # Feature-centric PBF3
    feat_dir = output_base / "feature_pbf3"
    gen2 = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    for feat in features:
        gen2.add_feature(feat)
    gen2.generate_feature_pbf3(str(feat_dir), world_bounds)
    paths["feature_pbf3"] = feat_dir

    # Neuroglancer
    ng_dir = output_base / "neuroglancer"
    gen3 = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
    for feat in features:
        gen3.add_feature(feat)
    gen3.generate_neuroglancer_multilod(str(ng_dir), world_bounds)
    paths["neuroglancer"] = ng_dir

    return paths


def generate_all_formats_obj(
    obj_paths: list[str],
    world_bounds: tuple[float, ...],
    output_base: Path,
    max_zoom: int = 2,
) -> dict[str, Path]:
    """Generate all formats from OBJ files (parallel Rust pipeline)."""
    paths = {}
    tags_list = [{"name": Path(p).stem, "id": i} for i, p in enumerate(obj_paths)]

    for name, gen_func in [
        ("tile_pbf3", "generate_pbf3"),
        ("feature_pbf3", "generate_feature_pbf3"),
        ("neuroglancer", "generate_neuroglancer_multilod"),
    ]:
        out_dir = output_base / name
        gen = StreamingTileGenerator(min_zoom=0, max_zoom=max_zoom)
        gen.add_obj_files(obj_paths, world_bounds, tags_list)

        if name == "tile_pbf3":
            gen.generate_pbf3(str(out_dir))
        elif name == "feature_pbf3":
            gen.generate_feature_pbf3(str(out_dir), world_bounds)
        else:
            gen.generate_neuroglancer_multilod(str(out_dir), world_bounds)

        paths[name] = out_dir

    return paths


# ---------------------------------------------------------------------------
# Retrieval benchmarks
# ---------------------------------------------------------------------------

def retrieve_tile_centric(tile_dir: Path, feature_id: int, max_zoom: int) -> dict:
    """Retrieve feature X by scanning all max-zoom tiles."""
    t0 = time.perf_counter()
    bytes_read = 0
    decode_time = 0.0
    found_positions = []

    zoom_dir = tile_dir / str(max_zoom)
    if not zoom_dir.exists():
        return {"time": time.perf_counter() - t0, "bytes_read": 0, "decode_time": 0, "n_verts": 0}

    for tile_path in zoom_dir.rglob("*.pbf3"):
        data = tile_path.read_bytes()
        bytes_read += len(data)

        td0 = time.perf_counter()
        layers = decode_tile(data)
        decode_time += time.perf_counter() - td0

        for layer in layers:
            for feat in layer["features"]:
                if feat["tags"].get("id") == feature_id:
                    if feat["mesh_positions"]:
                        n_verts = len(feat["mesh_positions"]) // 12
                        found_positions.append(n_verts)

    total_time = time.perf_counter() - t0
    return {
        "time": total_time,
        "bytes_read": bytes_read,
        "decode_time": decode_time,
        "n_verts": sum(found_positions),
    }


def retrieve_feature_centric(feat_dir: Path, feature_id: int) -> dict:
    """Retrieve feature X by reading one .pbf3 file."""
    t0 = time.perf_counter()

    feat_path = feat_dir / f"{feature_id}.pbf3"
    if not feat_path.exists():
        return {"time": time.perf_counter() - t0, "bytes_read": 0, "decode_time": 0, "n_verts": 0}

    data = feat_path.read_bytes()
    bytes_read = len(data)

    td0 = time.perf_counter()
    layers = decode_tile(data)
    decode_time = time.perf_counter() - td0

    n_verts = 0
    for layer in layers:
        for feat in layer["features"]:
            if feat["mesh_positions"]:
                n_verts += len(feat["mesh_positions"]) // 12

    total_time = time.perf_counter() - t0
    return {
        "time": total_time,
        "bytes_read": bytes_read,
        "decode_time": decode_time,
        "n_verts": n_verts,
    }


def retrieve_neuroglancer(ng_dir: Path, feature_id: int) -> dict:
    """Retrieve feature X by reading one Neuroglancer binary file."""
    t0 = time.perf_counter()

    seg_path = ng_dir / str(feature_id)
    if not seg_path.exists():
        return {"time": time.perf_counter() - t0, "bytes_read": 0, "decode_time": 0, "n_verts": 0}

    data = seg_path.read_bytes()
    bytes_read = len(data)

    td0 = time.perf_counter()
    (n_verts,) = struct.unpack_from("<I", data, 0)
    # Just verify we can read the vertices (no full decode needed for benchmark)
    needed = 4 + n_verts * 3 * 4
    if needed <= len(data):
        _ = np.frombuffer(data, dtype=np.float32, offset=4, count=n_verts * 3)
    decode_time = time.perf_counter() - td0

    total_time = time.perf_counter() - t0
    return {
        "time": total_time,
        "bytes_read": bytes_read,
        "decode_time": decode_time,
        "n_verts": n_verts,
    }


def benchmark_scale(
    n_features: int,
    n_samples: int = 10,
    n_iterations: int = 5,
    max_zoom: int = 2,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Run benchmark at a given feature count scale."""
    world_bounds = (0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        if data_dir and data_dir.exists():
            obj_paths, world_bounds = generate_from_obj_dir(data_dir, max_features=n_features)
            actual_n = len(obj_paths)
            print(f"  Generating formats from {actual_n} OBJ files...")
            paths = generate_all_formats_obj(obj_paths, world_bounds, base, max_zoom=max_zoom)
        else:
            actual_n = n_features
            print(f"  Generating {n_features} synthetic features...")
            features = generate_synthetic_features(n_features)
            paths = generate_all_formats(features, world_bounds, base, max_zoom=max_zoom)

        # Count files and sizes
        sizes = {}
        for name, path in paths.items():
            sizes[name] = _dir_size(path)

        # Sample feature IDs to query
        feature_ids = list(range(actual_n))
        rng = random.Random(42)
        sample_ids = rng.sample(feature_ids, min(n_samples, len(feature_ids)))

        # Benchmark each access pattern
        results = {}
        for method_name, retrieve_fn, extra_args in [
            ("tile_pbf3", retrieve_tile_centric, (paths["tile_pbf3"], max_zoom)),
            ("feature_pbf3", retrieve_feature_centric, (paths["feature_pbf3"],)),
            ("neuroglancer", retrieve_neuroglancer, (paths["neuroglancer"],)),
        ]:
            times = []
            bytes_read_list = []
            decode_times = []
            n_verts_list = []

            for fid in sample_ids:
                for _ in range(n_iterations):
                    if method_name == "tile_pbf3":
                        r = retrieve_fn(extra_args[0], fid, extra_args[1])
                    else:
                        r = retrieve_fn(extra_args[0], fid)
                    times.append(r["time"])
                    bytes_read_list.append(r["bytes_read"])
                    decode_times.append(r["decode_time"])
                    n_verts_list.append(r["n_verts"])

            results[method_name] = {
                "median_time": statistics.median(times),
                "p95_time": sorted(times)[int(len(times) * 0.95)] if times else 0,
                "median_bytes_read": statistics.median(bytes_read_list),
                "median_decode_time": statistics.median(decode_times),
                "total_size": sizes.get(method_name, 0),
            }

    return {
        "n_features": actual_n,
        "n_samples": len(sample_ids),
        "n_iterations": n_iterations,
        "max_zoom": max_zoom,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_results(all_results: list[dict]) -> None:
    """Print benchmark results as a formatted table."""
    print()
    print("=" * 100)
    print("Per-Feature Retrieval Benchmark: Dual-Index Architecture")
    print("=" * 100)
    print()

    for result in all_results:
        n = result["n_features"]
        print(f"--- {n} features (zoom 0-{result['max_zoom']}, "
              f"{result['n_samples']} samples x {result['n_iterations']} iterations) ---")
        print()
        print(f"{'Access Pattern':<22} {'Median Time':>12} {'P95 Time':>12} "
              f"{'Bytes Read':>14} {'Decode Time':>12} {'Total Size':>12}")
        print("-" * 86)

        for method in ["tile_pbf3", "feature_pbf3", "neuroglancer"]:
            r = result["results"][method]
            labels = {
                "tile_pbf3": "Tile-centric PBF3",
                "feature_pbf3": "Feature-centric PBF3",
                "neuroglancer": "Neuroglancer",
            }
            print(f"{labels[method]:<22} "
                  f"{_fmt_time(r['median_time']):>12} "
                  f"{_fmt_time(r['p95_time']):>12} "
                  f"{_fmt_bytes(int(r['median_bytes_read'])):>14} "
                  f"{_fmt_time(r['median_decode_time']):>12} "
                  f"{_fmt_bytes(r['total_size']):>12}")
        print()

    # Speedup summary
    if all_results:
        print("Speedup Summary (feature-centric PBF3 vs tile-centric PBF3):")
        for result in all_results:
            n = result["n_features"]
            tile_time = result["results"]["tile_pbf3"]["median_time"]
            feat_time = result["results"]["feature_pbf3"]["median_time"]
            if feat_time > 0:
                speedup = tile_time / feat_time
                print(f"  {n} features: {speedup:.1f}x faster")
        print()


def write_csv(all_results: list[dict], csv_path: str) -> None:
    """Write results to CSV."""
    with open(csv_path, "w") as f:
        f.write("n_features,method,median_time_ms,p95_time_ms,median_bytes_read,"
                "median_decode_time_ms,total_size_bytes\n")
        for result in all_results:
            n = result["n_features"]
            for method, r in result["results"].items():
                f.write(f"{n},{method},"
                        f"{r['median_time'] * 1000:.3f},"
                        f"{r['p95_time'] * 1000:.3f},"
                        f"{int(r['median_bytes_read'])},"
                        f"{r['median_decode_time'] * 1000:.3f},"
                        f"{r['total_size']}\n")
    print(f"Results written to {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Per-feature retrieval benchmark")
    parser.add_argument("--scales", type=int, nargs="+", default=[50, 200],
                        help="Feature counts to benchmark (default: 50 200)")
    parser.add_argument("--max-zoom", type=int, default=2,
                        help="Max zoom level (default: 2)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of features to sample per scale (default: 10)")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Iterations per sample (default: 5)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Optional OBJ data directory")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional CSV output path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else None

    all_results = []
    for scale in args.scales:
        print(f"\nBenchmarking {scale} features...")
        result = benchmark_scale(
            n_features=scale,
            n_samples=args.samples,
            n_iterations=args.iterations,
            max_zoom=args.max_zoom,
            data_dir=data_dir,
        )
        all_results.append(result)

    print_results(all_results)

    if args.csv:
        write_csv(all_results, args.csv)


if __name__ == "__main__":
    main()
