#!/usr/bin/env python3
"""Benchmark tile generation speed — Python vs Rust backend.

Usage::

    .venv/bin/python scripts/benchmark_generation.py
    .venv/bin/python scripts/benchmark_generation.py --scales 100 500 1000
    .venv/bin/python scripts/benchmark_generation.py --swc swcs/n120.CNG.swc
    .venv/bin/python scripts/benchmark_generation.py --profile
    .venv/bin/python scripts/benchmark_generation.py --max-zoom 4

Generates synthetic 3D features at configurable scales and measures
tile generation time. Reports median latency, tiles/sec, and backend.
"""

from __future__ import annotations

import argparse
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mudm_tools.tiling3d import RUST_AVAILABLE, TileGenerator3D, OctreeConfig


def _generate_collection(
    n_features: int,
    triangles: int = 6,
    swc_path: str | None = None,
):
    """Generate a MuDMFeatureCollection for benchmarking."""
    if swc_path:
        from mudm_tools.swc import swc_to_feature_collection
        return swc_to_feature_collection(swc_path)

    from mudm_tools.polygen3d import generate_3d_collection
    return generate_3d_collection(
        n_tins=n_features,
        n_points=0,
        n_lines=0,
        bounds=(0, 0, 0, 100, 100, 100),
        triangles_per_tin=triangles,
    )


def _run_generation(
    collection, max_zoom: int = 3, warmup: int = 1, runs: int = 3,
    workers: int | None = None,
):
    """Run tile generation and return timing results."""
    tmpdir = tempfile.mkdtemp(prefix="bench_gen_")

    try:
        # Warmup
        for i in range(warmup):
            out = Path(tmpdir) / f"warmup_{i}"
            gen = TileGenerator3D(OctreeConfig(max_zoom=max_zoom), workers=workers)
            gen.add_features(collection)
            gen.generate(out)

        times = []
        tile_counts = []
        for i in range(runs):
            out = Path(tmpdir) / f"run_{i}"
            gen = TileGenerator3D(OctreeConfig(max_zoom=max_zoom), workers=workers)
            gen.add_features(collection)

            t0 = time.perf_counter()
            n_tiles = gen.generate(out)
            t1 = time.perf_counter()

            times.append(t1 - t0)
            tile_counts.append(n_tiles)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return {
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "tiles": tile_counts[0],
        "tiles_per_sec": tile_counts[0] / statistics.median(times),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark 3D tile generation")
    parser.add_argument(
        "--scales", nargs="+", type=int, default=[100, 500],
        help="Number of features to generate (default: 100 500)",
    )
    parser.add_argument(
        "--triangles", type=int, default=6,
        help="Triangles per TIN feature (default: 6)",
    )
    parser.add_argument("--swc", type=str, help="Path to SWC file (overrides --scales)")
    parser.add_argument("--max-zoom", type=int, default=3, help="Max zoom level (default: 3)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs (default: 3)")
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Worker processes (default: None=auto, 1=serial)",
    )
    parser.add_argument("--profile", action="store_true", help="Run cProfile on one generation")
    args = parser.parse_args()

    backend = "Rust" if RUST_AVAILABLE else "Python"
    workers = args.workers
    effective = workers if workers and workers >= 1 else (os.cpu_count() or 1)
    print(f"Backend: {backend}")
    print(f"Workers: {effective} ({'auto' if workers is None else 'explicit'})")
    print(f"Max zoom: {args.max_zoom}")
    print(f"Triangles/TIN: {args.triangles}")
    print(f"Runs per scale: {args.runs}")
    print()

    if args.profile:
        import cProfile
        import pstats
        scale = args.scales[0] if not args.swc else 0
        collection = _generate_collection(scale, triangles=args.triangles, swc_path=args.swc)
        tmpdir = tempfile.mkdtemp(prefix="bench_prof_")
        gen = TileGenerator3D(OctreeConfig(max_zoom=args.max_zoom), workers=workers)
        gen.add_features(collection)
        n_feat = len(collection.features)
        total_tris = sum(
            len(f.geometry.coordinates) if f.geometry else 0
            for f in collection.features
        )
        print(f"Profiling generation ({n_feat} features, ~{total_tris} triangles)...")
        prof = cProfile.Profile()
        prof.enable()
        gen.generate(tmpdir)
        prof.disable()
        stats = pstats.Stats(prof)
        stats.sort_stats("cumulative")
        stats.print_stats(40)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    if args.swc:
        collection = _generate_collection(0, args.swc)
        n = len(collection.features)
        print(f"SWC: {args.swc} ({n} features)")
        result = _run_generation(collection, args.max_zoom, runs=args.runs, workers=workers)
        print(
            f"  {result['median_s']:.3f}s median | "
            f"{result['tiles']} tiles | "
            f"{result['tiles_per_sec']:.0f} tiles/s"
        )
        return

    print(f"{'Features':>10}  {'Median (s)':>10}  {'Tiles':>8}  {'Tiles/s':>10}")
    print("-" * 50)

    for scale in args.scales:
        collection = _generate_collection(scale, triangles=args.triangles)
        result = _run_generation(collection, args.max_zoom, runs=args.runs, workers=workers)
        print(
            f"{scale:>10}  {result['median_s']:>10.3f}  "
            f"{result['tiles']:>8}  {result['tiles_per_sec']:>10.0f}"
        )


if __name__ == "__main__":
    main()
