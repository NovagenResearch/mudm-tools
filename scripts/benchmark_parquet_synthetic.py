#!/usr/bin/env python3
"""Synthetic benchmark for Parquet generation modes.

Compares in-memory vs streaming vs partitioned Parquet output across
different mesh sizes (100, 1000, 10000 triangles per feature).

Measures: generation time, peak memory, file size, throughput.

Usage:
    uv run python scripts/benchmark_parquet_synthetic.py
"""

from __future__ import annotations

import random
import shutil
import statistics
import time
import tracemalloc
from pathlib import Path

from mudm_tools._rs import StreamingTileGenerator
from mudm_tools.tiling3d.parquet_writer import generate_parquet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRIANGLE_COUNTS = [100, 1_000, 10_000]
N_FEATURES = 10  # features per run
MIN_ZOOM = 0
MAX_ZOOM = 3
WORLD_BOUNDS = (0.0, 0.0, 0.0, 1000.0, 1000.0, 1000.0)
RUNS = 3  # repeat each config this many times
BATCH_SIZES = [100, 1_000, 10_000]  # streaming batch sizes to test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _make_synthetic_mesh(n_triangles: int, seed: int = 0) -> dict:
    """Build a TIN feature with n_triangles spread across [0.05, 0.95]³."""
    rng = random.Random(seed)
    xy = []
    z = []
    ring_lengths = []
    for _ in range(n_triangles):
        cx = rng.uniform(0.1, 0.9)
        cy = rng.uniform(0.1, 0.9)
        cz = rng.uniform(0.1, 0.9)
        d = 0.02
        xy.extend([cx - d, cy - d, cx + d, cy - d, cx, cy + d])
        z.extend([cz - d, cz + d, cz])
        ring_lengths.append(3)
    n = len(z)
    return {
        "geometry": xy,
        "geometry_z": z,
        "ring_lengths": ring_lengths,
        "type": 5,  # TIN
        "tags": {"n_tri": n_triangles, "seed": seed},
        "minX": min(xy[i * 2] for i in range(n)),
        "minY": min(xy[i * 2 + 1] for i in range(n)),
        "minZ": min(z),
        "maxX": max(xy[i * 2] for i in range(n)),
        "maxY": max(xy[i * 2 + 1] for i in range(n)),
        "maxZ": max(z),
    }


def _build_generator(n_triangles: int) -> StreamingTileGenerator:
    """Create generator and add N_FEATURES synthetic meshes."""
    gen = StreamingTileGenerator(
        min_zoom=MIN_ZOOM, max_zoom=MAX_ZOOM, base_cells=10,
    )
    for i in range(N_FEATURES):
        feat = _make_synthetic_mesh(n_triangles, seed=i)
        gen.add_feature(feat)
    return gen


def _dir_size(p: Path) -> int:
    """Recursive directory size in bytes."""
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_inmemory(n_triangles: int, out_dir: Path) -> dict:
    """Benchmark in-memory Parquet generation (original path)."""
    times = []
    peaks = []
    sizes = []
    row_counts = []

    for run in range(RUNS):
        gen = _build_generator(n_triangles)
        out = out_dir / f"inmem_{n_triangles}_{run}.parquet"

        tracemalloc.start()
        t0 = time.perf_counter()
        # Force in-memory path by calling _collect_parquet_data directly
        # We use a very large batch_size to effectively disable batching
        # but since hasattr check will route to streaming, we need to
        # temporarily use the generate_parquet with the in-memory approach.
        # Actually, since streaming is always available now, the in-memory
        # path is only used when _init_parquet_stream doesn't exist.
        # For fair comparison, we'll just use generate_parquet with a huge
        # batch_size (effectively in-memory behavior for single-file streaming).
        n_rows = generate_parquet(gen, out, WORLD_BOUNDS, batch_size=1_000_000)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        peaks.append(peak)
        sizes.append(out.stat().st_size)
        row_counts.append(n_rows)

    return {
        "mode": "single (large batch)",
        "n_triangles": n_triangles,
        "n_features": N_FEATURES,
        "rows": row_counts[0],
        "time_median": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "peak_mem_mb": max(peaks) / (1024 * 1024),
        "file_size": sizes[0],
    }


def bench_streaming(n_triangles: int, batch_size: int, out_dir: Path) -> dict:
    """Benchmark streaming single-file Parquet generation."""
    times = []
    peaks = []
    sizes = []
    row_counts = []

    for run in range(RUNS):
        gen = _build_generator(n_triangles)
        out = out_dir / f"stream_{n_triangles}_bs{batch_size}_{run}.parquet"

        tracemalloc.start()
        t0 = time.perf_counter()
        n_rows = generate_parquet(
            gen, out, WORLD_BOUNDS, batch_size=batch_size,
        )
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        peaks.append(peak)
        sizes.append(out.stat().st_size)
        row_counts.append(n_rows)

    return {
        "mode": f"single (batch={batch_size})",
        "n_triangles": n_triangles,
        "n_features": N_FEATURES,
        "rows": row_counts[0],
        "time_median": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "peak_mem_mb": max(peaks) / (1024 * 1024),
        "file_size": sizes[0],
    }


def bench_partitioned(n_triangles: int, batch_size: int, out_dir: Path) -> dict:
    """Benchmark partitioned streaming Parquet generation."""
    times = []
    peaks = []
    sizes = []
    row_counts = []

    for run in range(RUNS):
        gen = _build_generator(n_triangles)
        part_dir = out_dir / f"part_{n_triangles}_bs{batch_size}_{run}"

        tracemalloc.start()
        t0 = time.perf_counter()
        n_rows = generate_parquet(
            gen, part_dir, WORLD_BOUNDS,
            batch_size=batch_size, partitioned=True,
        )
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(elapsed)
        peaks.append(peak)
        sizes.append(_dir_size(part_dir))
        row_counts.append(n_rows)

    return {
        "mode": f"partitioned (batch={batch_size})",
        "n_triangles": n_triangles,
        "n_features": N_FEATURES,
        "rows": row_counts[0],
        "time_median": statistics.median(times),
        "time_min": min(times),
        "time_max": max(times),
        "peak_mem_mb": max(peaks) / (1024 * 1024),
        "file_size": sizes[0],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("/tmp/microjson_parquet_bench")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    results: list[dict] = []

    print("=" * 78)
    print("Synthetic Parquet Benchmark")
    print(f"  {N_FEATURES} features × {TRIANGLE_COUNTS} triangles/feature")
    print(f"  Zoom {MIN_ZOOM}–{MAX_ZOOM}, {RUNS} runs each")
    print("=" * 78)

    for n_tri in TRIANGLE_COUNTS:
        raw_verts = n_tri * 3 * N_FEATURES  # total vertices across features
        raw_bytes = raw_verts * 3 * 4  # 3 floats × 4 bytes each
        print(f"\n{'─' * 78}")
        print(f"  {n_tri:,} triangles/feature ({N_FEATURES} features, "
              f"~{_fmt_bytes(raw_bytes)} raw vertex data)")
        print(f"{'─' * 78}")

        # Large-batch single file (baseline)
        r = bench_inmemory(n_tri, out_dir)
        results.append(r)
        _print_result(r)

        # Streaming with different batch sizes
        for bs in BATCH_SIZES:
            r = bench_streaming(n_tri, bs, out_dir)
            results.append(r)
            _print_result(r)

        # Partitioned with medium batch
        for bs in BATCH_SIZES:
            r = bench_partitioned(n_tri, bs, out_dir)
            results.append(r)
            _print_result(r)

    # Summary table
    print(f"\n{'=' * 78}")
    print("SUMMARY TABLE")
    print(f"{'=' * 78}")
    print(f"{'Mode':<30} {'Tri/feat':>9} {'Rows':>7} "
          f"{'Time':>9} {'Peak MB':>9} {'Size':>10} {'Rows/s':>10}")
    print("─" * 88)
    for r in results:
        throughput = r["rows"] / r["time_median"] if r["time_median"] > 0 else 0
        print(
            f"{r['mode']:<30} {r['n_triangles']:>9,} {r['rows']:>7,} "
            f"{_fmt_time(r['time_median']):>9} {r['peak_mem_mb']:>9.1f} "
            f"{_fmt_bytes(r['file_size']):>10} {throughput:>10,.0f}"
        )

    # Cleanup
    shutil.rmtree(out_dir)
    print(f"\nCleaned up {out_dir}")


def _print_result(r: dict):
    throughput = r["rows"] / r["time_median"] if r["time_median"] > 0 else 0
    print(f"  {r['mode']:<35} "
          f"{_fmt_time(r['time_median']):>8} "
          f"(min={_fmt_time(r['time_min'])}, max={_fmt_time(r['time_max'])})  "
          f"peak={r['peak_mem_mb']:.1f} MB  "
          f"size={_fmt_bytes(r['file_size'])}  "
          f"rows={r['rows']:,}  "
          f"{throughput:,.0f} rows/s")


if __name__ == "__main__":
    main()
