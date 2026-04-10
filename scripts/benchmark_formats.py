#!/usr/bin/env python3
"""Benchmark .pbf3 vs OGC 3D Tiles — file size, decode latency, metadata query.

Usage::

    .venv/bin/python scripts/benchmark_formats.py
    .venv/bin/python scripts/benchmark_formats.py --scales 100 1000
    .venv/bin/python scripts/benchmark_formats.py --swc swcs/n120.CNG.swc
    .venv/bin/python scripts/benchmark_formats.py --max-zoom 4
    .venv/bin/python scripts/benchmark_formats.py --csv results.csv
    .venv/bin/python scripts/benchmark_formats.py --dataloader   # requires torch

Metrics:
    - File size (raw bytes + gzip-compressed)
    - Single-tile decode latency (protobuf vs pygltflib)
    - Metadata query: extract all properties from decoded features
    - ML DataLoader throughput (tiles/sec, optional torch)
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

# Force unbuffered stdout for progress output
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mudm_tools.polygen3d import generate_3d_collection
from mudm_tools.tiling3d.generator3d import TileGenerator3D
from mudm_tools.tiling3d.octree import OctreeConfig
from mudm_tools.tiling3d.reader3d import TileReader3D, decode_tile
from mudm_tools.tiling3d.reader_3dtiles import TileReader3DTiles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dir_size(path: Path) -> int:
    """Total bytes of all files under *path*."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _dir_size_gzipped(path: Path) -> int:
    """Total gzip-compressed size of all tile files under *path*."""
    total = 0
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        data = f.read_bytes()
        total += len(gzip.compress(data, compresslevel=6))
    return total


def _collect_tile_files(path: Path, suffix: str) -> list[Path]:
    """Collect all tile files with the given suffix."""
    return sorted(path.rglob(f"*{suffix}"))


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def _fmt_us(seconds: float) -> str:
    return f"{seconds * 1_000_000:.0f} us"


# ---------------------------------------------------------------------------
# Benchmark: File Size
# ---------------------------------------------------------------------------


def bench_file_size(pbf3_dir: Path, tiles3d_dir: Path) -> dict[str, Any]:
    """Measure raw and gzipped file sizes for both formats."""
    pbf3_raw = _dir_size(pbf3_dir)
    tiles3d_raw = _dir_size(tiles3d_dir)

    pbf3_gz = _dir_size_gzipped(pbf3_dir)
    tiles3d_gz = _dir_size_gzipped(tiles3d_dir)

    pbf3_count = len(_collect_tile_files(pbf3_dir, ".pbf3"))
    tiles3d_count = len(_collect_tile_files(tiles3d_dir, ".glb"))

    return {
        "pbf3_raw": pbf3_raw,
        "pbf3_gzip": pbf3_gz,
        "pbf3_tiles": pbf3_count,
        "3dtiles_raw": tiles3d_raw,
        "3dtiles_gzip": tiles3d_gz,
        "3dtiles_tiles": tiles3d_count,
    }


# ---------------------------------------------------------------------------
# Benchmark: Decode Latency
# ---------------------------------------------------------------------------


def bench_decode_latency(
    pbf3_dir: Path,
    tiles3d_dir: Path,
    n_iterations: int = 50,
    max_sample: int = 30,
) -> dict[str, Any]:
    """Measure per-tile decode latency for both formats.

    Samples up to *max_sample* tiles to keep runtime bounded.
    """
    import random as _rng

    import pygltflib

    pbf3_files = _collect_tile_files(pbf3_dir, ".pbf3")
    glb_files = _collect_tile_files(tiles3d_dir, ".glb")

    if not pbf3_files or not glb_files:
        return {"pbf3_decode_ms": 0, "3dtiles_decode_ms": 0}

    # Sample tiles to keep runtime bounded
    if len(pbf3_files) > max_sample:
        pbf3_files = _rng.sample(pbf3_files, max_sample)
    if len(glb_files) > max_sample:
        glb_files = _rng.sample(glb_files, max_sample)

    pbf3_bytes = [f.read_bytes() for f in pbf3_files]
    glb_bytes = [f.read_bytes() for f in glb_files]

    # Benchmark pbf3 decode
    pbf3_times: list[float] = []
    for _ in range(n_iterations):
        for data in pbf3_bytes:
            t0 = time.perf_counter()
            decode_tile(data)
            pbf3_times.append(time.perf_counter() - t0)

    # Benchmark glb decode
    glb_times: list[float] = []
    for _ in range(n_iterations):
        for data in glb_bytes:
            t0 = time.perf_counter()
            pygltflib.GLTF2.load_from_bytes(data)
            glb_times.append(time.perf_counter() - t0)

    return {
        "pbf3_decode_median_ms": statistics.median(pbf3_times) * 1000,
        "pbf3_decode_p95_ms": sorted(pbf3_times)[int(len(pbf3_times) * 0.95)] * 1000,
        "3dtiles_decode_median_ms": statistics.median(glb_times) * 1000,
        "3dtiles_decode_p95_ms": sorted(glb_times)[int(len(glb_times) * 0.95)] * 1000,
        "pbf3_sampled": len(pbf3_bytes),
        "3dtiles_sampled": len(glb_bytes),
    }


# ---------------------------------------------------------------------------
# Benchmark: Metadata Query
# ---------------------------------------------------------------------------


def bench_metadata_query(
    pbf3_dir: Path,
    tiles3d_dir: Path,
    n_iterations: int = 100,
    max_sample: int = 30,
) -> dict[str, Any]:
    """Measure time to extract all feature properties from decoded tiles."""
    import random as _rng

    import pygltflib

    pbf3_files = _collect_tile_files(pbf3_dir, ".pbf3")
    glb_files = _collect_tile_files(tiles3d_dir, ".glb")

    if not pbf3_files or not glb_files:
        return {"pbf3_meta_ms": 0, "3dtiles_meta_ms": 0}

    if len(pbf3_files) > max_sample:
        pbf3_files = _rng.sample(pbf3_files, max_sample)
    if len(glb_files) > max_sample:
        glb_files = _rng.sample(glb_files, max_sample)

    pbf3_bytes = [f.read_bytes() for f in pbf3_files]
    glb_bytes = [f.read_bytes() for f in glb_files]

    # pbf3: decode + extract tags from all features
    pbf3_times: list[float] = []
    pbf3_feature_count = 0
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        for data in pbf3_bytes:
            layers = decode_tile(data)
            for layer in layers:
                for feat in layer["features"]:
                    _ = feat["tags"]
                    pbf3_feature_count += 1
        pbf3_times.append(time.perf_counter() - t0)

    # 3dtiles: decode glb + extract extras/metadata
    glb_times: list[float] = []
    glb_feature_count = 0
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        for data in glb_bytes:
            gltf = pygltflib.GLTF2.load_from_bytes(data)
            for mesh in gltf.meshes:
                if mesh.extras:
                    _ = mesh.extras
                glb_feature_count += 1
        glb_times.append(time.perf_counter() - t0)

    return {
        "pbf3_meta_median_ms": statistics.median(pbf3_times) * 1000,
        "pbf3_features_per_iter": pbf3_feature_count // max(n_iterations, 1),
        "3dtiles_meta_median_ms": statistics.median(glb_times) * 1000,
        "3dtiles_features_per_iter": glb_feature_count // max(n_iterations, 1),
    }


# ---------------------------------------------------------------------------
# Benchmark: Memory Footprint
# ---------------------------------------------------------------------------


def bench_memory(pbf3_dir: Path, tiles3d_dir: Path) -> dict[str, Any]:
    """Measure peak memory during full-tileset decode."""
    import pygltflib

    pbf3_bytes = [f.read_bytes() for f in _collect_tile_files(pbf3_dir, ".pbf3")]
    glb_bytes = [f.read_bytes() for f in _collect_tile_files(tiles3d_dir, ".glb")]

    # pbf3
    tracemalloc.start()
    for data in pbf3_bytes:
        decode_tile(data)
    _, pbf3_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 3dtiles
    tracemalloc.start()
    for data in glb_bytes:
        pygltflib.GLTF2.load_from_bytes(data)
    _, glb_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "pbf3_peak_kb": pbf3_peak / 1024,
        "3dtiles_peak_kb": glb_peak / 1024,
    }


# ---------------------------------------------------------------------------
# Benchmark: ML DataLoader Throughput (optional torch)
# ---------------------------------------------------------------------------


def bench_dataloader(
    pbf3_dir: Path,
    tiles3d_dir: Path,
    n_epochs: int = 3,
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict[str, Any] | None:
    """Measure tiles/sec through a PyTorch DataLoader."""
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError:
        return None

    import pygltflib

    class Mvt3Dataset(Dataset):
        def __init__(self, tile_dir: Path):
            self._files = _collect_tile_files(tile_dir, ".pbf3")
            self._data = [f.read_bytes() for f in self._files]

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> torch.Tensor:
            layers = decode_tile(self._data[idx])
            # Use indexed mesh bytes directly when available (zero-copy)
            parts: list[torch.Tensor] = []
            for layer in layers:
                for feat in layer["features"]:
                    mesh_pos = feat.get("mesh_positions", b"")
                    if mesh_pos:
                        # Zero-copy: raw float32 LE bytes → tensor
                        parts.append(torch.frombuffer(
                            bytearray(mesh_pos), dtype=torch.float32,
                        ))
                    else:
                        coords: list[float] = []
                        for x, y in feat["xy"]:
                            coords.extend([float(x), float(y)])
                        for z in feat["z"]:
                            coords.append(float(z))
                        if coords:
                            parts.append(torch.tensor(coords, dtype=torch.float32))
            if not parts:
                return torch.zeros(1)
            return torch.cat(parts) if len(parts) > 1 else parts[0]

    class Glb3DDataset(Dataset):
        def __init__(self, tile_dir: Path):
            self._files = _collect_tile_files(tile_dir, ".glb")
            self._data = [f.read_bytes() for f in self._files]

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, idx: int) -> torch.Tensor:
            gltf = pygltflib.GLTF2.load_from_bytes(self._data[idx])
            # Count primitives as proxy for data volume
            n = sum(len(m.primitives) for m in gltf.meshes)
            return torch.tensor([float(n)], dtype=torch.float32)

    def _collate(batch: list[torch.Tensor]) -> list[torch.Tensor]:
        return batch  # variable-length tensors, no padding

    results: dict[str, Any] = {}

    for label, ds_cls, tile_dir in [
        ("pbf3", Mvt3Dataset, pbf3_dir),
        ("3dtiles", Glb3DDataset, tiles3d_dir),
    ]:
        ds = ds_cls(tile_dir)
        if len(ds) == 0:
            results[f"{label}_tiles_per_sec"] = 0
            continue

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=_collate,
        )

        total_tiles = 0
        t0 = time.perf_counter()
        for _ in range(n_epochs):
            for batch in loader:
                total_tiles += len(batch)
        elapsed = time.perf_counter() - t0
        results[f"{label}_tiles_per_sec"] = total_tiles / elapsed if elapsed > 0 else 0
        results[f"{label}_total_tiles"] = total_tiles
        results[f"{label}_elapsed_sec"] = elapsed

    return results


# ---------------------------------------------------------------------------
# Generate + Tile
# ---------------------------------------------------------------------------


def generate_and_tile(
    n_features: int,
    max_zoom: int,
    work_dir: Path,
    label: str,
) -> tuple[Path, Path]:
    """Generate synthetic data and tile in both formats. Returns (pbf3_dir, 3dtiles_dir)."""
    # Use roughly 70% TINs, 15% points, 15% lines
    n_tins = max(1, int(n_features * 0.7))
    n_points = max(1, int(n_features * 0.15))
    n_lines = n_features - n_tins - n_points

    collection = generate_3d_collection(
        n_tins=n_tins,
        n_points=n_points,
        n_lines=n_lines,
        bounds=(0, 0, 0, 100, 100, 100),
        triangles_per_tin=6,
        n_meta_keys=5,
        n_meta_variants=8,
        seed=42,
    )

    pbf3_dir = work_dir / f"{label}_pbf3"
    tiles3d_dir = work_dir / f"{label}_3dtiles"

    config = OctreeConfig(max_zoom=max_zoom)

    # pbf3
    gen_pbf3 = TileGenerator3D(config, output_format="pbf3")
    gen_pbf3.add_features(collection)
    gen_pbf3.generate(pbf3_dir)
    gen_pbf3.write_metadata(pbf3_dir)

    # 3dtiles
    gen_3dt = TileGenerator3D(OctreeConfig(max_zoom=max_zoom), output_format="3dtiles")
    gen_3dt.add_features(collection)
    gen_3dt.generate(tiles3d_dir)
    gen_3dt.write_metadata(tiles3d_dir)

    return pbf3_dir, tiles3d_dir


def generate_swc_tiles(
    swc_path: Path,
    max_zoom: int,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Load SWC file and tile in both formats."""
    from mudm_tools.swc import swc_to_feature_collection

    collection = swc_to_feature_collection(swc_path)

    pbf3_dir = work_dir / "swc_pbf3"
    tiles3d_dir = work_dir / "swc_3dtiles"

    config = OctreeConfig(max_zoom=max_zoom)

    gen_pbf3 = TileGenerator3D(config, output_format="pbf3")
    gen_pbf3.add_features(collection)
    gen_pbf3.generate(pbf3_dir)
    gen_pbf3.write_metadata(pbf3_dir)

    gen_3dt = TileGenerator3D(OctreeConfig(max_zoom=max_zoom), output_format="3dtiles")
    gen_3dt.add_features(collection)
    gen_3dt.generate(tiles3d_dir)
    gen_3dt.write_metadata(tiles3d_dir)

    return pbf3_dir, tiles3d_dir


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(label: str, results: dict[str, dict[str, Any]]) -> None:
    """Print formatted comparison table."""
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")

    if "size" in results:
        s = results["size"]
        print(f"\n  File Size")
        print(f"  {'':30s} {'pbf3':>15s} {'3dtiles':>15s}")
        print(f"  {'─' * 62}")
        print(f"  {'Tile count':30s} {s['pbf3_tiles']:>15d} {s['3dtiles_tiles']:>15d}")
        print(f"  {'Raw total':30s} {_fmt_bytes(s['pbf3_raw']):>15s} {_fmt_bytes(s['3dtiles_raw']):>15s}")
        print(f"  {'Gzipped total':30s} {_fmt_bytes(s['pbf3_gzip']):>15s} {_fmt_bytes(s['3dtiles_gzip']):>15s}")
        if s["3dtiles_raw"] > 0:
            ratio = s["pbf3_raw"] / s["3dtiles_raw"]
            print(f"  {'Ratio (pbf3/3dtiles raw)':30s} {ratio:>15.2f}x")

    if "decode" in results:
        d = results["decode"]
        print(f"\n  Decode Latency (per tile)")
        print(f"  {'':30s} {'pbf3':>15s} {'3dtiles':>15s}")
        print(f"  {'─' * 62}")
        print(f"  {'Median':30s} {d['pbf3_decode_median_ms']:>13.3f}ms {d['3dtiles_decode_median_ms']:>13.3f}ms")
        print(f"  {'P95':30s} {d['pbf3_decode_p95_ms']:>13.3f}ms {d['3dtiles_decode_p95_ms']:>13.3f}ms")
        if d["3dtiles_decode_median_ms"] > 0:
            speedup = d["3dtiles_decode_median_ms"] / d["pbf3_decode_median_ms"]
            print(f"  {'Speedup (pbf3 vs 3dtiles)':30s} {speedup:>15.1f}x")

    if "meta" in results:
        m = results["meta"]
        print(f"\n  Metadata Query (all tiles)")
        print(f"  {'':30s} {'pbf3':>15s} {'3dtiles':>15s}")
        print(f"  {'─' * 62}")
        print(f"  {'Median total':30s} {m['pbf3_meta_median_ms']:>13.3f}ms {m['3dtiles_meta_median_ms']:>13.3f}ms")
        print(f"  {'Features/iter':30s} {m['pbf3_features_per_iter']:>15d} {m['3dtiles_features_per_iter']:>15d}")

    if "memory" in results:
        mem = results["memory"]
        print(f"\n  Peak Memory (full decode)")
        print(f"  {'':30s} {'pbf3':>15s} {'3dtiles':>15s}")
        print(f"  {'─' * 62}")
        print(f"  {'Peak':30s} {mem['pbf3_peak_kb']:>13.1f}KB {mem['3dtiles_peak_kb']:>13.1f}KB")

    if "dataloader" in results and results["dataloader"]:
        dl = results["dataloader"]
        print(f"\n  ML DataLoader Throughput")
        print(f"  {'':30s} {'pbf3':>15s} {'3dtiles':>15s}")
        print(f"  {'─' * 62}")
        pbf3_tps = dl.get("pbf3_tiles_per_sec", 0)
        dt_tps = dl.get("3dtiles_tiles_per_sec", 0)
        print(f"  {'Tiles/sec':30s} {pbf3_tps:>15.1f} {dt_tps:>15.1f}")

    print()


def results_to_csv_row(label: str, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten results into a single dict for CSV export."""
    row: dict[str, Any] = {"dataset": label}
    for section, data in results.items():
        if data is None:
            continue
        for k, v in data.items():
            row[f"{section}_{k}"] = v
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark .pbf3 vs OGC 3D Tiles formats",
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
        help="Max zoom level for tiling (default: 3)",
    )
    parser.add_argument(
        "--swc",
        type=Path,
        default=None,
        help="Path to SWC file for real-data benchmark",
    )
    parser.add_argument(
        "--decode-iters",
        type=int,
        default=50,
        help="Iterations for decode latency benchmark (default: 50)",
    )
    parser.add_argument(
        "--meta-iters",
        type=int,
        default=100,
        help="Iterations for metadata query benchmark (default: 100)",
    )
    parser.add_argument(
        "--dataloader",
        action="store_true",
        help="Run PyTorch DataLoader throughput benchmark (requires torch)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Write results to CSV file",
    )
    parser.add_argument(
        "--keep-tiles",
        action="store_true",
        help="Keep generated tile directories (default: clean up)",
    )
    args = parser.parse_args()

    csv_rows: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmp:
        work_dir = Path(tmp)

        # --- Synthetic benchmarks ---
        for n in args.scales:
            label = f"synthetic_{n}"
            print(f"\n>>> Generating {n} features (zoom 0-{args.max_zoom})...")
            t0 = time.perf_counter()
            pbf3_dir, tiles3d_dir = generate_and_tile(
                n, args.max_zoom, work_dir, label,
            )
            gen_time = time.perf_counter() - t0
            print(f"    Generated in {gen_time:.1f}s")

            results: dict[str, dict[str, Any] | None] = {}
            print("    Measuring file size...")
            results["size"] = bench_file_size(pbf3_dir, tiles3d_dir)
            print("    Measuring decode latency...")
            results["decode"] = bench_decode_latency(
                pbf3_dir, tiles3d_dir, args.decode_iters,
            )
            print("    Measuring metadata query...")
            results["meta"] = bench_metadata_query(
                pbf3_dir, tiles3d_dir, args.meta_iters,
            )
            print("    Measuring memory...")
            results["memory"] = bench_memory(pbf3_dir, tiles3d_dir)

            if args.dataloader:
                print("    Measuring DataLoader throughput...")
                results["dataloader"] = bench_dataloader(pbf3_dir, tiles3d_dir)
            else:
                results["dataloader"] = None

            print_report(label, results)
            csv_rows.append(results_to_csv_row(label, results))

            if not args.keep_tiles:
                shutil.rmtree(pbf3_dir, ignore_errors=True)
                shutil.rmtree(tiles3d_dir, ignore_errors=True)

        # --- SWC benchmark ---
        if args.swc and args.swc.exists():
            label = f"swc_{args.swc.stem}"
            print(f"\n>>> Loading SWC: {args.swc} (zoom 0-{args.max_zoom})...")
            t0 = time.perf_counter()
            pbf3_dir, tiles3d_dir = generate_swc_tiles(
                args.swc, args.max_zoom, work_dir,
            )
            gen_time = time.perf_counter() - t0
            print(f"    Generated in {gen_time:.1f}s")

            results = {}
            print("    Measuring file size...")
            results["size"] = bench_file_size(pbf3_dir, tiles3d_dir)
            print("    Measuring decode latency...")
            results["decode"] = bench_decode_latency(
                pbf3_dir, tiles3d_dir, args.decode_iters,
            )
            print("    Measuring metadata query...")
            results["meta"] = bench_metadata_query(
                pbf3_dir, tiles3d_dir, args.meta_iters,
            )
            print("    Measuring memory...")
            results["memory"] = bench_memory(pbf3_dir, tiles3d_dir)

            if args.dataloader:
                print("    Measuring DataLoader throughput...")
                results["dataloader"] = bench_dataloader(pbf3_dir, tiles3d_dir)
            else:
                results["dataloader"] = None

            print_report(label, results)
            csv_rows.append(results_to_csv_row(label, results))

        # --- CSV export ---
        if args.csv and csv_rows:
            import csv

            all_keys = list(dict.fromkeys(
                k for row in csv_rows for k in row.keys()
            ))
            with open(args.csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"Results written to {args.csv}")

    print("Done.")


if __name__ == "__main__":
    main()
