#!/usr/bin/env python3
"""Unified multi-dataset benchmark runner.

Runs decode/memory/DataLoader benchmarks across all 5 benchmark datasets
and produces a comparison table (console, CSV, and LaTeX).

Assumes each dataset has already been downloaded, converted, and tiled.
Looks for tiles in: data/{dataset}/tiles/{pbf3,3dtiles}/

Usage::

    # Run all benchmarks:
    .venv/bin/python scripts/benchmark_all_datasets.py

    # With CSV and LaTeX export:
    .venv/bin/python scripts/benchmark_all_datasets.py \
      --csv results/multi_dataset.csv \
      --latex results/multi_dataset.tex

    # Include DataLoader benchmark:
    .venv/bin/python scripts/benchmark_all_datasets.py --dataloader

    # Specific datasets only:
    .venv/bin/python scripts/benchmark_all_datasets.py --datasets mouselight hemibrain
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

from mudm_tools.tiling3d import RUST_AVAILABLE
from mudm_tools.tiling3d.reader3d import decode_tile


# ---------------------------------------------------------------------------
# Dataset Registry
# ---------------------------------------------------------------------------

DATASETS = {
    "mouselight": {
        "label": "Allen CCF / MouseLight",
        "domain": "Mouse brain anatomy",
        "geometry": "TIN (mesh)",
        "tiles_dir": _ROOT / "data" / "mouselight" / "tiles",
        "metadata": _ROOT / "data" / "mouselight" / "tiles" / "3dtiles" / "features.json",
    },
    "hemibrain": {
        "label": "Hemibrain v1.2.1",
        "domain": "Fly brain connectome",
        "geometry": "TIN (mesh)",
        "tiles_dir": _ROOT / "data" / "hemibrain" / "tiles",
        "metadata": _ROOT / "data" / "hemibrain" / "metadata.json",
    },
    "merfish": {
        "label": "Allen MERFISH",
        "domain": "Spatial transcriptomics",
        "geometry": "Point",
        "tiles_dir": _ROOT / "data" / "merfish" / "tiles",
        "metadata": _ROOT / "data" / "merfish" / "metadata.json",
    },
    "hubmap": {
        "label": "HuBMAP HRA",
        "domain": "Human organ anatomy",
        "geometry": "TIN (mesh)",
        "tiles_dir": _ROOT / "data" / "hubmap" / "tiles",
        "metadata": _ROOT / "data" / "hubmap" / "metadata.json",
    },
    "3dbag": {
        "label": "3DBAG Amsterdam",
        "domain": "Urban geospatial",
        "geometry": "TIN (mesh)",
        "tiles_dir": _ROOT / "data" / "3dbag" / "tiles",
        "metadata": _ROOT / "data" / "3dbag" / "metadata.json",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _dir_size_gzipped(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if not f.is_file():
            continue
        total += len(gzip.compress(f.read_bytes(), compresslevel=6))
    return total


def _collect_files(path: Path, suffix: str) -> list[Path]:
    return sorted(path.rglob(f"*{suffix}"))


# ---------------------------------------------------------------------------
# Per-dataset benchmarks
# ---------------------------------------------------------------------------

def benchmark_dataset(
    name: str,
    info: dict,
    *,
    decode_iters: int = 20,
    max_sample: int = 50,
    do_dataloader: bool = False,
) -> dict[str, Any] | None:
    """Run benchmarks for a single dataset. Returns results dict or None."""
    tiles_dir = info["tiles_dir"]
    pbf3_dir = tiles_dir / "pbf3"
    tiles3d_dir = tiles_dir / "3dtiles"

    has_pbf3 = pbf3_dir.exists() and list(pbf3_dir.rglob("*.pbf3"))
    has_3dt = tiles3d_dir.exists() and list(tiles3d_dir.rglob("*.glb"))

    if not has_pbf3 and not has_3dt:
        print(f"  SKIP: No tiles found at {tiles_dir}")
        return None

    results: dict[str, Any] = {
        "dataset": name,
        "label": info["label"],
        "domain": info["domain"],
        "geometry": info["geometry"],
    }

    # --- Tile stats ---
    if has_pbf3:
        pbf3_files = _collect_files(pbf3_dir, ".pbf3")
        results["pbf3_tile_count"] = len(pbf3_files)
        results["pbf3_size_raw"] = _dir_size(pbf3_dir)
        results["pbf3_size_gzip"] = _dir_size_gzipped(pbf3_dir)

    if has_3dt:
        glb_files = _collect_files(tiles3d_dir, ".glb")
        results["3dt_tile_count"] = len(glb_files)
        results["3dt_size_raw"] = _dir_size(tiles3d_dir)
        results["3dt_size_gzip"] = _dir_size_gzipped(tiles3d_dir)

    # --- Feature count from metadata ---
    meta_path = info.get("metadata")
    if meta_path and Path(meta_path).exists():
        try:
            meta = json.loads(Path(meta_path).read_text())
            if "features" in meta:
                results["feature_count"] = len(meta["features"])
            elif "cell_count" in meta:
                results["feature_count"] = meta["cell_count"]
            elif "building_count" in meta:
                results["feature_count"] = meta["building_count"]
            elif "neurons" in meta:
                results["feature_count"] = len(meta["neurons"])
            elif isinstance(meta, list):
                results["feature_count"] = len(meta)
        except Exception:
            pass

    # --- pbf3 decode benchmark ---
    if has_pbf3:
        pbf3_files = _collect_files(pbf3_dir, ".pbf3")
        sample = pbf3_files
        if len(sample) > max_sample:
            sample = random.sample(sample, max_sample)

        tile_bytes = [f.read_bytes() for f in sample]

        # Count features + triangles from one pass
        total_features = 0
        total_triangles = 0
        for data in tile_bytes:
            layers = decode_tile(data)
            for layer in layers:
                for feat in layer["features"]:
                    total_features += 1
                    mesh_pos = feat.get("mesh_positions", b"")
                    if mesh_pos:
                        total_triangles += len(mesh_pos) // 12 // 3  # 3 floats per vertex, 3 verts per tri

        results["sample_features"] = total_features

        # Warmup
        for data in tile_bytes[:3]:
            decode_tile(data)

        # Timed decode
        times: list[float] = []
        for _ in range(decode_iters):
            for data in tile_bytes:
                t0 = time.perf_counter()
                decode_tile(data)
                times.append(time.perf_counter() - t0)

        results["pbf3_decode_median_us"] = statistics.median(times) * 1_000_000
        results["pbf3_decode_p95_us"] = sorted(times)[int(len(times) * 0.95)] * 1_000_000

    # --- 3D Tiles decode benchmark ---
    if has_3dt:
        try:
            import pygltflib
        except ImportError:
            pygltflib = None

        if pygltflib:
            glb_files = _collect_files(tiles3d_dir, ".glb")
            sample = glb_files
            if len(sample) > max_sample:
                sample = random.sample(sample, max_sample)

            tile_bytes = [f.read_bytes() for f in sample]

            # Warmup
            for data in tile_bytes[:3]:
                pygltflib.GLTF2.load_from_bytes(data)

            times = []
            for _ in range(decode_iters):
                for data in tile_bytes:
                    t0 = time.perf_counter()
                    pygltflib.GLTF2.load_from_bytes(data)
                    times.append(time.perf_counter() - t0)

            results["3dt_decode_median_us"] = statistics.median(times) * 1_000_000
            results["3dt_decode_p95_us"] = sorted(times)[int(len(times) * 0.95)] * 1_000_000

    # --- Memory benchmark ---
    if has_pbf3:
        pbf3_bytes = [f.read_bytes() for f in _collect_files(pbf3_dir, ".pbf3")]
        tracemalloc.start()
        for data in pbf3_bytes:
            decode_tile(data)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["pbf3_peak_mb"] = peak / (1024 * 1024)

    if has_3dt and pygltflib:
        glb_bytes = [f.read_bytes() for f in _collect_files(tiles3d_dir, ".glb")]
        tracemalloc.start()
        for data in glb_bytes:
            pygltflib.GLTF2.load_from_bytes(data)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results["3dt_peak_mb"] = peak / (1024 * 1024)

    # --- Metadata extraction benchmark (pbf3) ---
    if has_pbf3:
        pbf3_files = _collect_files(pbf3_dir, ".pbf3")
        sample = pbf3_files[:max_sample]
        tile_bytes = [f.read_bytes() for f in sample]

        meta_times: list[float] = []
        for data in tile_bytes:
            t0 = time.perf_counter()
            layers = decode_tile(data)
            for layer in layers:
                for feat in layer["features"]:
                    _ = feat.get("properties", {})
            meta_times.append(time.perf_counter() - t0)

        if meta_times:
            results["pbf3_metadata_median_us"] = statistics.median(meta_times) * 1_000_000

    # --- DataLoader benchmark (optional) ---
    if do_dataloader and has_pbf3:
        try:
            import torch
            from torch.utils.data import DataLoader, Dataset

            class Mvt3Dataset(Dataset):
                def __init__(self, tile_dir: Path):
                    self._data = [f.read_bytes() for f in _collect_files(tile_dir, ".pbf3")]
                def __len__(self):
                    return len(self._data)
                def __getitem__(self, idx):
                    layers = decode_tile(self._data[idx])
                    parts = []
                    for layer in layers:
                        for feat in layer["features"]:
                            mesh = feat.get("mesh_positions", b"")
                            if mesh:
                                parts.append(torch.frombuffer(bytearray(mesh), dtype=torch.float32))
                    return torch.cat(parts) if parts else torch.zeros(1)

            ds = Mvt3Dataset(pbf3_dir)
            if len(ds) > 0:
                loader = DataLoader(ds, batch_size=4, num_workers=0, collate_fn=lambda x: x)
                total = 0
                t0 = time.perf_counter()
                for _ in range(3):
                    for batch in loader:
                        total += len(batch)
                elapsed = time.perf_counter() - t0
                results["pbf3_dataloader_tps"] = total / elapsed if elapsed > 0 else 0
        except ImportError:
            pass

    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

def print_comparison_table(all_results: list[dict]) -> None:
    """Print formatted comparison table to console."""
    print(f"\n{'=' * 120}")
    print(f"  Multi-Dataset Benchmark Comparison")
    print(f"  Backend: {'Rust' if RUST_AVAILABLE else 'Python'}")
    print(f"{'=' * 120}")

    # Header
    labels = [r["label"] for r in all_results]
    col_w = max(22, max(len(l) for l in labels) + 2)
    metric_w = 28

    header = f"  {'Metric':<{metric_w}}"
    for r in all_results:
        header += f" {r['label']:>{col_w}}"
    print(f"\n{header}")
    print(f"  {'─' * (metric_w + (col_w + 1) * len(all_results))}")

    def _row(metric: str, key: str, fmt_fn=str):
        line = f"  {metric:<{metric_w}}"
        for r in all_results:
            val = r.get(key)
            line += f" {fmt_fn(val) if val is not None else '—':>{col_w}}"
        print(line)

    # Tile stats
    _row("Domain", "domain")
    _row("Geometry type", "geometry")
    _row("Features", "feature_count", lambda v: f"{v:,}")
    _row("pbf3 tiles", "pbf3_tile_count", lambda v: f"{v:,}")
    _row("pbf3 raw size", "pbf3_size_raw", _fmt_bytes)
    _row("pbf3 gzip size", "pbf3_size_gzip", _fmt_bytes)
    _row("3DT tiles", "3dt_tile_count", lambda v: f"{v:,}")
    _row("3DT raw size", "3dt_size_raw", _fmt_bytes)

    print()
    _row("pbf3 decode median", "pbf3_decode_median_us", lambda v: f"{v:.0f}us")
    _row("pbf3 decode P95", "pbf3_decode_p95_us", lambda v: f"{v:.0f}us")
    _row("3DT decode median", "3dt_decode_median_us", lambda v: f"{v:.0f}us")
    _row("3DT decode P95", "3dt_decode_p95_us", lambda v: f"{v:.0f}us")

    # Speedup
    line = f"  {'Decode speedup (pbf3/3DT)':<{metric_w}}"
    for r in all_results:
        pbf3 = r.get("pbf3_decode_median_us")
        tdt = r.get("3dt_decode_median_us")
        if pbf3 and tdt and tdt > 0:
            line += f" {tdt / pbf3:>{col_w}.1f}x"
        else:
            line += f" {'—':>{col_w}}"
    print(line)

    print()
    _row("pbf3 peak memory", "pbf3_peak_mb", lambda v: f"{v:.1f} MB")
    _row("3DT peak memory", "3dt_peak_mb", lambda v: f"{v:.1f} MB")

    # Memory ratio
    line = f"  {'Memory ratio (3DT/pbf3)':<{metric_w}}"
    for r in all_results:
        pbf3 = r.get("pbf3_peak_mb")
        tdt = r.get("3dt_peak_mb")
        if pbf3 and tdt and pbf3 > 0:
            line += f" {tdt / pbf3:>{col_w}.1f}x"
        else:
            line += f" {'—':>{col_w}}"
    print(line)

    _row("pbf3 metadata extract", "pbf3_metadata_median_us", lambda v: f"{v:.0f}us")
    _row("pbf3 DataLoader t/s", "pbf3_dataloader_tps", lambda v: f"{v:.1f}")

    print()


def export_csv_table(path: Path, all_results: list[dict]) -> None:
    """Export results to CSV."""
    if not all_results:
        return

    all_keys = set()
    for r in all_results:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"CSV written to {path}")


def export_latex_table(path: Path, all_results: list[dict]) -> None:
    """Export results as a LaTeX table for the paper."""
    if not all_results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(all_results)
    col_spec = "l" + "r" * n

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Multi-dataset benchmark comparison. Decode and metadata times are per-tile medians.}",
        r"\label{tab:multi-dataset}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
    ]

    # Header row
    header = "Metric"
    for r in all_results:
        header += f" & {r['label']}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\hline")

    def _latex_row(label: str, key: str, fmt_fn=str):
        row = label
        for r in all_results:
            val = r.get(key)
            row += f" & {fmt_fn(val) if val is not None else '---'}"
        row += r" \\"
        lines.append(row)

    _latex_row("Domain", "domain")
    _latex_row("Geometry", "geometry")
    _latex_row("Features", "feature_count", lambda v: f"{v:,}")
    _latex_row("pbf3 tiles", "pbf3_tile_count", lambda v: f"{v:,}")
    _latex_row("pbf3 size (raw)", "pbf3_size_raw", _fmt_bytes)
    _latex_row("pbf3 size (gzip)", "pbf3_size_gzip", _fmt_bytes)
    _latex_row("3DT tiles", "3dt_tile_count", lambda v: f"{v:,}")
    _latex_row("3DT size (raw)", "3dt_size_raw", _fmt_bytes)
    lines.append(r"\hline")

    _latex_row(r"pbf3 decode ($\mu$s)", "pbf3_decode_median_us", lambda v: f"{v:.0f}")
    _latex_row(r"3DT decode ($\mu$s)", "3dt_decode_median_us", lambda v: f"{v:.0f}")

    # Speedup row
    row = "Decode speedup"
    for r in all_results:
        pbf3 = r.get("pbf3_decode_median_us")
        tdt = r.get("3dt_decode_median_us")
        if pbf3 and tdt and tdt > 0:
            row += f" & {tdt / pbf3:.0f}$\\times$"
        else:
            row += " & ---"
    row += r" \\"
    lines.append(row)
    lines.append(r"\hline")

    _latex_row("pbf3 memory (MB)", "pbf3_peak_mb", lambda v: f"{v:.1f}")
    _latex_row("3DT memory (MB)", "3dt_peak_mb", lambda v: f"{v:.1f}")

    # Memory ratio
    row = "Memory ratio"
    for r in all_results:
        pbf3 = r.get("pbf3_peak_mb")
        tdt = r.get("3dt_peak_mb")
        if pbf3 and tdt and pbf3 > 0:
            row += f" & {tdt / pbf3:.1f}$\\times$"
        else:
            row += " & ---"
    row += r" \\"
    lines.append(row)

    _latex_row(r"Metadata ($\mu$s)", "pbf3_metadata_median_us", lambda v: f"{v:.0f}")
    _latex_row("DataLoader (t/s)", "pbf3_dataloader_tps", lambda v: f"{v:.1f}")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])

    path.write_text("\n".join(lines))
    print(f"LaTeX table written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified multi-dataset benchmark runner",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=list(DATASETS.keys()),
        help="Datasets to benchmark (default: all available)",
    )
    parser.add_argument("--decode-iters", type=int, default=20, help="Decode iterations")
    parser.add_argument("--dataloader", action="store_true", help="Include DataLoader benchmark")
    parser.add_argument("--csv", type=Path, default=None, help="Export CSV")
    parser.add_argument("--latex", type=Path, default=None, help="Export LaTeX table")
    args = parser.parse_args()

    dataset_names = args.datasets or list(DATASETS.keys())

    print(f"Backend: {'Rust' if RUST_AVAILABLE else 'Python'}")
    print(f"Datasets: {', '.join(dataset_names)}\n")

    all_results: list[dict] = []

    for name in dataset_names:
        info = DATASETS[name]
        print(f"{'─' * 60}")
        print(f"Benchmarking: {info['label']} ({name})")
        print(f"{'─' * 60}")

        result = benchmark_dataset(
            name, info,
            decode_iters=args.decode_iters,
            do_dataloader=args.dataloader,
        )

        if result:
            all_results.append(result)
            print(f"  pbf3 decode: {result.get('pbf3_decode_median_us', 0):.0f}us median")
            if result.get("3dt_decode_median_us"):
                print(f"  3DT decode:  {result['3dt_decode_median_us']:.0f}us median")
        print()

    if all_results:
        print_comparison_table(all_results)

        if args.csv:
            export_csv_table(args.csv, all_results)

        if args.latex:
            export_latex_table(args.latex, all_results)

    print("Done.")


if __name__ == "__main__":
    main()
