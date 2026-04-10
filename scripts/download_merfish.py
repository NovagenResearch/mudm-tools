#!/usr/bin/env python3
"""Download Allen Brain Cell Atlas MERFISH data for benchmarking.

Downloads ~4M cell coordinates and metadata from the public S3 bucket,
converts to MuDM Point features, tiles with TileGenerator3D, and benchmarks.

Data is in Parquet format — pyarrow (already a dependency) handles reading.

Usage::

    # Full pipeline:
    .venv/bin/python scripts/download_merfish.py --download --convert --tile --benchmark

    # Download only:
    .venv/bin/python scripts/download_merfish.py --download

    # From existing local files:
    .venv/bin/python scripts/download_merfish.py --convert --tile --benchmark

    # Limit cells for testing:
    .venv/bin/python scripts/download_merfish.py --download --convert --tile --max-cells 100000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "merfish"
_TILES_DIR = _DATA_DIR / "tiles"

# S3 paths (public, no auth needed)
_S3_BUCKET = "s3://allen-brain-cell-atlas"
_S3_CELL_META = f"{_S3_BUCKET}/metadata/MERFISH-C57BL6J-638850/20241115/"
_S3_CCF_COORDS = f"{_S3_BUCKET}/metadata/MERFISH-C57BL6J-638850-CCF/20231215/"


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
    return f"{n / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Step 1: Download from S3
# ---------------------------------------------------------------------------

def download_from_s3(data_dir: Path) -> None:
    """Download MERFISH cell metadata and CCF coordinates from S3."""
    cell_dir = data_dir / "cell_metadata"
    ccf_dir = data_dir / "ccf_coordinates"

    cell_dir.mkdir(parents=True, exist_ok=True)
    ccf_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading cell metadata from S3...")
    t0 = time.perf_counter()
    subprocess.run(
        [
            "aws", "s3", "sync",
            _S3_CELL_META, str(cell_dir),
            "--no-sign-request",
        ],
        check=True,
    )
    print(f"  Cell metadata: {_fmt_time(time.perf_counter() - t0)}")

    print("Downloading CCF coordinates from S3...")
    t0 = time.perf_counter()
    subprocess.run(
        [
            "aws", "s3", "sync",
            _S3_CCF_COORDS, str(ccf_dir),
            "--no-sign-request",
        ],
        check=True,
    )
    print(f"  CCF coordinates: {_fmt_time(time.perf_counter() - t0)}")


# ---------------------------------------------------------------------------
# Step 2: Load and merge data
# ---------------------------------------------------------------------------

def load_cell_data(
    data_dir: Path,
    max_cells: int | None = None,
):
    """Load cell metadata + CCF coordinates from CSV or Parquet.

    Returns (cell_df, ccf_df) as pandas DataFrames.
    """
    import pandas as pd

    cell_dir = data_dir / "cell_metadata"
    ccf_dir = data_dir / "ccf_coordinates"

    # Try parquet first, fall back to CSV
    cell_parquet = sorted(cell_dir.glob("*.parquet"))
    cell_csv = sorted(cell_dir.glob("cell_metadata*.csv"))

    if cell_parquet:
        print(f"Loading cell metadata (parquet)...")
        t0 = time.perf_counter()
        cell_df = pd.read_parquet(cell_parquet[0])
    elif cell_csv:
        print(f"Loading cell metadata (CSV: {cell_csv[0].name})...")
        t0 = time.perf_counter()
        cell_df = pd.read_csv(
            cell_csv[0],
            dtype={"cell_label": str},
            nrows=max_cells,
        )
    else:
        print(f"ERROR: No cell_metadata.csv or .parquet in {cell_dir}",
              file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(cell_df):,} cells in "
          f"{_fmt_time(time.perf_counter() - t0)}")
    print(f"  Columns: {list(cell_df.columns)}")

    # Load CCF coordinates if available
    ccf_df = None
    ccf_parquet = sorted(ccf_dir.glob("*.parquet"))
    ccf_csv = sorted(ccf_dir.glob("ccf_coordinates*.csv"))

    if ccf_parquet:
        print(f"Loading CCF coordinates (parquet)...")
        t0 = time.perf_counter()
        ccf_df = pd.read_parquet(ccf_parquet[0])
    elif ccf_csv:
        print(f"Loading CCF coordinates (CSV: {ccf_csv[0].name})...")
        t0 = time.perf_counter()
        ccf_df = pd.read_csv(ccf_csv[0], dtype={"cell_label": str})
    else:
        print("  No CCF coordinate files found, using cell metadata coords")

    if ccf_df is not None:
        print(f"  Loaded {len(ccf_df):,} CCF rows in "
              f"{_fmt_time(time.perf_counter() - t0)}")

    # Limit rows if not already limited by nrows
    if max_cells and len(cell_df) > max_cells:
        cell_df = cell_df.head(max_cells)
        print(f"  Limited to {max_cells:,} cells")

    return cell_df, ccf_df


# ---------------------------------------------------------------------------
# Step 3: Convert to MuDM
# ---------------------------------------------------------------------------

def convert_to_microjson(
    cell_df,
    ccf_df,
    data_dir: Path,
):
    """Convert MERFISH cell data to MuDMFeatureCollection with Point geometry.

    Args:
        cell_df: pandas DataFrame with cell metadata (x, y, z, cluster_alias, etc.)
        ccf_df: pandas DataFrame with CCF coordinates, or None.
        data_dir: output directory for metadata.json.
    """
    from mudm.model import (
        MuDMFeature,
        MuDMFeatureCollection,
        OntologyTerm,
        Vocabulary,
    )

    # Try CCF coordinates first (registered to Allen CCF space)
    has_ccf = False
    if ccf_df is not None:
        # Merge on cell_label
        if "cell_label" in cell_df.columns and "cell_label" in ccf_df.columns:
            print("Merging cell metadata with CCF coordinates...")
            t0 = time.perf_counter()
            # Rename CCF x/y/z to avoid collision with cell_metadata x/y/z
            ccf_rename = {}
            for col in ccf_df.columns:
                if col != "cell_label" and col in cell_df.columns:
                    ccf_rename[col] = f"{col}_ccf"
            if ccf_rename:
                ccf_df = ccf_df.rename(columns=ccf_rename)
            cell_df = cell_df.merge(ccf_df, on="cell_label", how="left")
            has_ccf = any(c.endswith("_ccf") or c in ("x_ccf", "y_ccf", "z_ccf")
                         for c in cell_df.columns)
            print(f"  Merged in {_fmt_time(time.perf_counter() - t0)}")

    # Determine coordinate columns
    if has_ccf:
        x_col, y_col, z_col = "x_ccf", "y_ccf", "z_ccf"
        print("  Using CCF coordinates (x_ccf, y_ccf, z_ccf)")
    elif all(c in cell_df.columns for c in ["x_section", "y_section", "z_section"]):
        x_col, y_col, z_col = "x_section", "y_section", "z_section"
        print("  Using section coordinates (x_section, y_section, z_section)")
    elif all(c in cell_df.columns for c in ["x", "y", "z"]):
        x_col, y_col, z_col = "x", "y", "z"
        print("  Using x, y, z coordinates")
    else:
        # Try to find any coordinate-like columns
        coord_cols = [c for c in cell_df.columns if c.startswith(("x_", "y_", "z_"))]
        print(f"Available columns: {list(cell_df.columns)}")
        print(f"Coordinate-like columns: {coord_cols}", file=sys.stderr)
        print("ERROR: Could not find coordinate columns", file=sys.stderr)
        sys.exit(1)

    # Drop rows with missing coordinates
    valid_mask = cell_df[x_col].notna() & cell_df[y_col].notna() & cell_df[z_col].notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        cell_df = cell_df[valid_mask]
        print(f"  Dropped {n_dropped:,} cells with missing coordinates")

    n_cells = len(cell_df)
    print(f"Converting {n_cells:,} cells to MuDM Point features...")
    t0 = time.perf_counter()

    # Determine metadata columns
    meta_cols = []
    for col in ["cluster_alias", "class", "subclass", "supertype",
                 "brain_section_label", "parcellation_index"]:
        if col in cell_df.columns:
            meta_cols.append(col)

    # Build features in batches for memory efficiency
    features: list[MuDMFeature] = []
    cell_types = set()

    coords_x = cell_df[x_col].values
    coords_y = cell_df[y_col].values
    coords_z = cell_df[z_col].values

    for i in range(n_cells):
        if (i + 1) % 500_000 == 0 or i + 1 == n_cells:
            print(f"  [{i+1:,}/{n_cells:,}]", end="\r", file=sys.stderr)

        x, y, z = float(coords_x[i]), float(coords_y[i]), float(coords_z[i])

        props: dict = {}
        feature_class = "cell"

        for col in meta_cols:
            val = cell_df.iloc[i][col]
            if val is not None and str(val) != "nan":
                props[col] = str(val) if not isinstance(val, (int, float)) else val

        ct = props.get("cluster_alias") or props.get("subclass") or props.get("class")
        if ct:
            feature_class = str(ct)
            cell_types.add(feature_class)

        features.append(MuDMFeature(
            type="Feature",
            geometry={
                "type": "Point",
                "coordinates": [x, y, z],
            },
            properties=props,
            featureClass=feature_class,
        ))

    print(file=sys.stderr)
    convert_time = time.perf_counter() - t0

    # Build vocabulary
    vocabs = None
    if cell_types:
        terms = {
            ct: OntologyTerm(
                uri=f"https://knowledge.brain-map.org/celltypes/CCN/{ct}",
                label=ct,
            )
            for ct in sorted(cell_types)
        }
        vocabs = {
            "abc_cell_types": Vocabulary(
                namespace="https://knowledge.brain-map.org/celltypes/",
                description="Allen Brain Cell Atlas cell type taxonomy",
                terms=terms,
            ),
        }

    collection = MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={
            "dataset": "MERFISH-C57BL6J-638850",
            "cell_count": n_cells,
            "cell_types": len(cell_types),
            "coordinate_space": "CCF" if has_ccf else "section",
        },
        vocabularies=vocabs,
    )

    print(f"  {n_cells:,} features, {len(cell_types)} cell types")
    print(f"  Conversion time: {_fmt_time(convert_time)}")

    # Save metadata summary
    meta = {
        "dataset": "MERFISH-C57BL6J-638850",
        "cell_count": n_cells,
        "cell_types": sorted(cell_types),
        "coordinate_space": "CCF" if has_ccf else "section",
        "coordinate_range": {
            "x": [float(coords_x.min()), float(coords_x.max())],
            "y": [float(coords_y.min()), float(coords_y.max())],
            "z": [float(coords_z.min()), float(coords_z.max())],
        },
    }
    meta_path = data_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Saved metadata to {meta_path}")

    return collection, convert_time


# ---------------------------------------------------------------------------
# Step 4 & 5: Tile + Benchmark
# ---------------------------------------------------------------------------

def tile_and_benchmark(
    collection,
    output_dir: Path,
    *,
    max_zoom: int = 3,
    workers: int | None = None,
    do_tile: bool = True,
    do_benchmark: bool = True,
    skip_3dtiles: bool = False,
    csv_path: Path | None = None,
) -> dict:
    """Tile the collection and run benchmarks."""
    from benchmark_mouselight import (
        bench_decode,
        bench_decode_3dtiles,
        bench_memory,
        export_csv,
        generate_tiles,
        print_report,
    )

    results: dict = {}

    if do_tile:
        tile_results = generate_tiles(
            collection,
            output_dir,
            max_zoom=max_zoom,
            workers=workers,
            skip_3dtiles=skip_3dtiles,
        )
        results["tile"] = tile_results

    if do_benchmark:
        pbf3_dir = output_dir / "pbf3"
        tiles3d_dir = output_dir / "3dtiles" if not skip_3dtiles else None

        decode_pbf3: dict = {}
        if pbf3_dir.exists():
            print(f"\nBenchmarking pbf3 decode...")
            decode_pbf3 = bench_decode(pbf3_dir)

        decode_3dt: dict = {}
        if tiles3d_dir and tiles3d_dir.exists():
            print(f"Benchmarking 3D Tiles decode...")
            decode_3dt = bench_decode_3dtiles(tiles3d_dir)

        print("Measuring peak memory...")
        memory = bench_memory(
            pbf3_dir if pbf3_dir.exists() else Path("/dev/null"),
            tiles3d_dir if tiles3d_dir and tiles3d_dir.exists() else None,
        )

        results["decode_pbf3"] = decode_pbf3
        results["decode_3dt"] = decode_3dt
        results["memory"] = memory

        if do_tile:
            print_report(0, tile_results, decode_pbf3, decode_3dt, memory, None)

        if csv_path:
            export_csv(csv_path, 0, tile_results if do_tile else {},
                       decode_pbf3, decode_3dt, memory, None)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Allen MERFISH download, conversion, tiling, and benchmark",
    )
    parser.add_argument("--download", action="store_true", help="Download from S3")
    parser.add_argument("--convert", action="store_true", help="Convert to MuDM")
    parser.add_argument("--tile", action="store_true", help="Generate tiles")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--max-cells", type=int, default=None, help="Limit number of cells")
    parser.add_argument("--max-zoom", type=int, default=3, help="Max zoom level")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument("--skip-3dtiles", action="store_true", help="Skip 3D Tiles")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--csv", type=Path, default=None, help="Export CSV")
    args = parser.parse_args()

    if not any([args.download, args.convert, args.tile, args.benchmark]):
        parser.print_help()
        sys.exit(1)

    data_dir = args.data_dir
    tiles_dir = data_dir / "tiles"

    # --- Download ---
    if args.download:
        download_from_s3(data_dir)

    # --- Convert ---
    collection = None
    convert_time = 0.0
    if args.convert or args.tile or args.benchmark:
        cell_table, ccf_table = load_cell_data(data_dir, max_cells=args.max_cells)
        collection, convert_time = convert_to_microjson(cell_table, ccf_table, data_dir)

    # --- Tile + Benchmark ---
    if args.tile or args.benchmark:
        tile_and_benchmark(
            collection,
            tiles_dir,
            max_zoom=args.max_zoom,
            workers=args.workers,
            do_tile=args.tile,
            do_benchmark=args.benchmark,
            skip_3dtiles=args.skip_3dtiles,
            csv_path=args.csv,
        )

    print("Done.")


if __name__ == "__main__":
    main()
