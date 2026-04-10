#!/usr/bin/env python3
"""Download the CoNSeP dataset and convert instance masks to GeoJSON.

CoNSeP = Colorectal Nuclear Segmentation and Phenotypes dataset
(Graham et al., Medical Image Analysis 2019).

~25,000 nuclei across 41 H&E histopathology images (1000x1000 px).
Cell types: inflammatory, epithelial, dysplastic/malignant, fibroblast,
muscle, endothelial.

Data source: https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/

Usage::

    uv run python scripts/download_consep.py --output-dir data/consep
    uv run python scripts/download_consep.py --output-dir data/consep --tile

If the automatic download fails (the Warwick server may require a browser),
manually download CoNSeP.zip and place it at data/consep/CoNSeP.zip, then
re-run the script.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# CoNSeP cell type mapping.
# The .mat files use inst_type with these IDs:
#   0 = background (skip)
#   1 = other / miscellaneous (skip — too heterogeneous)
#   2 = inflammatory
#   3 = epithelial (healthy)
#   4 = dysplastic / malignant epithelial
#   5 = fibroblast
#   6 = muscle
#   7 = endothelial
# Note: some versions of the dataset use slightly different numbering.
# We detect the actual mapping from the data.
CELL_TYPE_NAMES = {
    1: "other",
    2: "inflammatory",
    3: "epithelial",
    4: "dysplastic_malignant",
    5: "fibroblast",
    6: "muscle",
    7: "endothelial",
}

# Skip background (0) and "other" (1) — too heterogeneous for classification
SKIP_TYPES = {0, 1}

# Minimum area in pixels to keep an instance
MIN_AREA_PX = 10

# Download URLs to try (Warwick site requires login — these may not work
# without a browser session).  The script tries each URL in order and falls
# back to clear manual-download instructions.
CONSEP_URLS = [
    "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep.zip",
    "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/CoNSeP.zip",
    "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep_dataset.zip",
]


def download_consep(output_dir: Path) -> Path:
    """Download and extract CoNSeP.zip."""
    zip_path = output_dir / "CoNSeP.zip"

    if zip_path.exists():
        print(f"  ZIP already exists: {zip_path}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Downloading CoNSeP dataset...")
        downloaded = False
        for url in CONSEP_URLS:
            try:
                print(f"    Trying {url}")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    # Check for redirect to login page
                    if "slogin" in resp.url or "websignon" in resp.url:
                        print(f"    Redirected to login page — skipping")
                        continue
                    data = resp.read()
                    # Sanity check: a valid ZIP starts with PK (0x504B)
                    if len(data) < 100 or data[:2] != b"PK":
                        print(f"    Response is not a ZIP file ({len(data)} bytes) — skipping")
                        continue
                    zip_path.write_bytes(data)
                print(f"    Downloaded {len(data) / (1024*1024):.1f} MB")
                downloaded = True
                break
            except Exception as e:
                print(f"    Failed: {e}")
        if not downloaded:
            print(
                "\n  Automatic download failed (Warwick server requires browser login).\n"
                "  Please manually download CoNSeP.zip:\n"
                "    1. Visit https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/\n"
                "    2. Log in / accept terms if prompted\n"
                "    3. Download CoNSeP.zip\n"
                f"    4. Place the file at: {zip_path}\n"
                "    5. Re-run this script\n"
            )
            sys.exit(1)

    # Extract
    extract_dir = output_dir / "raw"
    if extract_dir.exists() and any(extract_dir.rglob("*.mat")):
        print(f"  Already extracted: {extract_dir}")
    else:
        print(f"  Extracting to {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"  Extracted {len(list(extract_dir.rglob('*.mat')))} .mat files")

    return extract_dir


def load_mat_file(mat_path: Path) -> dict:
    """Load a CoNSeP .mat file, handling both v5 and v7.3 formats."""
    import scipy.io

    try:
        data = scipy.io.loadmat(str(mat_path))
        return data
    except NotImplementedError:
        # v7.3 format requires h5py
        try:
            import h5py
        except ImportError:
            print("  ERROR: .mat file is v7.3 format. Install h5py: uv add --dev h5py")
            sys.exit(1)
        with h5py.File(str(mat_path), "r") as f:
            data = {}
            for key in f.keys():
                data[key] = np.array(f[key])
            return data


def instance_mask_to_polygons(
    inst_map: np.ndarray,
    type_map: np.ndarray | None,
    inst_type: np.ndarray | None,
) -> list[dict]:
    """Convert instance segmentation mask to polygon features.

    Args:
        inst_map: (H, W) integer array, each non-zero value is one instance.
        type_map: (H, W) integer array, cell type per pixel. May be None.
        inst_type: (N, 2) array of (instance_id, type_id). May be None.

    Returns:
        List of dicts with keys: polygon, cell_type_id, area_px.
    """
    from skimage.measure import find_contours, regionprops

    # Build instance_id -> type_id lookup
    type_lookup: dict[int, int] = {}
    if inst_type is not None:
        # inst_type shape is typically (N, 2): [inst_id, type]
        for row in inst_type:
            iid = int(row[0])
            tid = int(row[1])
            type_lookup[iid] = tid

    features = []
    instance_ids = np.unique(inst_map)

    for iid in instance_ids:
        iid = int(iid)
        if iid == 0:
            continue  # background

        # Get cell type
        if iid in type_lookup:
            cell_type_id = type_lookup[iid]
        elif type_map is not None:
            # Majority vote from type_map
            mask = inst_map == iid
            types_in_mask = type_map[mask]
            if len(types_in_mask) == 0:
                continue
            cell_type_id = int(np.bincount(types_in_mask.astype(int)).argmax())
        else:
            continue

        if cell_type_id in SKIP_TYPES:
            continue

        # Compute area
        mask = (inst_map == iid).astype(np.uint8)
        area_px = float(mask.sum())
        if area_px < MIN_AREA_PX:
            continue

        # Find contour — use 0.5 level to get the boundary
        contours = find_contours(mask, level=0.5)
        if not contours:
            continue

        # Take the longest contour (outer boundary)
        contour = max(contours, key=len)
        if len(contour) < 3:
            continue

        # contour is (N, 2) in (row, col) order → convert to [x, y] = [col, row]
        polygon = [[float(pt[1]), float(pt[0])] for pt in contour]

        # Close the polygon if not already closed
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])

        features.append({
            "polygon": polygon,
            "cell_type_id": cell_type_id,
            "area_px": area_px,
        })

    return features


def features_to_geojson(features: list[dict], image_id: str) -> dict:
    """Convert extracted features to a GeoJSON FeatureCollection."""
    geojson_features = []
    for feat in features:
        cell_type_id = feat["cell_type_id"]
        cell_type_name = CELL_TYPE_NAMES.get(cell_type_id, f"unknown_{cell_type_id}")

        geojson_features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [feat["polygon"]],
            },
            "properties": {
                "cell_type": cell_type_name,
                "cell_type_id": cell_type_id,
                "area_px": round(feat["area_px"], 1),
                "image_id": image_id,
            },
        })

    return {
        "type": "FeatureCollection",
        "features": geojson_features,
    }


def process_mat_files(raw_dir: Path, geojson_dir: Path) -> dict:
    """Process all .mat files and write GeoJSON files.

    Returns metadata dict with counts.
    """
    import scipy.io

    geojson_dir.mkdir(parents=True, exist_ok=True)

    # Find all .mat files (both Train/ and Test/ subdirectories)
    mat_files = sorted(raw_dir.rglob("*.mat"))

    # Filter to label files (Labels/ subdirectory, not Images/)
    label_mats = [m for m in mat_files if "label" in str(m).lower()]
    if not label_mats:
        # Some extractions put everything flat — just use all .mat files
        label_mats = mat_files

    print(f"  Found {len(label_mats)} label .mat files")

    total_cells = 0
    type_counts: dict[str, int] = {}
    images_processed = 0
    all_image_bounds: list[tuple[float, float, float, float]] = []

    for mat_path in label_mats:
        image_id = mat_path.stem
        print(f"    Processing {image_id}...", end="", flush=True)

        data = load_mat_file(mat_path)

        # Extract arrays — keys vary by dataset version
        inst_map = None
        type_map = None
        inst_type = None

        for key in ["inst_map", "inst_map "]:
            if key in data:
                inst_map = np.squeeze(data[key]).astype(np.int32)
                break

        for key in ["type_map", "type_map "]:
            if key in data:
                type_map = np.squeeze(data[key]).astype(np.int32)
                break

        for key in ["inst_type", "inst_type "]:
            if key in data:
                val = data[key]
                if isinstance(val, np.ndarray) and val.ndim >= 2:
                    inst_type = val.astype(np.int32)
                break

        if inst_map is None:
            print(f" SKIPPED (no inst_map, keys: {[k for k in data.keys() if not k.startswith('__')]})")
            continue

        features = instance_mask_to_polygons(inst_map, type_map, inst_type)
        geojson = features_to_geojson(features, image_id)

        # Write GeoJSON
        out_path = geojson_dir / f"{image_id}.geojson"
        out_path.write_text(json.dumps(geojson))

        n = len(geojson["features"])
        total_cells += n
        images_processed += 1

        # Track bounds (image is HxW pixels, coordinates are in pixel space)
        h, w = inst_map.shape
        all_image_bounds.append((0.0, 0.0, float(w), float(h)))

        # Count types
        for feat in geojson["features"]:
            ct = feat["properties"]["cell_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1

        print(f" {n} cells")

    # Compute union bounds across all images
    # Since each image has its own coordinate system (pixel space),
    # for tiling we assign each image an offset so they tile together
    # Layout: stack images horizontally with 100px gaps
    metadata = {
        "dataset": "CoNSeP",
        "source": "https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/",
        "reference": "Graham et al., Medical Image Analysis 2019",
        "total_cells": total_cells,
        "total_images": images_processed,
        "type_counts": dict(sorted(type_counts.items())),
        "cell_types": {str(k): v for k, v in CELL_TYPE_NAMES.items() if k not in SKIP_TYPES},
        "image_bounds": {
            f"image_{i}": list(b) for i, b in enumerate(all_image_bounds)
        },
    }

    return metadata


def tile_geojson(output_dir: Path, metadata: dict) -> None:
    """Tile all GeoJSON files using the muDM 2D pipeline."""
    from mudm_tools._rs import StreamingTileGenerator2D
    from mudm_tools.tiling2d import generate_parquet

    geojson_dir = output_dir / "geojson"
    geojson_files = sorted(geojson_dir.glob("*.geojson"))

    if not geojson_files:
        print("  No GeoJSON files found to tile.")
        return

    # Assign each image a spatial offset so features don't overlap.
    # Layout: place images on a grid (images are ~1000x1000 px).
    # Use 1100px spacing to leave gaps.
    SPACING = 1100
    n_cols = int(np.ceil(np.sqrt(len(geojson_files))))

    print(f"  Tiling {len(geojson_files)} images into Parquet...")
    print(f"  Layout: {n_cols} columns, {SPACING}px spacing")

    # First pass: compute total bounds and prepare offset data
    image_offsets: list[tuple[str, float, float]] = []
    global_xmax = 0.0
    global_ymax = 0.0

    for idx, gj_path in enumerate(geojson_files):
        col = idx % n_cols
        row = idx // n_cols
        x_off = col * SPACING
        y_off = row * SPACING
        image_offsets.append((str(gj_path), float(x_off), float(y_off)))
        global_xmax = max(global_xmax, x_off + 1000.0)
        global_ymax = max(global_ymax, y_off + 1000.0)

    world_bounds = (0.0, 0.0, global_xmax, global_ymax)
    print(f"  World bounds: {world_bounds}")

    # Choose zoom levels — images are ~1000px, tiles at zoom=0 cover full extent.
    # At zoom 4, tiles cover ~(extent/16) which is reasonable.
    min_zoom = 0
    max_zoom = 4
    buffer = 64 / 4096.0

    gen = StreamingTileGenerator2D(
        min_zoom=min_zoom, max_zoom=max_zoom, buffer=buffer
    )

    # Second pass: offset coordinates and add to generator
    total_features = 0
    for gj_path_str, x_off, y_off in image_offsets:
        gj_path = Path(gj_path_str)
        raw = json.loads(gj_path.read_text())

        # Offset all coordinates
        for feat in raw["features"]:
            coords = feat["geometry"]["coordinates"]
            for ring in coords:
                for pt in ring:
                    pt[0] += x_off
                    pt[1] += y_off

        geojson_str = json.dumps(raw)
        fids = gen.add_geojson(geojson_str, world_bounds)
        total_features += len(fids)

    print(f"  Ingested {total_features} features")

    # Write Parquet
    parquet_path = output_dir / "tiles.parquet"
    t0 = time.perf_counter()
    n_rows = generate_parquet(gen, parquet_path, world_bounds, simplify=False)
    elapsed = time.perf_counter() - t0
    print(f"  Wrote {n_rows} rows to {parquet_path} in {elapsed:.2f}s")

    # Also save the tiling metadata
    tile_meta = {
        "world_bounds": list(world_bounds),
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "n_images": len(geojson_files),
        "n_features": total_features,
        "n_parquet_rows": n_rows,
        "image_offsets": [
            {"file": Path(p).stem, "x_offset": xo, "y_offset": yo}
            for p, xo, yo in image_offsets
        ],
    }
    (output_dir / "tile_metadata.json").write_text(json.dumps(tile_meta, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Download CoNSeP dataset and convert to GeoJSON"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/consep",
        help="Output directory for downloaded data and GeoJSON files",
    )
    parser.add_argument(
        "--tile", action="store_true",
        help="Also tile the GeoJSON files into a Parquet file using the 2D pipeline",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, assume data already exists in raw/",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    geojson_dir = output_dir / "geojson"
    meta_path = output_dir / "metadata.json"

    # Step 1: Download
    if not args.skip_download:
        print("Step 1: Download CoNSeP dataset")
        raw_dir = download_consep(output_dir)
    else:
        raw_dir = output_dir / "raw"
        if not raw_dir.exists():
            print(f"ERROR: --skip-download but {raw_dir} does not exist")
            sys.exit(1)
        print("Step 1: Skipping download (--skip-download)")

    # Step 2: Convert to GeoJSON
    print("\nStep 2: Convert instance masks to GeoJSON")
    if meta_path.exists() and any(geojson_dir.glob("*.geojson")):
        print(f"  Already converted. Loading existing metadata from {meta_path}")
        metadata = json.loads(meta_path.read_text())
    else:
        metadata = process_mat_files(raw_dir, geojson_dir)
        meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"\n  Total cells: {metadata['total_cells']}")
    print(f"  Total images: {metadata['total_images']}")
    print(f"  Type counts:")
    for ct, count in sorted(metadata["type_counts"].items()):
        print(f"    {ct}: {count}")

    # Step 3: Tile (optional)
    if args.tile:
        print("\nStep 3: Tile GeoJSON into Parquet")
        tile_geojson(output_dir, metadata)

    print("\nDone.")


if __name__ == "__main__":
    main()
