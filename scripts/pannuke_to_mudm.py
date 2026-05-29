#!/usr/bin/env python3
"""Convert the PanNuke H&E dataset (HuggingFace parquet) to muDM tiles.

PanNuke (Gamper et al., 2019/2020) — pan-cancer nuclei segmentation and
classification across 19 tissue types and 5 cell categories. This script
reads the HuggingFace ``RationAI/PanNuke`` parquet folds, where each row is a
256x256 RGB image carrying a list of per-nucleus boolean instance masks and a
parallel list of integer cell-type ``categories``. Each nucleus is converted to
a 2D polygon (boundary contour) with its cell type, written as one GeoJSON
FeatureCollection per image, then tiled into a single Parquet file via the muDM
2D pipeline — the same output contract as ``download_consep.py`` so that
``benchmark_consep_ml.py`` can train on it unchanged.

Usage::

    uv run python scripts/pannuke_to_mudm.py \
        --pannuke-dir data/pannuke --output-dir data/pannuke --tile

    # Quick smoke test on a few images:
    uv run python scripts/pannuke_to_mudm.py --max-images 20 --tile
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from skimage.measure import find_contours

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# PanNuke cell-type categories (HuggingFace ClassLabel order, 0-indexed).
CELL_TYPE_NAMES = {
    0: "Neoplastic",
    1: "Inflammatory",
    2: "Connective",
    3: "Dead",
    4: "Epithelial",
}

IMG_SIZE = 256          # PanNuke tiles are 256x256 px
MIN_AREA_PX = 10        # drop sub-resolution instances
GRID_SPACING = 300      # px between images when laid on the tiling grid


def mask_to_polygon(mask: np.ndarray) -> list[list[float]] | None:
    """Return the outer-boundary polygon of a boolean instance *mask*.

    Coordinates are ``[x, y] = [col, row]`` in pixel space, closed ring.
    Returns None if the contour has fewer than 3 vertices.
    """
    contours = find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    if len(contour) < 3:
        return None
    polygon = [[float(pt[1]), float(pt[0])] for pt in contour]
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon


def row_to_features(instances: list, categories: list) -> list[dict]:
    """Convert one image's instance masks + categories to polygon features."""
    features: list[dict] = []
    for inst, cat in zip(instances, categories):
        cat = int(cat)
        if cat not in CELL_TYPE_NAMES:
            continue
        mask = np.array(Image.open(io.BytesIO(inst["bytes"])))
        mask = mask > 0  # boolean instance mask
        area_px = float(mask.sum())
        if area_px < MIN_AREA_PX:
            continue
        polygon = mask_to_polygon(mask)
        if polygon is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
            "properties": {
                "cell_type": CELL_TYPE_NAMES[cat],
                "cell_type_id": cat,
                "area_px": round(area_px, 1),
            },
        })
    return features


def convert_folds(
    pannuke_dir: Path, geojson_dir: Path, max_images: int = 0,
) -> dict:
    """Convert all PanNuke parquet folds to per-image GeoJSON files."""
    geojson_dir.mkdir(parents=True, exist_ok=True)
    fold_files = sorted((pannuke_dir / "data").glob("fold*-*.parquet"))
    if not fold_files:
        print(f"ERROR: no PanNuke parquet folds in {pannuke_dir/'data'}", file=sys.stderr)
        sys.exit(1)

    total_cells = 0
    images_processed = 0
    type_counts: dict[str, int] = {}
    image_ids: list[str] = []

    for fold_path in fold_files:
        fold = fold_path.stem.split("-")[0]  # e.g. "fold1"
        table = pq.read_table(fold_path, columns=["instances", "categories"])
        instances_col = table.column("instances").to_pylist()
        categories_col = table.column("categories").to_pylist()
        print(f"  {fold}: {len(instances_col)} images")

        for idx, (instances, categories) in enumerate(
            zip(instances_col, categories_col)
        ):
            if max_images and images_processed >= max_images:
                break
            features = row_to_features(instances, categories)
            if not features:
                continue
            image_id = f"{fold}_{idx:05d}"
            (geojson_dir / f"{image_id}.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": features})
            )
            image_ids.append(image_id)
            images_processed += 1
            total_cells += len(features)
            for f in features:
                ct = f["properties"]["cell_type"]
                type_counts[ct] = type_counts.get(ct, 0) + 1
        if max_images and images_processed >= max_images:
            break

    return {
        "dataset": "PanNuke",
        "source": "https://huggingface.co/datasets/RationAI/PanNuke",
        "reference": "Gamper et al., 2019/2020",
        "total_cells": total_cells,
        "total_images": images_processed,
        "type_counts": dict(sorted(type_counts.items())),
        "cell_types": {str(k): v for k, v in CELL_TYPE_NAMES.items()},
        "image_ids": image_ids,
    }


def tile_geojsons(output_dir: Path, image_ids: list[str]) -> None:
    """Tile per-image GeoJSON files into one Parquet via the muDM 2D pipeline."""
    from mudm_tools._rs import StreamingTileGenerator2D
    from mudm_tools.tiling2d import generate_parquet

    geojson_dir = output_dir / "geojson"
    n_cols = int(np.ceil(np.sqrt(len(image_ids))))
    offsets: list[tuple[str, float, float]] = []
    gx = gy = 0.0
    for idx, image_id in enumerate(image_ids):
        x_off = (idx % n_cols) * GRID_SPACING
        y_off = (idx // n_cols) * GRID_SPACING
        offsets.append((image_id, float(x_off), float(y_off)))
        gx = max(gx, x_off + IMG_SIZE)
        gy = max(gy, y_off + IMG_SIZE)
    world_bounds = (0.0, 0.0, gx, gy)
    print(f"  Tiling {len(image_ids)} images; world bounds {world_bounds}")

    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=64 / 4096.0)
    total_features = 0
    for image_id, x_off, y_off in offsets:
        raw = json.loads((geojson_dir / f"{image_id}.geojson").read_text())
        for feat in raw["features"]:
            for ring in feat["geometry"]["coordinates"]:
                for pt in ring:
                    pt[0] += x_off
                    pt[1] += y_off
        total_features += len(gen.add_geojson(json.dumps(raw), world_bounds))
    print(f"  Ingested {total_features} features")

    parquet_path = output_dir / "tiles.parquet"
    t0 = time.perf_counter()
    n_rows = generate_parquet(gen, parquet_path, world_bounds, simplify=False)
    print(f"  Wrote {n_rows} rows to {parquet_path} in {time.perf_counter()-t0:.2f}s")
    (output_dir / "tile_metadata.json").write_text(json.dumps({
        "world_bounds": list(world_bounds), "min_zoom": 0, "max_zoom": 4,
        "n_images": len(image_ids), "n_features": total_features,
        "n_parquet_rows": n_rows,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert PanNuke (HF parquet) to muDM tiles")
    ap.add_argument("--pannuke-dir", type=str, default="data/pannuke")
    ap.add_argument("--output-dir", type=str, default="data/pannuke")
    ap.add_argument("--tile", action="store_true", help="Also tile GeoJSON into Parquet")
    ap.add_argument("--max-images", type=int, default=0, help="Limit images (smoke test)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    geojson_dir = output_dir / "geojson"

    print("Step 1: Convert PanNuke instance masks to GeoJSON")
    meta = convert_folds(Path(args.pannuke_dir), geojson_dir, args.max_images)
    (output_dir / "mudm_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  Total nuclei: {meta['total_cells']} across {meta['total_images']} images")
    for ct, c in meta["type_counts"].items():
        print(f"    {ct}: {c}")

    if args.tile:
        print("\nStep 2: Tile GeoJSON into Parquet")
        tile_geojsons(output_dir, meta["image_ids"])
    print("\nDone.")


if __name__ == "__main__":
    main()
