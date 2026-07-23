"""Convert 10x Genomics Xenium spatial transcriptomics data to muDM tiled format.

Usage:
    python scripts/xenium_to_tiles.py \
        --data-dir data/Xenium_V1_Protein_Human_Kidney_tiny_outs \
        --output-dir tiles/kidney_tiny
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from PIL import Image

from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d import generate_pbf, generate_parquet


LAYER_COLORS = {
    "cells": "#00ffff",
    "nuclei": "#00ff00",
    "transcripts": "#ff4444",
}


def boundaries_to_geojson(
    parquet_path: str | Path,
    id_column: str = "cell_id",
    coord_scale: float = 1.0,
) -> str:
    """Convert a Xenium boundary parquet file to a GeoJSON FeatureCollection string.

    Args:
        coord_scale: Multiply all coordinates by this factor.
            Use 1/um_per_px to convert from microns to pixels.
    """
    df = pl.read_parquet(parquet_path)

    if id_column not in df.columns:
        for alt in ("label_id", "cell_id"):
            if alt in df.columns:
                id_column = alt
                break

    features = []
    for group in df.partition_by(id_column, maintain_order=True):
        cell_id = str(group[id_column][0])
        xs = group["vertex_x"].to_list()
        ys = group["vertex_y"].to_list()

        ring = [[x * coord_scale, y * coord_scale] for x, y in zip(xs, ys)]
        if ring[0] != ring[-1]:
            ring.append(ring[0])

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ring],
            },
            "properties": {id_column: cell_id},
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }
    return json.dumps(fc)


def transcripts_to_geojson(
    parquet_path: str | Path,
    coord_scale: float = 1.0,
) -> str:
    """Convert a Xenium transcripts parquet file to a GeoJSON FeatureCollection string.

    Args:
        coord_scale: Multiply all coordinates by this factor.
            Use 1/um_per_px to convert from microns to pixels.
    """
    df = pl.read_parquet(parquet_path)

    # feature_name may be Binary in older Xenium formats — cast to String
    if df.schema["feature_name"] == pl.Binary:
        df = df.with_columns(pl.col("feature_name").cast(pl.Utf8))

    xs = df["x_location"].to_list()
    ys = df["y_location"].to_list()
    genes = df["feature_name"].to_list()

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [x * coord_scale, y * coord_scale]},
            "properties": {"gene_name": gene},
        }
        for x, y, gene in zip(xs, ys, genes)
    ]

    return json.dumps({"type": "FeatureCollection", "features": features})


def add_boundaries_direct(
    gen: StreamingTileGenerator2D,
    parquet_path: str | Path,
    bounds: tuple[float, float, float, float],
    id_column: str = "cell_id",
    coord_scale: float = 1.0,
    layer_type: str = "cells",
) -> int:
    """Add boundary polygons directly to a generator, bypassing GeoJSON.

    Returns number of features added.
    """
    df = pl.read_parquet(parquet_path)

    if id_column not in df.columns:
        for alt in ("label_id", "cell_id"):
            if alt in df.columns:
                id_column = alt
                break

    count = 0
    for group in df.partition_by(id_column, maintain_order=True):
        cell_id = str(group[id_column][0])
        xs = (group["vertex_x"].to_numpy() * coord_scale).astype(np.float64)
        ys = (group["vertex_y"].to_numpy() * coord_scale).astype(np.float64)

        # Flatten to [x0,y0,x1,y1,...] for add_feature
        xy = np.empty(len(xs) * 2, dtype=np.float64)
        xy[0::2] = xs
        xy[1::2] = ys

        # Close ring if needed
        if xs[0] != xs[-1] or ys[0] != ys[-1]:
            xy = np.append(xy, [xs[0], ys[0]])

        n_verts = len(xy) // 2

        gen.add_feature({
            "xy": xy.tolist(),
            "geom_type": 3,  # Polygon
            "ring_lengths": [n_verts],
            "min_x": float(xs.min()),
            "min_y": float(ys.min()),
            "max_x": float(xs.max()),
            "max_y": float(ys.max()),
            "tags": {id_column: cell_id, "layer_type": layer_type},
        })
        count += 1

    return count


def add_transcripts_direct(
    gen: StreamingTileGenerator2D,
    parquet_path: str | Path,
    bounds: tuple[float, float, float, float],
    coord_scale: float = 1.0,
    layer_type: str = "transcripts",
) -> int:
    """Add transcript points directly to a generator, bypassing GeoJSON.

    Returns number of features added.
    """
    df = pl.read_parquet(parquet_path)

    if df.schema["feature_name"] == pl.Binary:
        df = df.with_columns(pl.col("feature_name").cast(pl.Utf8))

    xs = (df["x_location"].to_numpy() * coord_scale).astype(np.float64)
    ys = (df["y_location"].to_numpy() * coord_scale).astype(np.float64)
    genes = df["feature_name"].to_list()

    for i in range(len(xs)):
        x, y = float(xs[i]), float(ys[i])
        gen.add_feature({
            "xy": [x, y],
            "geom_type": 1,  # Point
            "ring_lengths": [],
            "min_x": x,
            "min_y": y,
            "max_x": x,
            "max_y": y,
            "tags": {"gene_name": genes[i], "layer_type": layer_type},
        })

    return len(xs)


def read_um_per_px(experiment_path: str | Path) -> float:
    """Read microns-per-pixel from experiment.xenium JSON."""
    with open(experiment_path) as f:
        data = json.load(f)
    return float(data["pixel_size"])


def generate_raster_tiles(
    image_path: str | Path,
    output_dir: str | Path,
    tile_size: int = 256,
) -> dict:
    """Generate a PNG tile pyramid from a DAPI image.

    Reads the image, contrast-normalizes (1st-99th percentile),
    and slices into {z}/{x}/{y}.png tiles at multiple zoom levels.

    Returns dict with max_zoom and image_size_px.
    """
    output_dir = Path(output_dir)

    # Read image
    with tifffile.TiffFile(str(image_path)) as tf:
        raw = tf.pages[0].asarray()
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {raw.shape}")

    # Contrast normalize to uint8
    raw = raw.astype(np.float32)
    p_lo, p_hi = np.percentile(raw, [1, 99])
    normalized = np.clip((raw - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
    img_uint8 = (normalized * 255).astype(np.uint8)

    h, w = img_uint8.shape
    max_zoom = max(0, math.ceil(math.log2(max(h, w) / tile_size)))

    # Build from max zoom down to 0
    current = img_uint8
    for z in range(max_zoom, -1, -1):
        ch, cw = current.shape
        nx = math.ceil(cw / tile_size)
        ny = math.ceil(ch / tile_size)

        for ty in range(ny):
            for tx in range(nx):
                y0 = ty * tile_size
                x0 = tx * tile_size
                y1 = min(y0 + tile_size, ch)
                x1 = min(x0 + tile_size, cw)
                tile_data = current[y0:y1, x0:x1]

                if tile_data.shape != (tile_size, tile_size):
                    padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
                    padded[: tile_data.shape[0], : tile_data.shape[1]] = tile_data
                    tile_data = padded

                tile_path = output_dir / str(z) / str(tx) / f"{ty}.png"
                tile_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(tile_data, mode="L").save(tile_path)

        # Downsample for next zoom level
        if z > 0:
            pil_img = Image.fromarray(current, mode="L")
            new_w = max(1, cw // 2)
            new_h = max(1, ch // 2)
            current = np.array(pil_img.resize((new_w, new_h), Image.LANCZOS))

    return {
        "max_zoom": max_zoom,
        "image_size_px": [w, h],
    }


def generate_vector_tiles(
    geojson_str: str,
    output_dir: str | Path,
    bounds: tuple[float, float, float, float],
    layer_name: str,
    min_zoom: int = 0,
    max_zoom: int = 7,
    temp_dir: str = "/data/tmp",
) -> dict:
    """Generate MVT + Parquet tiles from a GeoJSON FeatureCollection string.

    Returns dict with feature_count and tile_count.
    """
    output_dir = Path(output_dir)
    mvt_dir = output_dir / layer_name
    parquet_path = output_dir / f"{layer_name}.parquet"

    gen = StreamingTileGenerator2D(
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        buffer=64 / 4096.0,
        temp_dir=temp_dir,
    )

    fids = gen.add_geojson(geojson_str, bounds)

    tile_count = generate_pbf(
        gen,
        str(mvt_dir),
        bounds,
        simplify=True,
        layer_name=layer_name,
    )

    generate_parquet(
        gen,
        str(parquet_path),
        bounds,
        simplify=True,
    )

    return {"feature_count": len(fids), "tile_count": tile_count}


def generate_combined_tiles(
    geojson_layers: list[tuple[str, str]],
    output_dir: str | Path,
    bounds: tuple[float, float, float, float],
    min_zoom: int = 0,
    max_zoom: int = 7,
    temp_dir: str = "/data/tmp",
) -> dict:
    """Generate a single MVT pyramid with multiple named layers + partitioned Parquet.

    Each layer gets its own named MVT layer inside every .pbf tile.
    MVT protobuf tiles are merged by concatenating per-layer bytes
    (valid because MVT layers are repeated protobuf fields).

    Args:
        geojson_layers: list of (layer_name, geojson_str) tuples.

    Returns dict with per-layer feature counts and total tile count.
    """
    import shutil
    import tempfile

    output_dir = Path(output_dir)
    mvt_dir = output_dir / "vectors"
    parquet_dir = output_dir / "features.parquet"

    # Generate each layer into a separate temp directory, then merge
    layer_counts = {}
    layer_tmp_dirs = []
    layer_fields = {}  # layer_name → {field: type}

    for layer_name, geojson_str in geojson_layers:
        # Collect user-facing field names, inject layer_type for viewer
        fc = json.loads(geojson_str)
        fields = {}
        for feat in fc["features"]:
            for k in feat.get("properties", {}):
                fields[k] = "String"
            feat["properties"]["layer_type"] = layer_name
        layer_fields[layer_name] = fields  # layer_type excluded from TileJSON

        gen = StreamingTileGenerator2D(
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            buffer=64 / 4096.0,
            temp_dir=temp_dir,
        )
        fids = gen.add_geojson(json.dumps(fc), bounds)
        layer_counts[layer_name] = len(fids)

        # Generate PBF into a temp directory with the layer's name
        tmp_dir = Path(tempfile.mkdtemp(dir=temp_dir, prefix=f"mvt_{layer_name}_"))
        generate_pbf(gen, str(tmp_dir), bounds, simplify=True, layer_name=layer_name)
        layer_tmp_dirs.append((layer_name, tmp_dir))

    # Merge PBF tiles: for each (z,x,y) collect bytes from all layers
    # and concatenate (valid MVT protobuf — layers are repeated fields)
    mvt_dir.mkdir(parents=True, exist_ok=True)
    all_tile_keys = set()
    tile_files = {}  # (z,x,y) → [bytes, ...]

    for layer_name, tmp_dir in layer_tmp_dirs:
        for pbf_path in tmp_dir.rglob("*.pbf"):
            rel = pbf_path.relative_to(tmp_dir)
            key = str(rel)
            if key not in tile_files:
                tile_files[key] = []
            tile_files[key].append(pbf_path.read_bytes())
            all_tile_keys.add(key)

    for rel_path, chunks in tile_files.items():
        merged_path = mvt_dir / rel_path
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_bytes(b"".join(chunks))

    # Clean up temp dirs
    for _, tmp_dir in layer_tmp_dirs:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Generate partitioned Parquet (all layers combined, tagged with layer_type)
    gen_all = StreamingTileGenerator2D(
        min_zoom=min_zoom,
        max_zoom=max_zoom,
        buffer=64 / 4096.0,
        temp_dir=temp_dir,
    )
    for layer_name, geojson_str in geojson_layers:
        # Parquet gets layer_type tag since it's a single flat file
        fc = json.loads(geojson_str)
        for feat in fc["features"]:
            feat["properties"]["layer_type"] = layer_name
        gen_all.add_geojson(json.dumps(fc), bounds)

    generate_parquet(gen_all, str(parquet_dir), bounds, simplify=True, partitioned=True)

    # Write TileJSON with proper per-layer field info
    tj = {
        "tilejson": "3.0.0",
        "version": "1.0.0",
        "name": "MuDM Vector Tiles",
        "description": "Multi-layer vector tiles generated by mudm",
        "tiles": ["{z}/{x}/{y}.pbf"],
        "minzoom": min_zoom,
        "maxzoom": max_zoom,
        "bounds": list(bounds),
        "tile_count": len(tile_files),
        "vector_layers": [
            {
                "id": layer_name,
                "fields": fields,
                "minzoom": min_zoom,
                "maxzoom": max_zoom,
                "feature_count": layer_counts[layer_name],
            }
            for layer_name, fields in layer_fields.items()
        ],
    }
    (mvt_dir / "metadata.json").write_text(json.dumps(tj, indent=2))

    return {"layer_counts": layer_counts, "tile_count": len(tile_files)}


def write_metadata(
    output_dir: str | Path,
    name: str,
    um_per_px: float,
    bounds_um: tuple[float, float, float, float],
    raster_info: dict,
    vector_infos: dict[str, dict],
    max_zoom: int,
) -> None:
    """Write metadata.json describing all layers in the tile output."""
    output_dir = Path(output_dir)

    vector_layers = []
    for layer_name, info in vector_infos.items():
        vector_layers.append({
            "id": layer_name,
            "name": layer_name,
            "type": info["type"],
            "color": LAYER_COLORS.get(layer_name, "#ffffff"),
            "min_zoom": info.get("min_zoom", 0),
            "max_zoom": max_zoom,
            "feature_count": info["feature_count"],
        })

    metadata = {
        "name": name,
        "platform": "xenium",
        "um_per_px": um_per_px,
        "bounds_um": list(bounds_um),
        "raster": {
            "path": "raster/{z}/{x}/{y}.png",
            "min_zoom": 0,
            "max_zoom": raster_info["max_zoom"],
            "tile_size": 256,
            "image_size_px": raster_info["image_size_px"],
        },
        "vectors": {
            "path": "vectors/{z}/{x}/{y}.pbf",
            "layers": vector_layers,
        },
        "parquet": {
            "path": "features.parquet",
            "partitioned": True,
        },
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def compute_bounds(parquet_paths: list[Path]) -> tuple[float, float, float, float]:
    """Compute the union bounding box from multiple boundary/transcript parquets."""
    xmin, ymin = float("inf"), float("inf")
    xmax, ymax = float("-inf"), float("-inf")

    for p in parquet_paths:
        df = pl.read_parquet(p)
        if "vertex_x" in df.columns:
            xmin = min(xmin, df["vertex_x"].min())
            xmax = max(xmax, df["vertex_x"].max())
            ymin = min(ymin, df["vertex_y"].min())
            ymax = max(ymax, df["vertex_y"].max())
        elif "x_location" in df.columns:
            xmin = min(xmin, df["x_location"].min())
            xmax = max(xmax, df["x_location"].max())
            ymin = min(ymin, df["y_location"].min())
            ymax = max(ymax, df["y_location"].max())

    return (xmin, ymin, xmax, ymax)


def convert_xenium(
    data_dir: str | Path,
    output_dir: str | Path,
    max_zoom: int = 7,
    temp_dir: str = "/data/tmp",
) -> None:
    """Convert a Xenium dataset directory to muDM tiled format."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read pixel size
    experiment_path = data_dir / "experiment.xenium"
    um_per_px = read_um_per_px(experiment_path)
    print(f"Pixel size: {um_per_px} µm/px")

    # Find parquet files
    cell_boundaries_path = data_dir / "cell_boundaries.parquet"
    nucleus_boundaries_path = data_dir / "nucleus_boundaries.parquet"
    transcripts_path = data_dir / "transcripts.parquet"

    # Compute world bounds (microns)
    parquet_files = [p for p in [cell_boundaries_path, nucleus_boundaries_path, transcripts_path] if p.exists()]
    bounds_um = compute_bounds(parquet_files)
    print(f"World bounds (µm): {bounds_um}")

    # Generate raster tiles first (determines max_zoom and pixel grid).
    # Candidate order mirrors mudm_tools.converters.xenium._find_morphology_image: named DAPI, then the
    # XOA 2.0/3.0 numbered per-channel layout (channel 0 = DAPI), then a single-file morphology image.
    morph_candidates = [
        data_dir / "morphology_focus" / "ch0000_dapi.ome.tif",
        data_dir / "morphology_focus" / "morphology_focus_0000.ome.tif",
        data_dir / "morphology_focus.ome.tif",
    ]
    morph_path = None
    for candidate in morph_candidates:
        if candidate.exists():
            morph_path = candidate
            break

    raster_info = None
    if morph_path is not None:
        print(f"Generating raster tiles from {morph_path.name}...")
        t0 = time.time()
        raster_info = generate_raster_tiles(morph_path, output_dir / "raster")
        max_zoom = raster_info["max_zoom"]
        print(f"  Raster tiles done in {time.time() - t0:.1f}s (max_zoom={max_zoom})")
    else:
        print("Warning: No DAPI image found, skipping raster tiles")
        raster_info = {"max_zoom": max_zoom, "image_size_px": [0, 0]}

    # Vector tiles must align with the Leaflet tile grid.
    # The raster tiles are in pixel space; the quadtree must use the same grid.
    # At max_zoom, the Leaflet tile grid is 256 * 2^max_zoom pixels square.
    # Convert vector coordinates from microns to pixels (÷ um_per_px) and set
    # the quadtree world bounds to the full Leaflet grid so tile (z,x,y) maps
    # to the same spatial region for both raster and vector layers.
    # Vector tiles get one extra zoom level beyond raster for finer detail.
    # Grid size must match the RASTER zoom level so tile (z,x,y) aligns.
    vector_max_zoom = max_zoom + 1
    grid_size = 256.0 * (2 ** max_zoom)
    coord_scale = 1.0 / um_per_px  # microns → pixels
    tile_bounds = (0.0, 0.0, grid_size, grid_size)
    print(f"Tile grid: {int(grid_size)}×{int(grid_size)} px, vector zoom 0-{vector_max_zoom} (raster 0-{max_zoom})")

    # Single-pass pipeline: one generator per layer → clip once → emit both MVT and Parquet.
    # Points use add_parquet_points() (Rust-native, no JSON). Polygons use add_geojson().
    import shutil
    import tempfile

    point_min_zoom = max(0, vector_max_zoom - 3)  # points only at detailed zooms

    source_layers = [
        ("cells", cell_boundaries_path, "polygon", "cell_id"),
        ("nuclei", nucleus_boundaries_path, "polygon", "cell_id"),
        ("transcripts", transcripts_path, "point", None),
    ]

    layer_counts = {}
    layer_types = {}
    layer_fields = {}
    layer_min_zooms = {}
    layer_tmp_dirs = []   # (name, mvt_tmp, pq_tmp)

    for layer_name, parquet_path, geom_type, id_col in source_layers:
        if not parquet_path.exists():
            print(f"Skipping {layer_name}: {parquet_path.name} not found")
            continue

        layer_types[layer_name] = geom_type
        layer_min = point_min_zoom if geom_type == "point" else 0
        layer_min_zooms[layer_name] = layer_min

        # One generator per layer — fragments serve both MVT and Parquet
        gen = StreamingTileGenerator2D(
            min_zoom=layer_min, max_zoom=vector_max_zoom,
            buffer=64 / 4096.0, temp_dir=temp_dir,
        )

        # Ingest features
        print(f"Ingesting {layer_name} (zoom {layer_min}-{vector_max_zoom})...", end=" ", flush=True)
        t0 = time.time()

        if geom_type == "point":
            # Rust-native Parquet reader — no JSON intermediary
            count = gen.add_parquet_points(
                str(parquet_path),
                "x_location", "y_location",
                "feature_name", "gene_name",
                layer_name,
                tile_bounds,
                coord_scale,
            )
            layer_fields[layer_name] = {"gene_name": "String"}
        else:
            # Rust-native Parquet polygon reader — groups by cell_id
            count = gen.add_parquet_polygons(
                str(parquet_path),
                id_col, "vertex_x", "vertex_y",
                layer_name,
                tile_bounds,
                coord_scale,
            )
            layer_fields[layer_name] = {"cell_id": "String"}

        layer_counts[layer_name] = count
        print(f"{count:,} features ({time.time() - t0:.1f}s)", flush=True)

        # Encode PBF then Parquet (sequential — both read same fragments)
        mvt_tmp = Path(tempfile.mkdtemp(dir=temp_dir, prefix=f"mvt_{layer_name}_"))
        pq_tmp = Path(tempfile.mkdtemp(dir=temp_dir, prefix=f"pq_{layer_name}_"))

        print(f"  Encoding PBF...", end=" ", flush=True)
        t0 = time.time()
        generate_pbf(gen, str(mvt_tmp), tile_bounds, simplify=True, layer_name=layer_name)
        print(f"done ({time.time() - t0:.1f}s)", flush=True)

        print(f"  Encoding Parquet...", end=" ", flush=True)
        t0 = time.time()
        pq_rows = gen.generate_parquet_native(str(pq_tmp), tile_bounds, simplify=True)
        print(f"{pq_rows:,} rows ({time.time() - t0:.1f}s)", flush=True)

        layer_tmp_dirs.append((layer_name, mvt_tmp, pq_tmp))

    # Merge MVT tiles from all layers (protobuf concatenation)
    print("Merging MVT layers...", end=" ", flush=True)
    t0 = time.time()
    mvt_dir = output_dir / "vectors"
    mvt_dir.mkdir(parents=True, exist_ok=True)
    tile_files = {}
    for layer_name, mvt_tmp, _ in layer_tmp_dirs:
        for pbf_path in mvt_tmp.rglob("*.pbf"):
            key = str(pbf_path.relative_to(mvt_tmp))
            if key not in tile_files:
                tile_files[key] = []
            tile_files[key].append(pbf_path.read_bytes())
    for rel_path, chunks in tile_files.items():
        merged_path = mvt_dir / rel_path
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        merged_path.write_bytes(b"".join(chunks))
    print(f"{len(tile_files)} tiles ({time.time() - t0:.1f}s)", flush=True)

    # Merge Parquet partitions from all layers
    print("Merging Parquet partitions...", end=" ", flush=True)
    t0 = time.time()
    parquet_dir = output_dir / "features.parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    for layer_name, _, pq_tmp in layer_tmp_dirs:
        for zoom_dir in sorted(pq_tmp.glob("zoom=*")):
            target = parquet_dir / zoom_dir.name
            target.mkdir(exist_ok=True)
            for pq_file in zoom_dir.glob("*.parquet"):
                dest = target / f"{layer_name}_{pq_file.name}"
                shutil.move(str(pq_file), str(dest))
    print(f"done ({time.time() - t0:.1f}s)", flush=True)

    # Clean up temp dirs
    for _, mvt_tmp, pq_tmp in layer_tmp_dirs:
        shutil.rmtree(mvt_tmp, ignore_errors=True)
        shutil.rmtree(pq_tmp, ignore_errors=True)

    # Write TileJSON
    tj = {
        "tilejson": "3.0.0", "version": "1.0.0",
        "name": "MuDM Vector Tiles",
        "description": "Multi-layer vector tiles generated by mudm",
        "tiles": ["{z}/{x}/{y}.pbf"],
        "minzoom": 0, "maxzoom": vector_max_zoom,
        "bounds": list(tile_bounds),
        "tile_count": len(tile_files),
        "vector_layers": [
            {"id": name, "fields": fields,
             "minzoom": layer_min_zooms.get(name, 0), "maxzoom": vector_max_zoom,
             "feature_count": layer_counts[name]}
            for name, fields in layer_fields.items()
        ],
    }
    (mvt_dir / "metadata.json").write_text(json.dumps(tj, indent=2))

    # Build vector_infos for metadata
    vector_infos = {}
    for name, count in layer_counts.items():
        vector_infos[name] = {
            "feature_count": count,
            "tile_count": len(tile_files),
            "type": layer_types[name],
            "min_zoom": layer_min_zooms.get(name, 0),
        }

    # Write metadata
    dataset_name = data_dir.name
    write_metadata(output_dir, dataset_name, um_per_px, bounds_um, raster_info, vector_infos, vector_max_zoom)
    print(f"Done. Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Xenium data to muDM tiles")
    parser.add_argument("--data-dir", required=True, help="Path to Xenium output directory")
    parser.add_argument("--output-dir", required=True, help="Path for tile output")
    parser.add_argument("--max-zoom", type=int, default=7, help="Max zoom level (default: 7)")
    parser.add_argument("--temp-dir", default="/data/tmp", help="Temp directory for fragments")
    args = parser.parse_args()

    convert_xenium(args.data_dir, args.output_dir, args.max_zoom, args.temp_dir)


if __name__ == "__main__":
    main()
