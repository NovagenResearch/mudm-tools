"""Xenium spatial transcriptomics → muDM tiled format.

Converts 10x Genomics Xenium output (boundaries, transcripts, DAPI image)
into MVT vector tiles, partitioned Parquet, and a PNG raster tile pyramid.

Source files:
    cell_boundaries.parquet     — polygon vertices (cell_id, vertex_x, vertex_y)
    nucleus_boundaries.parquet  — polygon vertices
    transcripts.parquet         — point detections (x_location, y_location, feature_name)
    morphology_focus.ome.tif    — DAPI fluorescence image
    experiment.xenium           — metadata (pixel_size)
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import register


@register("xenium")
class XeniumConverter:
    """Convert 10x Genomics Xenium data to muDM tiled format."""

    # Default layer colors for the viewer
    LAYER_COLORS = {
        "cells": "#00ffff",
        "nuclei": "#00ff00",
        "transcripts": "#ff4444",
    }

    def convert(
        self,
        input_dir: str,
        output_dir: str,
        config: dict[str, Any],
    ) -> dict:
        """Run the full Xenium → muDM conversion.

        Config keys:
            temp_dir (str): Temp directory for fragments. Default: system temp.
            max_zoom (int): Override max zoom level. Default: derived from image.
            point_zoom_offset (int): Transcripts start at max_zoom - offset. Default: 3.
            id_column (str): Boundary ID column name. Default: "cell_id".
            skip_raster (bool): Skip raster tile generation. Default: False.
        """
        from mudm_tools._rs import StreamingTileGenerator2D
        from mudm_tools.tiling2d import generate_pbf

        data_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = config.get("temp_dir", tempfile.gettempdir())
        max_zoom_override = config.get("max_zoom")
        point_zoom_offset = config.get("point_zoom_offset", 3)
        id_column = config.get("id_column", "cell_id")
        skip_raster = config.get("skip_raster", False)

        timings = {}
        t_start = time.time()

        # Read pixel size
        experiment_path = data_dir / "experiment.xenium"
        um_per_px = self._read_um_per_px(experiment_path)
        print(f"Pixel size: {um_per_px} µm/px", flush=True)

        # Raster tiles
        raster_info, max_zoom = self._generate_raster(
            data_dir, out_dir, skip_raster, max_zoom_override
        )
        timings["raster"] = time.time() - t_start

        # Tile grid alignment
        vector_max_zoom = max_zoom + 1
        grid_size = 256.0 * (2 ** max_zoom)
        coord_scale = 1.0 / um_per_px
        tile_bounds = (0.0, 0.0, grid_size, grid_size)
        point_min_zoom = max(0, vector_max_zoom - point_zoom_offset)

        print(f"Tile grid: {int(grid_size)}×{int(grid_size)} px, "
              f"vector zoom 0-{vector_max_zoom} (raster 0-{max_zoom})", flush=True)

        # Define layers
        layers = [
            ("cells", data_dir / "cell_boundaries.parquet", "polygon", id_column),
            ("nuclei", data_dir / "nucleus_boundaries.parquet", "polygon", id_column),
            ("transcripts", data_dir / "transcripts.parquet", "point", None),
        ]

        layer_counts = {}
        layer_fields = {}
        layer_min_zooms = {}
        layer_tmp_dirs = []

        for layer_name, parquet_path, geom_type, id_col in layers:
            if not parquet_path.exists():
                print(f"Skipping {layer_name}: {parquet_path.name} not found", flush=True)
                continue

            layer_min = point_min_zoom if geom_type == "point" else 0
            layer_min_zooms[layer_name] = layer_min

            gen = StreamingTileGenerator2D(
                min_zoom=layer_min, max_zoom=vector_max_zoom,
                buffer=64 / 4096.0, temp_dir=temp_dir,
            )

            print(f"Ingesting {layer_name} (zoom {layer_min}-{vector_max_zoom})...",
                  end=" ", flush=True)
            t0 = time.time()

            if geom_type == "point":
                count = gen.add_parquet_points(
                    str(parquet_path),
                    "x_location", "y_location",
                    "feature_name", "gene_name",
                    layer_name, tile_bounds, coord_scale,
                )
                layer_fields[layer_name] = {"gene_name": "String"}
            else:
                count = gen.add_parquet_polygons(
                    str(parquet_path),
                    id_col, "vertex_x", "vertex_y",
                    layer_name, tile_bounds, coord_scale,
                )
                layer_fields[layer_name] = {"cell_id": "String"}

            layer_counts[layer_name] = count
            t_ingest = time.time() - t0
            print(f"{count:,} features ({t_ingest:.1f}s)", flush=True)

            # Encode PBF
            print(f"  Encoding PBF...", end=" ", flush=True)
            t0 = time.time()
            mvt_tmp = Path(tempfile.mkdtemp(dir=temp_dir, prefix=f"mvt_{layer_name}_"))
            generate_pbf(gen, str(mvt_tmp), tile_bounds, simplify=True, layer_name=layer_name)
            t_pbf = time.time() - t0
            print(f"done ({t_pbf:.1f}s)", flush=True)

            # Encode Parquet
            print(f"  Encoding Parquet...", end=" ", flush=True)
            t0 = time.time()
            pq_tmp = Path(tempfile.mkdtemp(dir=temp_dir, prefix=f"pq_{layer_name}_"))
            pq_rows = gen.generate_parquet_native(str(pq_tmp), tile_bounds, simplify=True)
            t_pq = time.time() - t0
            print(f"{pq_rows:,} rows ({t_pq:.1f}s)", flush=True)

            layer_tmp_dirs.append((layer_name, mvt_tmp, pq_tmp))
            timings[layer_name] = {"ingest": t_ingest, "pbf": t_pbf, "parquet": t_pq}

        # Merge MVT
        print("Merging MVT layers...", end=" ", flush=True)
        t0 = time.time()
        mvt_dir = out_dir / "vectors"
        mvt_dir.mkdir(parents=True, exist_ok=True)
        tile_files = {}
        for layer_name, mvt_tmp, _ in layer_tmp_dirs:
            for pbf_path in mvt_tmp.rglob("*.pbf"):
                key = str(pbf_path.relative_to(mvt_tmp))
                tile_files.setdefault(key, []).append(pbf_path.read_bytes())
        for rel_path, chunks in tile_files.items():
            merged_path = mvt_dir / rel_path
            merged_path.parent.mkdir(parents=True, exist_ok=True)
            merged_path.write_bytes(b"".join(chunks))
        print(f"{len(tile_files)} tiles ({time.time() - t0:.1f}s)", flush=True)

        # Merge Parquet
        print("Merging Parquet partitions...", end=" ", flush=True)
        t0 = time.time()
        parquet_dir = out_dir / "features.parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        for layer_name, _, pq_tmp in layer_tmp_dirs:
            for zoom_dir in sorted(pq_tmp.glob("zoom=*")):
                target = parquet_dir / zoom_dir.name
                target.mkdir(exist_ok=True)
                for pq_file in zoom_dir.glob("*.parquet"):
                    dest = target / f"{layer_name}_{pq_file.name}"
                    shutil.move(str(pq_file), str(dest))
        print(f"done ({time.time() - t0:.1f}s)", flush=True)

        # Clean up temp
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

        # Write metadata.json
        # Compute bounds_um from parquet files
        import polars as pl
        bounds_um = self._compute_bounds_um(data_dir, layers)

        vector_layers = []
        for name, fields in layer_fields.items():
            vector_layers.append({
                "id": name, "name": name,
                "type": "point" if "gene_name" in fields else "polygon",
                "color": self.LAYER_COLORS.get(name, "#ffffff"),
                "min_zoom": layer_min_zooms.get(name, 0),
                "max_zoom": vector_max_zoom,
                "feature_count": layer_counts[name],
            })

        metadata = {
            "name": data_dir.name,
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
            "vectors": {"path": "vectors/{z}/{x}/{y}.pbf", "layers": vector_layers},
            "parquet": {"path": "features.parquet", "partitioned": True},
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Write gene_list.json for the viewer gene filter
        transcripts_path = data_dir / "transcripts.parquet"
        if transcripts_path.exists():
            df = pl.read_parquet(transcripts_path, columns=["feature_name"])
            if df.schema["feature_name"] == pl.Binary:
                df = df.with_columns(pl.col("feature_name").cast(pl.Utf8))
            genes = sorted(df["feature_name"].unique().to_list())
            (out_dir / "gene_list.json").write_text(json.dumps(genes))
            print(f"Wrote gene_list.json ({len(genes)} genes)", flush=True)

        total_time = time.time() - t_start
        print(f"Done. Output: {out_dir} ({total_time:.0f}s)", flush=True)

        return {
            "total_time": total_time,
            "timings": timings,
            "layer_counts": layer_counts,
            "tile_count": len(tile_files),
        }

    def _read_um_per_px(self, experiment_path: Path) -> float:
        with open(experiment_path) as f:
            return float(json.load(f)["pixel_size"])

    def _generate_raster(self, data_dir, out_dir, skip_raster, max_zoom_override):
        """Generate raster tile pyramid from DAPI image."""
        import tifffile
        from PIL import Image

        morph_candidates = [
            data_dir / "morphology_focus" / "ch0000_dapi.ome.tif",
            data_dir / "morphology_focus.ome.tif",
        ]
        morph_path = next((p for p in morph_candidates if p.exists()), None)

        raster_dir = out_dir / "raster"
        if morph_path is None:
            max_zoom = max_zoom_override or 7
            return {"max_zoom": max_zoom, "image_size_px": [0, 0]}, max_zoom

        if skip_raster and raster_dir.exists():
            # Infer max_zoom from existing tiles
            zooms = [int(d.name) for d in raster_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            max_zoom = max(zooms) if zooms else 7
            with tifffile.TiffFile(str(morph_path)) as tf:
                raw = np.squeeze(tf.pages[0].asarray())
            h, w = raw.shape[:2]
            print(f"Raster tiles exist, skipping (max_zoom={max_zoom})", flush=True)
            return {"max_zoom": max_zoom, "image_size_px": [w, h]}, max_zoom

        print(f"Generating raster tiles from {morph_path.name}...", end=" ", flush=True)
        t0 = time.time()

        with tifffile.TiffFile(str(morph_path)) as tf:
            raw = np.squeeze(tf.pages[0].asarray())
        if raw.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {raw.shape}")

        raw = raw.astype(np.float32)
        p_lo, p_hi = np.percentile(raw, [1, 99])
        img_uint8 = np.clip((raw - p_lo) / (p_hi - p_lo + 1e-6), 0, 1)
        img_uint8 = (img_uint8 * 255).astype(np.uint8)

        h, w = img_uint8.shape
        max_zoom = max_zoom_override or max(0, math.ceil(math.log2(max(h, w) / 256)))
        tile_size = 256

        current = img_uint8
        for z in range(max_zoom, -1, -1):
            ch, cw = current.shape
            nx, ny = math.ceil(cw / tile_size), math.ceil(ch / tile_size)
            for ty in range(ny):
                for tx in range(nx):
                    y0, x0 = ty * tile_size, tx * tile_size
                    tile_data = current[y0:min(y0+tile_size, ch), x0:min(x0+tile_size, cw)]
                    if tile_data.shape != (tile_size, tile_size):
                        padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
                        padded[:tile_data.shape[0], :tile_data.shape[1]] = tile_data
                        tile_data = padded
                    tile_path = raster_dir / str(z) / str(tx) / f"{ty}.png"
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(tile_data, mode="L").save(tile_path)
            if z > 0:
                pil_img = Image.fromarray(current, mode="L")
                current = np.array(pil_img.resize((max(1, cw//2), max(1, ch//2)), Image.LANCZOS))

        print(f"done ({time.time()-t0:.1f}s, max_zoom={max_zoom})", flush=True)
        return {"max_zoom": max_zoom, "image_size_px": [w, h]}, max_zoom

    def _compute_bounds_um(self, data_dir, layers):
        import polars as pl
        xmin, ymin = float("inf"), float("inf")
        xmax, ymax = float("-inf"), float("-inf")
        for _, path, _, _ in layers:
            if not path.exists():
                continue
            df = pl.read_parquet(path)
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
