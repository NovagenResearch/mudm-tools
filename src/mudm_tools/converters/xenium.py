"""Xenium spatial transcriptomics → muDM tiled format.

Converts 10x Genomics Xenium output (boundaries, transcripts, DAPI image)
into MVT vector tiles, partitioned Parquet, and a PNG raster tile pyramid.

Source files:
    cell_boundaries.parquet     — polygon vertices (cell_id, vertex_x, vertex_y)
    nucleus_boundaries.parquet  — polygon vertices
    transcripts.parquet         — point detections (x_location, y_location, feature_name)
    morphology_focus.ome.tif    — DAPI fluorescence image
    experiment.xenium           — metadata (pixel_size)
    cells.parquet               — per-cell summary (cell_id, x_centroid, y_centroid, …)
    cell_feature_matrix/        — sparse expression matrix (cells × features)
        ├── matrix.mtx.gz
        ├── barcodes.tsv.gz
        └── features.tsv.gz
    analysis/clustering/.../clusters.csv  — graph clustering assignments
"""

from __future__ import annotations

import csv
import gzip
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

        timings: dict[str, float | dict[str, float]] = {}
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
        grid_size = 256.0 * (2**max_zoom)
        coord_scale = 1.0 / um_per_px
        tile_bounds = (0.0, 0.0, grid_size, grid_size)
        point_min_zoom = max(0, vector_max_zoom - point_zoom_offset)

        print(
            f"Tile grid: {int(grid_size)}×{int(grid_size)} px, "
            f"vector zoom 0-{vector_max_zoom} (raster 0-{max_zoom})",
            flush=True,
        )

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
                min_zoom=layer_min,
                max_zoom=vector_max_zoom,
                buffer=64 / 4096.0,
                temp_dir=temp_dir,
            )

            print(
                f"Ingesting {layer_name} (zoom {layer_min}-{vector_max_zoom})...",
                end=" ",
                flush=True,
            )
            t0 = time.time()

            if geom_type == "point":
                count = gen.add_parquet_points(
                    str(parquet_path),
                    "x_location",
                    "y_location",
                    "feature_name",
                    "gene_name",
                    layer_name,
                    tile_bounds,
                    coord_scale,
                )
                layer_fields[layer_name] = {"gene_name": "String"}
            else:
                count = gen.add_parquet_polygons(
                    str(parquet_path),
                    id_col,
                    "vertex_x",
                    "vertex_y",
                    layer_name,
                    tile_bounds,
                    coord_scale,
                )
                layer_fields[layer_name] = {"cell_id": "String"}

            layer_counts[layer_name] = count
            t_ingest = time.time() - t0
            print(f"{count:,} features ({t_ingest:.1f}s)", flush=True)

            # Encode PBF
            print("  Encoding PBF...", end=" ", flush=True)
            t0 = time.time()
            mvt_tmp = Path(tempfile.mkdtemp(dir=temp_dir, prefix=f"mvt_{layer_name}_"))
            generate_pbf(gen, str(mvt_tmp), tile_bounds, simplify=True, layer_name=layer_name)
            t_pbf = time.time() - t0
            print(f"done ({t_pbf:.1f}s)", flush=True)

            # Encode Parquet
            print("  Encoding Parquet...", end=" ", flush=True)
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
        tile_files: dict[str, list[bytes]] = {}
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
            "tilejson": "3.0.0",
            "version": "1.0.0",
            "name": "MuDM Vector Tiles",
            "description": "Multi-layer vector tiles generated by mudm",
            "tiles": ["{z}/{x}/{y}.pbf"],
            "minzoom": 0,
            "maxzoom": vector_max_zoom,
            "bounds": list(tile_bounds),
            "tile_count": len(tile_files),
            "vector_layers": [
                {
                    "id": name,
                    "fields": fields,
                    "minzoom": layer_min_zooms.get(name, 0),
                    "maxzoom": vector_max_zoom,
                    "feature_count": layer_counts[name],
                }
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
            vector_layers.append(
                {
                    "id": name,
                    "name": name,
                    "type": "point" if "gene_name" in fields else "polygon",
                    "color": self.LAYER_COLORS.get(name, "#ffffff"),
                    "min_zoom": layer_min_zooms.get(name, 0),
                    "max_zoom": vector_max_zoom,
                    "feature_count": layer_counts[name],
                }
            )

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
                    tile_data = current[y0 : min(y0 + tile_size, ch), x0 : min(x0 + tile_size, cw)]
                    if tile_data.shape != (tile_size, tile_size):
                        padded = np.zeros((tile_size, tile_size), dtype=np.uint8)
                        padded[: tile_data.shape[0], : tile_data.shape[1]] = tile_data
                        tile_data = padded
                    tile_path = raster_dir / str(z) / str(tx) / f"{ty}.png"
                    tile_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(tile_data, mode="L").save(tile_path)
            if z > 0:
                pil_img = Image.fromarray(current, mode="L")
                current = np.array(
                    pil_img.resize((max(1, cw // 2), max(1, ch // 2)), Image.LANCZOS)
                )

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


# ---------------------------------------------------------------------------
# Function-level entry point: xenium_to_mudm
# ---------------------------------------------------------------------------


def xenium_to_mudm(
    cell_boundaries_path: Path | str,
    cell_feature_matrix_path: Path | str,
    cells_path: Path | str | None = None,
    cell_type_annotations: Path | str | None = None,
    max_cells: int | None = None,
):
    """Build a muDM FeatureCollection from a Xenium output bundle.

    Each cell becomes a ``MuDMFeature`` whose geometry is the closed polygon
    produced by joining ``cell_boundaries.parquet`` rows on ``cell_id``. The
    per-cell expression vector from ``cell_feature_matrix`` is stored under
    ``properties["expression"]`` as a **JSON-encoded string** of the list
    of integer counts (one entry per feature row in the matrix). This
    encoding is required because the downstream Parquet ``tags`` column is
    ``map<utf8, utf8>`` and the Rust ingest path silently drops array-valued
    properties; storing the list as a JSON string lets it round-trip
    through ``StreamingTileGenerator2D.add_geojson`` -> ``generate_parquet``
    untouched. Consumers read it back via ``json.loads(props["expression"])``.

    Args:
        cell_boundaries_path: Path to ``cell_boundaries.parquet`` (vertex-
            per-row layout: ``cell_id``, ``vertex_x``, ``vertex_y``).
        cell_feature_matrix_path: Path to either the directory containing
            ``matrix.mtx.gz`` / ``barcodes.tsv.gz`` / ``features.tsv.gz``
            (Xenium ``cell_feature_matrix/``) or directly to a ``.zarr.zip``
            archive (the latter requires the optional ``zarr`` extra).
        cells_path: Optional path to ``cells.parquet`` (cell summary with
            centroids, total counts, areas). Used for cell-level metadata
            attachment when present.
        cell_type_annotations: Optional path to a per-cell-cluster CSV (e.g.
            ``analysis/clustering/gene_expression_graphclust/clusters.csv``).
            Each row is ``Barcode, Cluster``. Attaches a ``cluster_id``
            property and (when curated mappings are provided) Cell Ontology
            URIs in the future. The Xenium ``Rep1`` "preview" dataset has
            no curated cell-type → ontology mapping; pass ``None`` to skip.
        max_cells: Truncate to the first N cells (in ``cell_id`` order) for
            smoke-test ingestion. Use ``None`` for the full dataset.

    Returns:
        ``MuDMFeatureCollection`` with one ``Polygon`` feature per cell.
        Coordinates are in physical micrometres (Xenium native).
    """
    # Lazy import muDM model classes — keeps converters/__init__ light.
    from mudm.model import MuDMFeature, MuDMFeatureCollection
    from geojson_pydantic import Polygon
    import pyarrow.parquet as pq

    cell_boundaries_path = Path(cell_boundaries_path)
    cell_feature_matrix_path = Path(cell_feature_matrix_path)
    cells_path = Path(cells_path) if cells_path is not None else None
    cell_type_annotations = (
        Path(cell_type_annotations) if cell_type_annotations is not None else None
    )

    # 1) Boundary polygons -----------------------------------------------------
    table = pq.read_table(cell_boundaries_path)
    cols = table.column_names
    if "cell_id" not in cols:
        raise ValueError(f"cell_boundaries.parquet missing 'cell_id' column (got: {cols})")
    if "vertex_x" not in cols or "vertex_y" not in cols:
        raise ValueError(f"cell_boundaries.parquet missing 'vertex_x'/'vertex_y' (got: {cols})")

    cell_id_arr = table.column("cell_id").to_numpy(zero_copy_only=False)
    vx_arr = table.column("vertex_x").to_numpy(zero_copy_only=False)
    vy_arr = table.column("vertex_y").to_numpy(zero_copy_only=False)

    # Cap to first N cells using a stable, sorted cell_id ordering.
    distinct_ids = np.unique(cell_id_arr)
    distinct_ids.sort()
    if max_cells is not None:
        distinct_ids = distinct_ids[:max_cells]
    selected = set(distinct_ids.tolist())

    # 2) Optional per-cell summary (centroid, total_counts, areas) ------------
    cell_summary: dict[Any, dict[str, Any]] = {}
    if cells_path is not None and cells_path.exists():
        cells_table = pq.read_table(cells_path)
        cell_summary = _extract_cell_summary(cells_table, selected)

    # 3) Expression matrix -----------------------------------------------------
    expression_by_cell, gene_panel = _load_xenium_expression(
        cell_feature_matrix_path, only_cells=selected
    )

    # 4) Optional cluster annotations -----------------------------------------
    cluster_by_cell: dict[Any, int] = {}
    if cell_type_annotations is not None and cell_type_annotations.exists():
        cluster_by_cell = _load_xenium_clusters(cell_type_annotations, selected)

    # 5) Build features --------------------------------------------------------
    # Group vertex rows by cell_id while preserving 10x's vertex order.
    # mask = membership of selected ids (vectorised) -> stable group-by.
    mask = np.isin(cell_id_arr, distinct_ids)
    sel_ids = cell_id_arr[mask]
    sel_vx = vx_arr[mask]
    sel_vy = vy_arr[mask]
    # Build groups indexed by first-occurrence order
    groups: dict[Any, list[tuple[float, float]]] = {}
    for cid, x, y in zip(sel_ids.tolist(), sel_vx.tolist(), sel_vy.tolist()):
        groups.setdefault(cid, []).append((float(x), float(y)))

    features: list[MuDMFeature] = []
    for cell_id, ring in groups.items():
        if not ring:
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        if len(ring) < 4:
            # GeoJSON polygons need >=4 positions (3 distinct + closure)
            continue

        # Expression is JSON-encoded as a string so it survives the
        # tags: map<utf8, utf8> Parquet schema. Decode with json.loads.
        expr_list = expression_by_cell.get(cell_id, [])
        props: dict[str, Any] = {
            "cell_id": str(cell_id),
            "expression": json.dumps(expr_list, separators=(",", ":")),
        }
        summary = cell_summary.get(cell_id)
        if summary is not None:
            props.update({k: v for k, v in summary.items() if v is not None})
        if cell_id in cluster_by_cell:
            props["cluster_id"] = cluster_by_cell[cell_id]

        feat = MuDMFeature(
            type="Feature",
            geometry=Polygon(type="Polygon", coordinates=[ring]),  # type: ignore[list-item]  # geojson-pydantic accepts coord lists
            properties=props,
        )
        features.append(feat)

    fc = MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={
            "platform": "xenium",
            "crs": {"type": "physical", "units": "micrometers"},
            "gene_panel_dimension": len(gene_panel),
            "gene_panel": gene_panel,
        },
    )
    return fc


def _extract_cell_summary(
    table,
    only_cells: set[Any],
) -> dict[Any, dict[str, Any]]:
    """Filter a ``cells.parquet`` pyarrow Table to the selected cell_ids.

    Returns a dict from cell_id to a per-cell metadata dict. We pull only
    columns the muDM consumer cares about (centroids, counts, areas) when
    they are present.
    """
    cols_of_interest = (
        "x_centroid",
        "y_centroid",
        "transcript_counts",
        "total_counts",
        "cell_area",
        "nucleus_area",
    )
    available = [c for c in cols_of_interest if c in table.column_names]
    if "cell_id" not in table.column_names:
        return {}
    cid = table.column("cell_id").to_numpy(zero_copy_only=False)
    mask = np.isin(cid, list(only_cells))
    sub = table.select(["cell_id", *available]).filter(mask)

    out: dict[Any, dict[str, Any]] = {}
    rows = sub.to_pylist()
    for row in rows:
        out[row["cell_id"]] = {k: row[k] for k in available if k in row}
    return out


def _load_xenium_expression(
    matrix_path: Path,
    only_cells: set[Any] | None = None,
) -> tuple[dict[Any, list[int]], list[str]]:
    """Read a Xenium ``cell_feature_matrix`` and return per-cell vectors.

    Supports the directory layout (``matrix.mtx.gz`` + ``barcodes.tsv.gz`` +
    ``features.tsv.gz``). The ``.zarr.zip`` form is accepted as a path but
    only when the optional ``zarr`` dependency is installed; otherwise
    ``ValueError`` is raised so callers can fall back.

    Returns:
        (expression_by_cell, gene_panel) where:
          * expression_by_cell[cell_id] = list[int] of length len(gene_panel)
          * gene_panel is the ordered list of feature names
            (``ENSG…`` symbol or codeword id).
    """
    if matrix_path.is_dir():
        return _load_xenium_expression_mtx(matrix_path, only_cells)
    if matrix_path.suffix == ".zip" or matrix_path.name.endswith(".zarr.zip"):
        # Defer to zarr if available; otherwise look for a sibling mtx dir.
        sibling = matrix_path.parent / "cell_feature_matrix"
        if sibling.is_dir():
            return _load_xenium_expression_mtx(sibling, only_cells)
        raise ValueError(
            f"Cannot read {matrix_path}: zarr-zip is not yet supported by "
            "xenium_to_mudm; pass cell_feature_matrix/ directory instead."
        )
    raise FileNotFoundError(f"cell_feature_matrix path not understood: {matrix_path}")


def _load_xenium_expression_mtx(
    matrix_dir: Path,
    only_cells: set[Any] | None,
) -> tuple[dict[Any, list[int]], list[str]]:
    """Load matrix from an mtx.gz triplet (Xenium ``cell_feature_matrix/``).

    Matrix is stored in MatrixMarket convention (features × cells), 1-based
    row/column indices. Barcodes correspond to integer ``cell_id`` values.
    """
    import scipy.io as sio

    mtx_path = matrix_dir / "matrix.mtx.gz"
    bc_path = matrix_dir / "barcodes.tsv.gz"
    feat_path = matrix_dir / "features.tsv.gz"
    for p in (mtx_path, bc_path, feat_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")

    with gzip.open(bc_path, "rt") as f:
        barcodes = [line.strip() for line in f if line.strip()]
    with gzip.open(feat_path, "rt") as f:
        # 10x features.tsv.gz: id\tname\ttype
        gene_panel = [line.split("\t")[1].strip() for line in f if line.strip()]

    with gzip.open(mtx_path, "rb") as f:
        m = sio.mmread(f).tocsc()  # (features × cells), efficient column slicing
    # Defensive: if shape happens to be transposed (cells × features), re-orient.
    if m.shape[0] == len(barcodes) and m.shape[1] == len(gene_panel):
        m = m.T.tocsc()
    if m.shape != (len(gene_panel), len(barcodes)):
        raise ValueError(
            f"matrix shape {m.shape} doesn't match "
            f"({len(gene_panel)} features × {len(barcodes)} cells)"
        )

    # Map barcode (string) to its column index. Xenium barcodes are integer
    # strings ("1", "2", …) that correspond directly to ``cell_id``.
    expression_by_cell: dict[Any, list[int]] = {}
    barcode_lookup = {bc: idx for idx, bc in enumerate(barcodes)}
    for cell_id in only_cells or set(barcode_lookup.keys()):
        bc = str(cell_id)
        col_idx = barcode_lookup.get(bc)
        if col_idx is None:
            continue
        col = m.getcol(col_idx).toarray().ravel().astype(np.int64)
        expression_by_cell[cell_id] = [int(v) for v in col]
    return expression_by_cell, gene_panel


def _load_xenium_clusters(
    clusters_csv: Path,
    only_cells: set[Any] | None,
) -> dict[Any, int]:
    """Load 10x graph-clustering CSV (Barcode, Cluster) into a dict."""
    out: dict[Any, int] = {}
    with open(clusters_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bc = row.get("Barcode") or row.get("barcode")
            cl = row.get("Cluster") or row.get("cluster")
            if bc is None or cl is None:
                continue
            try:
                cell_id_int = int(bc)
            except ValueError:
                cell_id_int = bc  # type: ignore[assignment]
            if only_cells is not None and cell_id_int not in only_cells:
                continue
            try:
                out[cell_id_int] = int(cl)
            except ValueError:
                continue
    return out
