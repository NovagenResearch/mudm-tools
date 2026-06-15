"""GeoJSON → muDM tiled 2D format.

Converts GeoJSON FeatureCollection files into quadtree-tiled MVT vector tiles
and partitioned Parquet.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from . import register


@register("geojson")
class GeoJsonConverter:
    """Convert GeoJSON files to muDM tiled 2D format."""

    def convert(
        self,
        input_dir: str,
        output_dir: str,
        config: dict[str, Any],
    ) -> dict:
        """Convert GeoJSON to tiled output.

        input_dir can be a single .geojson/.json file or a directory.

        Config keys:
            temp_dir (str): Temp directory for fragments.
            max_zoom (int): Max zoom level. Default: 7.
            min_zoom (int): Min zoom level. Default: 0.
            bounds (tuple): World bounds (xmin,ymin,xmax,ymax).
                If not provided, computed from features.
            layer_name (str): MVT layer name. Default: "features".
            glob (str): Glob pattern if input_dir is a directory. Default: "*.geojson".
        """
        from mudm_tools._rs import StreamingTileGenerator2D
        from mudm_tools.tiling2d import generate_pbf

        input_path = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = config.get("temp_dir", tempfile.gettempdir())
        max_zoom = config.get("max_zoom", 7)
        min_zoom = config.get("min_zoom", 0)
        bounds = config.get("bounds")
        layer_name = config.get("layer_name", "features")
        # Polygon simplification (Douglas-Peucker at coarse zooms) is great for large shapes but COLLAPSES
        # small features (e.g. ~20px segmented cells) into slivers. Let callers disable it for cell/nucleus
        # datasets. Default True preserves prior behavior.
        simplify = config.get("simplify", True)

        t_start = time.time()

        # Load GeoJSON
        if input_path.is_file():
            geojson_files = [input_path]
        else:
            glob_pattern = config.get("glob", "*.geojson")
            geojson_files = sorted(input_path.glob(glob_pattern))

        if not geojson_files:
            raise FileNotFoundError(f"No GeoJSON files found at {input_path}")

        # Compute bounds if not provided
        if bounds is None:
            bounds = self._compute_bounds(geojson_files)

        gen = StreamingTileGenerator2D(
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            buffer=64 / 4096.0,
            temp_dir=temp_dir,
        )

        # Ingest
        print(f"Ingesting {len(geojson_files)} GeoJSON file(s)...", end=" ", flush=True)
        t0 = time.time()
        if len(geojson_files) == 1:
            geojson_str = geojson_files[0].read_text()
            fids = gen.add_geojson(geojson_str, bounds)
        else:
            fids = gen.add_geojson_files([str(f) for f in geojson_files], bounds)
        t_ingest = time.time() - t0
        print(f"{len(fids)} features ({t_ingest:.1f}s)", flush=True)

        # PBF
        print("Encoding PBF...", end=" ", flush=True)
        t0 = time.time()
        mvt_dir = out_dir / "vectors"
        generate_pbf(gen, str(mvt_dir), bounds, simplify=simplify, layer_name=layer_name)
        t_pbf = time.time() - t0
        print(f"done ({t_pbf:.1f}s)", flush=True)

        # Parquet
        print("Encoding Parquet...", end=" ", flush=True)
        t0 = time.time()
        pq_dir = out_dir / "features.parquet"
        # G1 (streaming_review.md §G): bounded path (was unbounded
        # generate_parquet_native, which read the whole corpus into RAM). Peak
        # scales with max_batch_bytes (default 2 GB), not corpus size.
        pq_rows = gen.generate_parquet_native_partitioned(str(pq_dir), bounds, simplify=True)
        t_pq = time.time() - t0
        print(f"{pq_rows:,} rows ({t_pq:.1f}s)", flush=True)

        total_time = time.time() - t_start
        print(f"Done. Output: {out_dir} ({total_time:.0f}s)", flush=True)

        return {
            "total_time": total_time,
            "feature_count": len(fids),
            "timings": {"ingest": t_ingest, "pbf": t_pbf, "parquet": t_pq},
        }

    def _compute_bounds(self, files):
        # PY-3 (streaming_review.md §G): this previously called a `pass`-only
        # helper, so auto-bounds ALWAYS fell through to the (0,0,1,1) fallback —
        # every feature then projected into a degenerate unit square. Accumulate
        # the real extent by walking the (arbitrarily nested) coordinate arrays.
        acc = [float("inf"), float("inf"), float("-inf"), float("-inf")]  # xmin,ymin,xmax,ymax
        for f in files:
            fc = json.loads(f.read_text())
            for feat in fc.get("features", []):
                self._update_bounds_from_geometry(feat.get("geometry") or {}, acc)
        xmin, ymin, xmax, ymax = acc
        return (xmin, ymin, xmax, ymax) if xmin != float("inf") else (0, 0, 1, 1)

    def _update_bounds_from_geometry(self, geom, acc):
        """Expand acc over a GeoJSON geometry. A GeometryCollection stores its
        members under `geometries` (not `coordinates`), so dispatch on type and
        recurse; every other geometry walks its `coordinates`."""
        if not isinstance(geom, dict):
            return
        if geom.get("type") == "GeometryCollection":
            for sub in geom.get("geometries", []) or []:
                self._update_bounds_from_geometry(sub, acc)
        else:
            self._update_bounds_from_coords(geom.get("coordinates", []), acc)

    def _update_bounds_from_coords(self, coords, acc):
        """Recursively expand acc=[xmin,ymin,xmax,ymax] over a GeoJSON
        coordinate array of arbitrary nesting (Point→…→MultiPolygon)."""
        if not isinstance(coords, (list, tuple)) or not coords:
            return
        # A position is [x, y, ...] with numeric leads; anything else recurses.
        if isinstance(coords[0], (int, float)) and len(coords) >= 2:
            x, y = coords[0], coords[1]
            if x < acc[0]:
                acc[0] = x
            if y < acc[1]:
                acc[1] = y
            if x > acc[2]:
                acc[2] = x
            if y > acc[3]:
                acc[3] = y
            return
        for c in coords:
            self._update_bounds_from_coords(c, acc)
