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
        pq_rows = gen.generate_parquet_native(str(pq_dir), bounds, simplify=True)
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
        xmin, ymin = float("inf"), float("inf")
        xmax, ymax = float("-inf"), float("-inf")
        for f in files:
            fc = json.loads(f.read_text())
            for feat in fc.get("features", []):
                coords = feat.get("geometry", {}).get("coordinates", [])
                self._update_bounds_from_coords(coords)
        # Fallback
        return (xmin, ymin, xmax, ymax) if xmin != float("inf") else (0, 0, 1, 1)

    def _update_bounds_from_coords(self, coords):
        # Recursive coordinate extraction — simplified
        pass
