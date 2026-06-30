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

        # Facet policy: when `config["facets"]` is present, high-card numeric
        # per-cell attributes are pulled OUT of the vector tiles into a
        # `cell_id`-keyed facet store. We strip those keys from every feature's
        # properties BEFORE ingest (so they never enter the PBF/Parquet) and emit
        # them separately after tiling. Geometry + the inline keys (cell_id,
        # cell_type, non-faceted props) still tile normally.
        facet_cfg = config.get("facets")
        facet_keys: list[str] = []
        cell_ids: list[str] = []
        facet_attrs: dict[str, Any] = {}
        policy = None
        if facet_cfg:
            from mudm_tools.facets import FacetPolicy, select_facet_keys

            policy = FacetPolicy.from_config(facet_cfg)
            features = self._load_features(geojson_files)
            columns, cardinality = self._scan_properties(features)
            facet_keys, _inline_keys = select_facet_keys(policy, columns, cardinality)
            if facet_keys:
                cell_ids, facet_attrs = self._collect_facet_attrs(
                    features, facet_keys, policy.key
                )
                # Write a properties-stripped GeoJSON (facet keys removed) to feed
                # the tiler. One combined file keeps the single-file ingest path.
                stripped = {
                    "type": "FeatureCollection",
                    "features": self._strip_facet_keys(features, facet_keys),
                }
                fd, stripped_path = tempfile.mkstemp(
                    suffix=".geojson", prefix="facet_stripped_", dir=temp_dir
                )
                Path(stripped_path).write_text(json.dumps(stripped))
                import os

                os.close(fd)
                geojson_files = [Path(stripped_path)]

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

        # Facet store: emit the cell_id-keyed Parquet + patch metadata.json. The
        # faceted keys are already absent from the tiles (stripped pre-ingest).
        facets_block = None
        if policy is not None and facet_keys:
            from mudm_tools.facets import emit_facet_store

            print("Encoding facet store...", end=" ", flush=True)
            t0 = time.time()
            facets_block = emit_facet_store(
                str(out_dir), cell_ids, facet_attrs, policy
            )
            print(f"{len(facet_keys)} field(s) ({time.time() - t0:.1f}s)", flush=True)

            metadata = {
                "name": Path(input_dir).stem,
                "vectors": {"path": "vectors/{z}/{x}/{y}.pbf"},
                "parquet": {"path": "features.parquet", "partitioned": True},
                "facets": facets_block,
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        total_time = time.time() - t_start
        print(f"Done. Output: {out_dir} ({total_time:.0f}s)", flush=True)

        result = {
            "total_time": total_time,
            "feature_count": len(fids),
            "timings": {"ingest": t_ingest, "pbf": t_pbf, "parquet": t_pq},
        }
        if facets_block is not None:
            result["facets"] = facets_block
        return result

    def _load_features(self, files) -> list[dict]:
        """Read all features from the input GeoJSON file(s), in file/feature order."""
        feats: list[dict] = []
        for f in files:
            fc = json.loads(Path(f).read_text())
            feats.extend(fc.get("features", []) or [])
        return feats

    def _scan_properties(self, features: list[dict]):
        """Infer per-property numpy dtype + cardinality from feature properties.

        Returns ``(columns, cardinality)`` where ``columns`` maps property name ->
        ``numpy.dtype`` (inferred from the non-null sample values) and
        ``cardinality`` maps property name -> distinct non-null value count.
        """
        import numpy as np

        values: dict[str, list] = {}
        distinct: dict[str, set] = {}
        for feat in features:
            props = feat.get("properties") or {}
            for k, v in props.items():
                if v is None:
                    continue
                values.setdefault(k, []).append(v)
                distinct.setdefault(k, set()).add(v)
        columns: dict[str, Any] = {}
        for k, vals in values.items():
            columns[k] = np.array(vals).dtype
        cardinality = {k: len(s) for k, s in distinct.items()}
        return columns, cardinality

    def _collect_facet_attrs(self, features: list[dict], facet_keys: list[str], key: str):
        """Build ``cell_id``-aligned arrays for the facet keys.

        Returns ``(cell_ids, attributes)`` where ``cell_ids`` is one id per
        feature (``properties[key]`` or the positional index as a fallback) and
        ``attributes`` maps each facet key to a per-feature ``numpy`` array
        (missing numeric values become NaN).
        """
        import numpy as np

        cell_ids: list[str] = []
        raw: dict[str, list] = {k: [] for k in facet_keys}
        for i, feat in enumerate(features):
            props = feat.get("properties") or {}
            cid = props.get(key, i)
            cell_ids.append(str(cid))
            for k in facet_keys:
                raw[k].append(props.get(k, None))
        attributes: dict[str, Any] = {}
        for k, vals in raw.items():
            arr = np.array([np.nan if v is None else v for v in vals], dtype="float32")
            attributes[k] = arr
        return cell_ids, attributes

    def _strip_facet_keys(self, features: list[dict], facet_keys: list[str]) -> list[dict]:
        """Return features with the faceted property keys removed (geometry kept)."""
        drop = set(facet_keys)
        out = []
        for feat in features:
            props = feat.get("properties") or {}
            new_props = {k: v for k, v in props.items() if k not in drop}
            out.append({**feat, "properties": new_props})
        return out

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
