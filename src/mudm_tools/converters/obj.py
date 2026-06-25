"""OBJ mesh → muDM tiled 3D format.

Converts OBJ mesh files into octree-tiled 3D Tiles (GLB + Meshopt)
and partitioned Parquet.

Source files:
    *.obj  — Wavefront OBJ mesh files (one per feature/neuron/region)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from . import register


@register("obj")
class ObjConverter:
    """Convert OBJ mesh files to muDM tiled 3D format."""

    def convert(
        self,
        input_dir: str,
        output_dir: str,
        config: dict[str, Any],
    ) -> dict:
        """Convert OBJ meshes to tiled 3D output.

        Config keys:
            temp_dir (str): Temp directory for fragments.
            max_zoom (int): Max zoom level. Default: 4.
            min_zoom (int): Min zoom level. Default: 0.
            bounds (tuple): World bounds (xmin,ymin,zmin,xmax,ymax,zmax).
                If not provided, scans all OBJ files.
            tags (dict): Per-file tags. Keys are filenames (without .obj),
                values are dicts of properties.
            glob (str): Glob pattern for OBJ files. Default: "*.obj".
            generate_parquet (bool): Also generate Parquet. Default: True.
            compression (str): GLB tile compression — "meshopt-q14" (default),
                "meshopt", "meshopt-q16", "meshopt-q10", "draco", or "none". The
                meshopt-q* variants u16-quantize positions via KHR_mesh_quantization
                (~25% smaller wire than raw-f32 "meshopt", visually lossless for these
                surfaces); all are decoded natively by the bundled 3D viewer
                (EXT_meshopt_compression [+ KHR_mesh_quantization]).
        """
        from mudm_tools._rs import StreamingTileGenerator

        data_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        temp_dir = config.get("temp_dir")
        max_zoom = config.get("max_zoom", 4)
        min_zoom = config.get("min_zoom", 0)
        bounds = config.get("bounds")
        tags_map = config.get("tags", {})
        glob_pattern = config.get("glob", "*.obj")
        do_parquet = config.get("generate_parquet", True)
        compression = config.get("compression", "meshopt-q14")
        # LOD mesh simplification (per-zoom QEM decimation). Default on. Set False for small
        # datasets where decimation facets smooth surfaces (e.g. HRA anatomy) and the size
        # savings aren't needed — tiles stay full-detail at every zoom.
        simplify = config.get("simplify", True)

        t_start = time.time()

        # Find OBJ files
        obj_files = sorted(data_dir.glob(glob_pattern))
        if not obj_files:
            raise FileNotFoundError(f"No {glob_pattern} files in {data_dir}")
        print(f"Found {len(obj_files)} OBJ files", flush=True)

        # Scan bounds if not provided
        if bounds is None:
            from mudm_tools._rs import scan_obj_bounds

            print("Scanning OBJ bounds...", end=" ", flush=True)
            t0 = time.time()
            bounds = scan_obj_bounds([str(f) for f in obj_files])
            print(f"done ({time.time()-t0:.1f}s)", flush=True)

        # Create generator
        gen = StreamingTileGenerator(
            min_zoom=min_zoom,
            max_zoom=max_zoom,
            temp_dir=temp_dir,
        )

        # Build tags list
        all_tags = []
        for f in obj_files:
            name = f.stem
            file_tags = tags_map.get(name, {"name": name})
            all_tags.append(file_tags)

        # Ingest all OBJ files (parallel Rayon)
        print(f"Ingesting {len(obj_files)} meshes...", end=" ", flush=True)
        t0 = time.time()
        fids = gen.add_obj_files(
            [str(f) for f in obj_files],
            bounds,
            all_tags,
            simplify=simplify,
        )
        t_ingest = time.time() - t0
        print(f"{len(fids)} features ({t_ingest:.1f}s)", flush=True)

        # Generate 3D Tiles
        print("Encoding 3D Tiles...", end=" ", flush=True)
        t0 = time.time()
        tiles_dir = out_dir / "3dtiles"
        gen.generate_3dtiles(str(tiles_dir), bounds, compression=compression)
        t_tiles = time.time() - t0
        print(f"done ({t_tiles:.1f}s, compression={compression})", flush=True)

        # Generate Parquet
        t_parquet = 0.0
        if do_parquet:
            # G1 (streaming_review.md §G): route through the bounded,
            # memory-ceiling-aware partitioned path. The previous call to the
            # unbounded gen.generate_parquet_native loaded the WHOLE decoded
            # corpus into RAM — the real OOM exposure for large OBJ corpora. The
            # wrapper derives max_batch_bytes from the configured ceiling; output
            # rows are content-identical (only part-file numbering differs).
            # `simplify` was already applied at ingest (add_obj_files above).
            from mudm_tools.tiling3d.parquet_writer import generate_parquet

            print("Encoding Parquet...", end=" ", flush=True)
            t0 = time.time()
            pq_dir = out_dir / "features.parquet"
            pq_rows = generate_parquet(gen, str(pq_dir), bounds, partitioned=True)
            t_parquet = time.time() - t0
            print(f"{pq_rows:,} rows ({t_parquet:.1f}s)", flush=True)

        total_time = time.time() - t_start
        print(f"Done. Output: {out_dir} ({total_time:.0f}s)", flush=True)

        return {
            "total_time": total_time,
            "feature_count": len(fids),
            "timings": {"ingest": t_ingest, "tiles": t_tiles, "parquet": t_parquet},
        }
