"""Example: 2D tiling with the Rust-based pipeline.

Generates random polygons with ``polygen``, then uses the Rust
``StreamingTileGenerator2D`` to clip them through a quadtree and write
a tiled Parquet file for ML training.

Usage::

    uv run python src/mudm/examples/tiling_rust.py
    uv run python src/mudm/examples/tiling_rust.py --max-zoom 6 --partitioned
    uv run python src/mudm/examples/tiling_rust.py my_data.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d import generate_parquet, read_parquet, generate_pbf, read_pbf


def _generate_sample_geojson(path: Path, grid_size: int, cell_size: int) -> None:
    """Generate random polygons and write them as GeoJSON."""
    from mudm_tools.polygen import generate_polygons

    meta_types = {"num_vertices": "int"}
    meta_values_options = {"polytype": ["Type1", "Type2", "Type3", "Type4"]}

    generate_polygons(
        grid_size,
        cell_size,
        min_vertices=10,
        max_vertices=100,
        meta_types=meta_types,
        meta_values_options=meta_values_options,
        microjson_data_path=str(path),
    )


def _compute_bounds(geojson_str: str) -> tuple[float, float, float, float]:
    """Compute the bounding box of a GeoJSON string."""
    data = json.loads(geojson_str)
    xs: list[float] = []
    ys: list[float] = []

    def _walk(coords):
        if not isinstance(coords, list) or len(coords) == 0:
            return
        if isinstance(coords[0], (int, float)):
            xs.append(coords[0])
            ys.append(coords[1])
        else:
            for c in coords:
                _walk(c)

    features = data.get("features", [data] if "geometry" in data else [])
    for feat in features:
        geom = feat.get("geometry", feat)
        _walk(geom.get("coordinates", []))

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    # Ensure non-degenerate bounds
    if xmin == xmax:
        xmin, xmax = xmin - 1, xmax + 1
    if ymin == ymax:
        ymin, ymax = ymin - 1, ymax + 1
    return (xmin, ymin, xmax, ymax)


def main():
    parser = argparse.ArgumentParser(description="2D tiling example using the Rust-based pipeline")
    parser.add_argument(
        "geojson_path",
        nargs="?",
        default="",
        help="Path to a GeoJSON file. If omitted, random polygons are generated.",
    )
    parser.add_argument("--min-zoom", type=int, default=0)
    parser.add_argument("--max-zoom", type=int, default=7)
    parser.add_argument("--output", default="tiles_2d.parquet", help="Output Parquet path")
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="Write Hive-partitioned output (one directory per zoom level)",
    )
    parser.add_argument(
        "--pbf",
        action="store_true",
        help="Output PBF vector tiles instead of Parquet",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable Douglas-Peucker simplification at coarse zoom levels",
    )
    parser.add_argument(
        "--buffer", type=int, default=64, help="Tile buffer in pixels (at extent 4096)"
    )
    parser.add_argument("--grid-size", type=int, default=10000)
    parser.add_argument("--cell-size", type=int, default=100)
    args = parser.parse_args()

    # --- 1. Get GeoJSON data ---
    if args.geojson_path:
        geojson_path = Path(args.geojson_path)
        print(f"Loading GeoJSON from {geojson_path}")
    else:
        geojson_path = Path("example_polygen.json")
        print(f"Generating random polygons " f"(grid={args.grid_size}, cell={args.cell_size}) ...")
        _generate_sample_geojson(geojson_path, args.grid_size, args.cell_size)
        print(f"  Wrote {geojson_path}")

    geojson_str = geojson_path.read_text()
    bounds = _compute_bounds(geojson_str)
    print(f"  Bounds: {bounds}")

    # --- 2. Ingest through the Rust quadtree pipeline ---
    # Buffer in normalized tile-fraction space: pixels / extent
    buffer = args.buffer / 4096.0
    gen = StreamingTileGenerator2D(min_zoom=args.min_zoom, max_zoom=args.max_zoom, buffer=buffer)

    t0 = time.perf_counter()
    fids = gen.add_geojson(geojson_str, bounds)
    t_ingest = time.perf_counter() - t0
    print(f"  Ingested {len(fids)} features in {t_ingest:.2f}s")

    # --- 3. Write output ---
    output = Path(args.output)
    t0 = time.perf_counter()

    if args.pbf:
        if output.suffix == ".parquet":
            output = Path("tiles")
        n_out = generate_pbf(
            gen,
            output,
            bounds,
            simplify=not args.no_simplify,
        )
        t_write = time.perf_counter() - t0
        print(f"  Wrote {n_out} PBF tiles to {output} in {t_write:.2f}s")

        # Quick verification
        rows = read_pbf(output, bounds, zoom=args.min_zoom)
        print(f"  Zoom {args.min_zoom}: {len(rows)} features")
        if rows:
            r = rows[0]
            print(
                f"    First: tile=({r['tile_x']},{r['tile_y']}), "
                f"geom_type={r['geom_type']}, "
                f"vertices={r['positions'].shape[0]}, "
                f"tags={r['tags']}"
            )
    else:
        n_rows = generate_parquet(
            gen,
            output,
            bounds,
            partitioned=args.partitioned,
            simplify=not args.no_simplify,
        )
        t_write = time.perf_counter() - t0
        print(f"  Wrote {n_rows} rows to {output} in {t_write:.2f}s")

        # Quick verification
        rows = read_parquet(output, zoom=args.min_zoom)
        print(f"  Zoom {args.min_zoom}: {len(rows)} rows")
        if rows:
            r = rows[0]
            print(
                f"    First row: tile=({r['tile_x']},{r['tile_y']}), "
                f"geom_type={r['geom_type']}, "
                f"vertices={r['positions'].shape[0]}, "
                f"tags={r['tags']}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
