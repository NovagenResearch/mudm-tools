#!/usr/bin/env python3
"""Export MuDM data to GeoParquet for ML/data-science pipelines.

Demonstrates two scenarios:
  1. SWC neuron files → GeoParquet (TIN geometry)
  2. Mixed 2D/3D geometry → GeoParquet

Usage:
    # Export SWC files from swcs/ directory
    .venv/bin/python scripts/export_parquet.py

    # Export specific SWC files
    .venv/bin/python scripts/export_parquet.py file1.swc file2.swc

    # Custom output path
    .venv/bin/python scripts/export_parquet.py -o neurons.parquet

    # Run the built-in mixed-geometry demo
    .venv/bin/python scripts/export_parquet.py --demo

Then inspect with:
    python -c "import geopandas; print(geopandas.read_parquet('output.parquet'))"
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from mudm_tools.arrow import ArrowConfig, to_arrow_table, to_geoparquet
from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
    PolyhedralSurface,
)

from geojson_pydantic import MultiLineString, MultiPolygon, Point, Polygon

DEFAULT_SWC_DIR = "swcs"
DEFAULT_OUT = "output.parquet"


def _pop_flag(args, flag):
    if flag in args:
        idx = args.index(flag)
        val = args[idx + 1]
        del args[idx : idx + 2]
        return val
    return None


def _demo_collection():
    """Build a mixed-geometry MuDMFeatureCollection for demonstration."""
    # A 3D point
    point_feat = MuDMFeature(
        type="Feature",
        id="cell_001",
        geometry=Point(type="Point", coordinates=(100.5, 200.3, 15.0)),
        properties={"cell_type": "pyramidal", "layer": 5, "area_um2": 312.7},
        featureClass="cell",
    )

    # A polygon (2D region)
    region_feat = MuDMFeature(
        type="Feature",
        id="region_001",
        geometry=Polygon(
            type="Polygon",
            coordinates=[
                [(0, 0), (500, 0), (500, 500), (0, 500), (0, 0)],
                [(100, 100), (400, 100), (400, 400), (100, 400), (100, 100)],
            ],
        ),
        properties={"region_name": "cortex", "area_um2": 210000.0},
        featureClass="region",
    )

    # A simple 3D neuron skeleton (MultiLineString — each edge is a segment)
    neuron_feat = MuDMFeature(
        type="Feature",
        id="neuron_001",
        geometry=MultiLineString(
            type="MultiLineString",
            coordinates=[
                [[250, 250, 10], [280, 250, 12]],  # soma → axon
                [[280, 250, 12], [310, 260, 15]],
                [[250, 250, 10], [250, 220, 8]],    # soma → basal dendrite
                [[250, 220, 8], [240, 190, 5]],
                [[250, 250, 10], [250, 280, 12]],   # soma → apical dendrite
                [[250, 280, 12], [260, 310, 18]],
            ],
        ),
        properties={"species": "mouse", "brain_region": "hippocampus"},
        featureClass="neuron",
    )

    # A 3D polyhedral surface (closed mesh)
    surface_feat = MuDMFeature(
        type="Feature",
        id="surface_001",
        geometry=PolyhedralSurface(
            type="PolyhedralSurface",
            coordinates=[
                [[(50, 50, 0), (150, 50, 0), (100, 100, 0), (50, 50, 0)]],
                [[(50, 50, 0), (150, 50, 0), (100, 100, 10), (50, 50, 0)]],
                [[(150, 50, 0), (100, 100, 0), (100, 100, 10), (150, 50, 0)]],
                [[(100, 100, 0), (50, 50, 0), (100, 100, 10), (100, 100, 0)]],
            ],
        ),
        properties={"structure": "surface_mesh"},
        featureClass="surface",
    )

    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=[point_feat, region_feat, neuron_feat, surface_feat],
    )


def _print_table_info(path):
    """Print a summary of the written GeoParquet file."""
    table = pq.read_table(str(path))
    meta = json.loads(table.schema.metadata[b"geo"])

    print(f"\n  Rows: {len(table)}")
    print(f"  Columns: {table.column_names}")
    print(f"  GeoParquet version: {meta['version']}")
    print(f"  Geometry types: {meta['columns']['geometry']['geometry_types']}")
    if "bbox" in meta["columns"]["geometry"]:
        bbox = meta["columns"]["geometry"]["bbox"]
        print(f"  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    print(f"  File size: {path.stat().st_size / 1024:.1f} KB")


def main():
    args = sys.argv[1:]

    out_path = _pop_flag(args, "-o") or _pop_flag(args, "--output") or DEFAULT_OUT
    demo = "--demo" in args
    if demo:
        args.remove("--demo")

    out = Path(out_path)

    if demo:
        # Mixed geometry demo
        print("Building mixed-geometry demo collection...")
        coll = _demo_collection()
        to_geoparquet(coll, out)
        print(f"Wrote {out}")
        _print_table_info(out)

        # Also show the Arrow table in-memory
        table = to_arrow_table(coll)
        print(f"\n  Arrow table schema:")
        for field in table.schema:
            print(f"    {field.name}: {field.type}")

        print(f"\n  Inspect with:")
        print(f"    python -c \"import geopandas; print(geopandas.read_parquet('{out}'))\"")
        return

    # SWC mode — one Feature per compartment, with "compartment" property
    from mudm_tools.swc import swc_to_feature_collection

    if args:
        swc_paths = [Path(a) for a in args]
    else:
        swc_dir = Path(DEFAULT_SWC_DIR)
        swc_paths = sorted(swc_dir.glob("*.swc"))
        if not swc_paths:
            print(f"No SWC files in {swc_dir}/. Use --demo for a built-in example.")
            sys.exit(1)

    for p in swc_paths:
        if not p.exists():
            print(f"Error: file not found: {p}")
            sys.exit(1)

    features = []
    for swc_path in swc_paths:
        neuron_coll = swc_to_feature_collection(str(swc_path))
        for feat in neuron_coll.features:
            if feat.properties is None:
                feat.properties = {}
            feat.properties["source_file"] = swc_path.name
            features.append(feat)

    print(f"Loaded {len(swc_paths)} SWC file(s) → {len(features)} feature(s)")

    coll = MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
    )

    to_geoparquet(coll, out)
    print(f"Wrote {out}")
    _print_table_info(out)


if __name__ == "__main__":
    main()
