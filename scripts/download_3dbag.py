#!/usr/bin/env python3
"""Download 3DBAG Amsterdam buildings for benchmarking.

Downloads building meshes with attributes from the 3DBAG API (CityJSON format),
converts to MuDM TIN features, tiles with TileGenerator3D, and benchmarks.

Data source: https://3dbag.nl — Netherlands 3D building models.
CRS: Amersfoort/RD New + NAP (EPSG:7415).

Usage::

    # Full pipeline (default: ~5K buildings for testing):
    .venv/bin/python scripts/download_3dbag.py --download --convert --tile --benchmark

    # Larger Amsterdam subset (~100K buildings):
    .venv/bin/python scripts/download_3dbag.py --download --convert --tile --benchmark \
      --bbox 119000,484000,124000,488000

    # Download only:
    .venv/bin/python scripts/download_3dbag.py --download --max-buildings 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "scripts"))

_DATA_DIR = _ROOT / "data" / "3dbag"
_TILES_DIR = _DATA_DIR / "tiles"

_API_BASE = "https://api.3dbag.nl/collections/pand/items"
# Default: 1km² in Amsterdam center (~5K buildings)
_DEFAULT_BBOX = "120000,485000,121000,486000"
# Full benchmark: ~5km × 4km → ~100K buildings
_FULL_BBOX = "119000,484000,124000,488000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.0f}s"


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


# ---------------------------------------------------------------------------
# Step 1: Download from 3DBAG API
# ---------------------------------------------------------------------------

def download_buildings(
    data_dir: Path,
    bbox: str,
    *,
    max_buildings: int | None = None,
    page_size: int = 100,
) -> tuple[list[dict], dict]:
    """Download building data from 3DBAG API with pagination.

    Returns (buildings, transform) where buildings is a list of CityJSON features
    and transform contains scale/translate.
    """
    import requests

    data_dir.mkdir(parents=True, exist_ok=True)

    # First request to get total count
    resp = requests.get(
        _API_BASE,
        params={"bbox": bbox, "limit": 1},
        timeout=30,
    )
    resp.raise_for_status()
    first = resp.json()
    total = first.get("numberMatched", 0)

    # Extract CityJSON transform from metadata
    metadata = first.get("metadata", {})
    transform = metadata.get("transform", {
        "scale": [0.001, 0.001, 0.001],
        "translate": [0, 0, 0],
    })

    target = min(total, max_buildings) if max_buildings else total
    print(f"3DBAG API: {total:,} buildings in bbox {bbox}")
    print(f"  Downloading {target:,} buildings (page size: {page_size})...")

    buildings: list[dict] = []
    t0 = time.perf_counter()

    # The 3DBAG API fails with offset=0; omit offset on first request,
    # then follow the "next" link for subsequent pages.
    # Note: API limit counts CityObjects (2 per building), so we request
    # 2× the desired building count per page.
    next_url: str | None = None
    first_page = True

    while len(buildings) < target:
        remaining = target - len(buildings)
        limit = min(page_size * 2, remaining * 2)  # CityObjects, not features

        if first_page:
            resp = requests.get(
                _API_BASE,
                params={"bbox": bbox, "limit": limit},
                timeout=60,
            )
            first_page = False
        elif next_url:
            resp = requests.get(next_url, timeout=60)
        else:
            break

        resp.raise_for_status()
        page = resp.json()

        features = page.get("features", [])
        if not features:
            break

        buildings.extend(features)

        # Find next page URL
        links = page.get("links", [])
        next_links = [l for l in links if l.get("rel") == "next"]
        next_url = next_links[0]["href"] if next_links else None

        if len(buildings) % 500 < page_size or not next_url:
            elapsed = time.perf_counter() - t0
            rate = len(buildings) / elapsed if elapsed > 0 else 0
            print(f"  [{len(buildings):,}/{target:,}] {rate:.0f} buildings/s")

        # Update transform from latest page if available
        page_meta = page.get("metadata", {})
        if "transform" in page_meta:
            transform = page_meta["transform"]

    dl_time = time.perf_counter() - t0
    print(f"  Downloaded {len(buildings):,} buildings in {_fmt_time(dl_time)}")

    # Save raw data
    raw_path = data_dir / "buildings_raw.json"
    json_data = {
        "buildings": buildings,
        "transform": transform,
        "bbox": bbox,
        "count": len(buildings),
    }
    raw_path.write_text(json.dumps(json_data))
    print(f"  Saved to {raw_path} ({_fmt_bytes(raw_path.stat().st_size)})")

    return buildings, transform


# ---------------------------------------------------------------------------
# Step 2: Parse CityJSON geometry
# ---------------------------------------------------------------------------

def cityjson_to_mesh(
    feature: dict,
    transform: dict,
    lod: str = "2.2",
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Extract mesh + attributes from a CityJSON feature.

    Returns (vertices, faces, attributes) or None if no suitable geometry.
    """
    scale = transform.get("scale", [1, 1, 1])
    translate = transform.get("translate", [0, 0, 0])

    # Transform integer vertices to real coordinates
    raw_verts = feature.get("vertices", [])
    if not raw_verts:
        return None

    vertices = np.array(raw_verts, dtype=np.float64)
    vertices[:, 0] = vertices[:, 0] * scale[0] + translate[0]
    vertices[:, 1] = vertices[:, 1] * scale[1] + translate[1]
    vertices[:, 2] = vertices[:, 2] * scale[2] + translate[2]

    # Find geometry at requested LoD
    city_objects = feature.get("CityObjects", {})
    geom = None
    attrs = {}

    # Prefer BuildingPart (has LoD geometry), fall back to Building
    for co_name, co in city_objects.items():
        co_type = co.get("type", "")

        # Collect attributes from Building (not BuildingPart)
        if co_type == "Building":
            a = co.get("attributes", {})
            attrs = {
                "building_id": co_name,
                "year_built": a.get("oorspronkelijkbouwjaar"),
                "height_max": a.get("b3_h_dak_max"),
                "height_ground": a.get("b3_h_maaiveld"),
                "floors": a.get("b3_bouwlagen"),
                "roof_type": a.get("b3_dak_type"),
                "volume_lod22": a.get("b3_volume_lod22"),
                "area_ground": a.get("b3_opp_grond"),
                "area_roof_flat": a.get("b3_opp_dak_plat"),
                "area_roof_slanted": a.get("b3_opp_dak_schuin"),
                "area_wall": a.get("b3_opp_buitenmuur"),
            }
            # Remove None values
            attrs = {k: v for k, v in attrs.items() if v is not None}

        if co_type == "BuildingPart":
            for g in co.get("geometry", []):
                if str(g.get("lod")) == lod:
                    geom = g
                    break
            # Fallback to any available LoD
            if geom is None and co.get("geometry"):
                geom = co["geometry"][-1]  # highest LoD

    if geom is None:
        # Try Building geometry
        for co_name, co in city_objects.items():
            if co.get("type") == "Building":
                for g in co.get("geometry", []):
                    if str(g.get("lod")) == lod:
                        geom = g
                        break
                if geom is None and co.get("geometry"):
                    geom = co["geometry"][-1]

    if geom is None:
        return None

    # Parse boundaries → triangulate faces
    faces: list[list[int]] = []
    boundaries = geom.get("boundaries", [])

    def _extract_faces(boundary_data, depth=0):
        """Recursively extract face rings from CityJSON boundary nesting."""
        if depth > 5:
            return
        if not boundary_data:
            return
        # Check if this is a ring (list of ints)
        if isinstance(boundary_data[0], int):
            ring = boundary_data
            # Fan-triangulate polygon
            for j in range(1, len(ring) - 1):
                faces.append([ring[0], ring[j], ring[j + 1]])
            return
        # Otherwise recurse deeper
        for item in boundary_data:
            _extract_faces(item, depth + 1)

    _extract_faces(boundaries)

    if not faces:
        return None

    face_arr = np.array(faces, dtype=np.uint32)

    # Validate face indices
    max_idx = face_arr.max()
    if max_idx >= len(vertices):
        return None

    return vertices, face_arr, attrs


# ---------------------------------------------------------------------------
# Step 3: Convert to MuDM
# ---------------------------------------------------------------------------

def convert_to_microjson(
    data_dir: Path,
    *,
    lod: str = "2.2",
    max_buildings: int | None = None,
):
    """Convert downloaded 3DBAG buildings to MuDMFeatureCollection."""
    from mudm.model import (
        MuDMFeature,
        MuDMFeatureCollection,
        Vocabulary,
    )
    from mudm_tools.swc import _mesh_to_tin

    raw_path = data_dir / "buildings_raw.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run --download first.", file=sys.stderr)
        sys.exit(1)

    print("Loading raw building data...")
    raw = json.loads(raw_path.read_text())
    buildings = raw["buildings"]
    transform = raw["transform"]

    if max_buildings and max_buildings < len(buildings):
        buildings = buildings[:max_buildings]

    print(f"Converting {len(buildings)} buildings to MuDM (LoD {lod})...")
    t0 = time.perf_counter()
    features: list[MuDMFeature] = []
    total_verts = 0
    total_faces = 0
    roof_types: set[str] = set()
    skipped = 0

    for i, building in enumerate(buildings):
        if (i + 1) % 1000 == 0 or i + 1 == len(buildings):
            print(f"  [{i+1:,}/{len(buildings):,}]", end="\r", file=sys.stderr)

        result = cityjson_to_mesh(building, transform, lod=lod)
        if result is None:
            skipped += 1
            continue

        vertices, face_arr, attrs = result
        tin = _mesh_to_tin(vertices, face_arr)

        props = dict(attrs)
        props["vertex_count"] = int(vertices.shape[0])
        props["face_count"] = int(face_arr.shape[0])

        rt = attrs.get("roof_type")
        if rt:
            roof_types.add(rt)

        feature_class = rt if rt else "building"

        features.append(MuDMFeature(
            type="Feature",
            geometry=tin,
            properties=props,
            featureClass=feature_class,
        ))
        total_verts += vertices.shape[0]
        total_faces += face_arr.shape[0]

    print(file=sys.stderr)
    convert_time = time.perf_counter() - t0

    # Build vocabulary for roof types
    vocabs = None
    if roof_types:
        from mudm.model import OntologyTerm
        terms = {
            rt: OntologyTerm(
                uri=f"https://3dbag.nl/schema/roof_type/{rt}",
                label=rt,
            )
            for rt in sorted(roof_types)
        }
        vocabs = {
            "roof_types": Vocabulary(
                namespace="https://3dbag.nl/schema/",
                description="3DBAG building roof type classification",
                terms=terms,
            ),
        }

    collection = MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={
            "dataset": "3DBAG_Amsterdam",
            "building_count": len(features),
            "total_vertices": total_verts,
            "total_faces": total_faces,
            "lod": lod,
            "crs": "EPSG:7415",
        },
        vocabularies=vocabs,
    )

    print(f"  {len(features)} buildings, {total_verts:,} vertices, {total_faces:,} faces")
    print(f"  {skipped} buildings skipped (no geometry)")
    print(f"  {len(roof_types)} roof types: {sorted(roof_types)}")
    print(f"  Conversion time: {_fmt_time(convert_time)}")

    # Save metadata
    meta = {
        "dataset": "3DBAG_Amsterdam",
        "building_count": len(features),
        "total_vertices": total_verts,
        "total_faces": total_faces,
        "lod": lod,
        "roof_types": sorted(roof_types),
        "bbox": raw.get("bbox", ""),
    }
    meta_path = data_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    return collection, convert_time


# ---------------------------------------------------------------------------
# Step 4 & 5: Tile + Benchmark
# ---------------------------------------------------------------------------

def tile_and_benchmark(
    collection,
    output_dir: Path,
    *,
    max_zoom: int = 3,
    workers: int | None = None,
    do_tile: bool = True,
    do_benchmark: bool = True,
    skip_3dtiles: bool = False,
    csv_path: Path | None = None,
) -> dict:
    """Tile the collection and run benchmarks."""
    from benchmark_mouselight import (
        bench_decode,
        bench_decode_3dtiles,
        bench_memory,
        export_csv,
        generate_tiles,
        print_report,
    )

    results: dict = {}

    if do_tile:
        tile_results = generate_tiles(
            collection,
            output_dir,
            max_zoom=max_zoom,
            workers=workers,
            skip_3dtiles=skip_3dtiles,
        )
        results["tile"] = tile_results

    if do_benchmark:
        pbf3_dir = output_dir / "pbf3"
        tiles3d_dir = output_dir / "3dtiles" if not skip_3dtiles else None

        decode_pbf3: dict = {}
        if pbf3_dir.exists():
            print(f"\nBenchmarking pbf3 decode...")
            decode_pbf3 = bench_decode(pbf3_dir)

        decode_3dt: dict = {}
        if tiles3d_dir and tiles3d_dir.exists():
            print(f"Benchmarking 3D Tiles decode...")
            decode_3dt = bench_decode_3dtiles(tiles3d_dir)

        print("Measuring peak memory...")
        memory = bench_memory(
            pbf3_dir if pbf3_dir.exists() else Path("/dev/null"),
            tiles3d_dir if tiles3d_dir and tiles3d_dir.exists() else None,
        )

        results["decode_pbf3"] = decode_pbf3
        results["decode_3dt"] = decode_3dt
        results["memory"] = memory

        if do_tile:
            print_report(0, tile_results, decode_pbf3, decode_3dt, memory, None)

        if csv_path:
            export_csv(csv_path, 0, tile_results if do_tile else {},
                       decode_pbf3, decode_3dt, memory, None)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="3DBAG Amsterdam download, conversion, tiling, and benchmark",
    )
    parser.add_argument("--download", action="store_true", help="Download from 3DBAG API")
    parser.add_argument("--convert", action="store_true", help="Convert to MuDM")
    parser.add_argument("--tile", action="store_true", help="Generate tiles")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--bbox", type=str, default=_DEFAULT_BBOX,
                        help=f"Bounding box in RD coords (default: {_DEFAULT_BBOX})")
    parser.add_argument("--max-buildings", type=int, default=None, help="Limit buildings")
    parser.add_argument("--lod", type=str, default="2.2", help="Level of detail (default: 2.2)")
    parser.add_argument("--max-zoom", type=int, default=3, help="Max zoom level")
    parser.add_argument("--workers", type=int, default=None, help="Worker processes")
    parser.add_argument("--skip-3dtiles", action="store_true", help="Skip 3D Tiles")
    parser.add_argument("--data-dir", type=Path, default=_DATA_DIR, help="Data directory")
    parser.add_argument("--csv", type=Path, default=None, help="Export CSV")
    args = parser.parse_args()

    if not any([args.download, args.convert, args.tile, args.benchmark]):
        parser.print_help()
        sys.exit(1)

    data_dir = args.data_dir
    tiles_dir = data_dir / "tiles"

    # --- Download ---
    if args.download:
        download_buildings(data_dir, args.bbox, max_buildings=args.max_buildings)

    # --- Convert ---
    collection = None
    if args.convert or args.tile or args.benchmark:
        collection, convert_time = convert_to_microjson(
            data_dir, lod=args.lod, max_buildings=args.max_buildings,
        )

    # --- Tile + Benchmark ---
    if args.tile or args.benchmark:
        tile_and_benchmark(
            collection,
            tiles_dir,
            max_zoom=args.max_zoom,
            workers=args.workers,
            do_tile=args.tile,
            do_benchmark=args.benchmark,
            skip_3dtiles=args.skip_3dtiles,
            csv_path=args.csv,
        )

    print("Done.")


if __name__ == "__main__":
    main()
