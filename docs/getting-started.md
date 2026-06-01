# Getting Started

This quickstart walks you through the three things `mudm-tools` does best, end to end:

- **[A. Tile 2D vector data](#a-tile-2d-vector-data)** — turn GeoJSON into Mapbox Vector Tiles (PBF/MVT) and ML-ready Parquet.
- **[B. Tile 3D meshes](#b-tile-3d-meshes)** — turn Wavefront OBJ meshes into OGC 3D Tiles (GLB).
- **[C. Convert a real dataset](#c-convert-a-real-dataset)** — run a one-call converter on a full bundle (e.g. 10x Xenium).

Then you'll **[serve and view your tiles](#view-your-tiles)** in a browser, and find **[next steps](#next-steps)** into the deeper guides.

!!! note "Prerequisites"
    Make sure `mudm-tools` is installed first — see **[Installation](installation.md)**. The quickest path is `pip install mudm-tools`. The heavy lifting in every example below runs in the compiled Rust extension `mudm_tools._rs`, which ships with the wheel.

!!! tip "Import paths at a glance"
    - The Python package is `mudm_tools`.
    - The compiled Rust extension is `mudm_tools._rs` (this is where `StreamingTileGenerator2D` and `StreamingTileGenerator` live).
    - Example scripts live under `src/mudm_tools/examples/` and run as `python -m mudm_tools.examples.<name>`.

---

## A. Tile 2D vector data

Goal: take a GeoJSON string, clip it through a quadtree, and write vector tiles you can drop into a web map — plus a Parquet pyramid you can train on.

The 2D engine is the `StreamingTileGenerator2D` class from `mudm_tools._rs`. You add features to it, then call an output method. The thin Python helpers in `mudm_tools.tiling2d` wrap those output methods and handle directory creation.

=== "Python"

    ```python
    from mudm_tools._rs import StreamingTileGenerator2D
    from mudm_tools.tiling2d import generate_pbf, generate_parquet, read_pbf

    # A tiny GeoJSON FeatureCollection in world coordinates.
    geojson_str = """
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {"polytype": "Type1"},
          "geometry": {
            "type": "Polygon",
            "coordinates": [[[10, 10], [90, 10], [90, 90], [10, 90], [10, 10]]]
          }
        }
      ]
    }
    """

    # World bounding box: (xmin, ymin, xmax, ymax).
    bounds = (0.0, 0.0, 100.0, 100.0)

    # buffer is in NORMALIZED [0,1] space. 64/4096 mirrors a 64px buffer
    # at the default MVT extent of 4096.
    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64 / 4096)

    # Project to [0,1]^2 against `bounds`, clip through the quadtree, write fragments.
    feature_ids = gen.add_geojson(geojson_str, bounds)
    print(f"Added {len(feature_ids)} features")

    # Write the {z}/{x}/{y}.pbf tree + metadata.json (TileJSON 3.0.0).
    n_tiles = generate_pbf(gen, "tiles/", bounds, simplify=True)
    print(f"Wrote {n_tiles} PBF tiles to tiles/")

    # Read a tile back to verify (returns one dict per feature).
    rows = read_pbf("tiles/", bounds, zoom=0)
    print(f"Zoom 0: {len(rows)} features")
    ```

=== "Example script"

    A complete, runnable version (it can also generate random sample polygons) ships in the repo:

    ```bash
    # Generate sample polygons, tile them, and write Parquet:
    uv run python -m mudm_tools.examples.tiling_rust --max-zoom 6

    # Write PBF vector tiles instead, partitioned by zoom:
    uv run python -m mudm_tools.examples.tiling_rust --pbf --partitioned

    # Tile your own GeoJSON file:
    uv run python -m mudm_tools.examples.tiling_rust my_data.json
    ```

    Source: `src/mudm_tools/examples/tiling_rust.py`.

!!! warning "Constructor default vs. example value"
    The `StreamingTileGenerator2D` constructor's own default `max_zoom` is **4**. The examples here pass `max_zoom=7` deliberately, for a deeper pyramid. Pass the value your data needs — `4` is not a hard cap.

### Key 2D constructor parameters

`StreamingTileGenerator2D` is a compiled Rust class, so its signature is documented here by hand:

```python
StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=None)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `min_zoom` | `int` | `0` | Minimum quadtree zoom level. |
| `max_zoom` | `int` | `4` | Maximum / leaf zoom level. Examples use `7`. |
| `buffer` | `float` | `0.0` | Tile buffer in **normalized** `[0,1]` space (fraction of the full extent, not per-tile). For PBF, examples pass `64/4096`. |
| `temp_dir` | `str` \| `None` | `None` | Base directory for on-disk fragment shards. `None` falls back to the OS temp dir. |

### Writing tiles: PBF vs. Parquet

Both wrappers live in `mudm_tools.tiling2d` and take the generator as their first argument.

```python
from mudm_tools.tiling2d import generate_pbf, generate_parquet

# Mapbox Vector Tiles for web maps: {z}/{x}/{y}.pbf + metadata.json
generate_pbf(gen, "tiles/", bounds, extent=4096, simplify=True,
             layer_name="geojsonLayer")

# Tiled Parquet for ML / analytics (single file by default).
generate_parquet(gen, "features.parquet", bounds, simplify=True)
```

| Helper | Signature (keyword-only after `world_bounds`) | Output |
| --- | --- | --- |
| `generate_pbf` | `generate_pbf(generator, output_path, world_bounds, *, extent=4096, simplify=True, layer_name="geojsonLayer")` | `{z}/{x}/{y}.pbf` tree + `metadata.json` |
| `generate_parquet` | `generate_parquet(generator, output_path, world_bounds, *, compression="zstd", batch_size=50_000, partitioned=False, simplify=True, ...)` | Single `.parquet` file, or a `zoom={z}/part_NNN.parquet` dir when `partitioned=True` |

!!! danger "One generator, one output pass"
    Calling any `generate_*` method **consumes** the generator's fragment writer. After that, `add_geojson` / `add_feature` raise a `RuntimeError`. Build a fresh `StreamingTileGenerator2D` if you need to tile again.

Want the full story — point decimation, Parquet schemas, partitioning, priming for fast reads, reading tiles back? See the **[2D Tiling guide](guides/2d-tiling.md)**.

---

## B. Tile 3D meshes

Goal: take a folder of Wavefront `.obj` meshes and produce OGC **3D Tiles** (`tileset.json` + per-tile `.glb`) you can stream in a WebGL viewer.

The 3D engine is `StreamingTileGenerator` (note: no `2D` suffix), also from `mudm_tools._rs`. The pattern is the same — add meshes, then emit a format — but it works in 3D, so bounds are a **6-tuple** `(xmin, ymin, zmin, xmax, ymax, zmax)`.

```python
from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

obj_paths = ["mesh_a.obj", "mesh_b.obj", "mesh_c.obj"]

# Auto-derive the combined world bounding box from the OBJ files.
bounds = scan_obj_bounds(obj_paths)

gen = StreamingTileGenerator(min_zoom=0, max_zoom=4)

# Parallel (Rayon) ingest. tags_list must be one dict per path.
tags_list = [{"name": p} for p in obj_paths]
feature_ids = gen.add_obj_files(obj_paths, bounds, tags_list)
print(f"Ingested {len(feature_ids)} meshes")

# Emit OGC 3D Tiles: out/3dtiles/tileset.json + {z}/{x}/{y}/{d}.glb
n_tiles = gen.generate_3dtiles("out/3dtiles", bounds, compression="meshopt")
print(f"Wrote {n_tiles} GLB tiles")
```

### Key 3D constructor parameters

`StreamingTileGenerator` is compiled Rust; signature documented by hand:

```python
StreamingTileGenerator(min_zoom=0, max_zoom=4, extent=4096, extent_z=4096,
                       buffer=0.0, base_cells=10, temp_dir=None, num_buckets=256)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `min_zoom` | `int` | `0` | Minimum octree zoom level. |
| `max_zoom` | `int` | `4` | Maximum (finest) octree zoom level. |
| `extent` / `extent_z` | `int` | `4096` | XY / Z tile extent in integer tile-local coords (used by the PBF3 encoders). |
| `buffer` | `float` | `0.0` | Tile buffer as a fraction of tile size (normalized space). |
| `base_cells` | `int` | `10` | Grid resolution at zoom 0 for vertex clustering / simplify target. Use `10` for solid regions, `50`–`200` for thin branching structures (e.g. neurons). |
| `temp_dir` | `str` \| `None` | `None` | Base dir for intermediate fragment files. `None` uses the OS temp dir. |
| `num_buckets` | `int` | `256` | Bucket count for redistribution. |

### Choosing a compression for `generate_3dtiles`

```python
generate_3dtiles(output_dir, world_bounds, layer_name="default",
                 compression="none", use_draco=False,
                 max_concurrent_buckets=8, max_memory_gb=0)
```

The GLB encoder is selected entirely by `compression`:

| `compression` | Effect |
| --- | --- |
| `"meshopt"` | EXT_meshopt_compression GLB (good default for the bundled viewer). |
| `"draco"` | KHR_draco_mesh_compression GLB. |
| `"none"` (or anything else) | Uncompressed GLB. |

!!! note "About `use_draco`"
    `use_draco` is a backward-compat alias with a single effect: when `use_draco=True` **and** `compression="none"`, the encoder is promoted to `"draco"`. A non-`"none"` `compression` always wins. Prefer setting `compression` directly.

`generate_3dtiles` writes:

```text
out/3dtiles/
  tileset.json
  {z}/{x}/{y}/{d}.glb
```

!!! tip "Need other 3D formats?"
    The same generator can also emit native Parquet (`generate_parquet_native`), PBF3, feature-centric PBF3, and Neuroglancer precomputed meshes. See the **[3D Tiling guide](guides/3d-tiling.md)** and the **[Neuroglancer guide](guides/neuroglancer.md)**.

---

## C. Convert a real dataset

Goal: skip the low-level generator entirely and run a one-call converter over a full input bundle. `mudm_tools.converters.convert` dispatches to a registered converter by name.

```python
from mudm_tools.converters import convert, list_formats

print(list_formats())   # ['geojson', 'obj', 'xenium']

result = convert(
    "xenium",
    input_dir="path/to/xenium_output/",
    output_dir="out/xenium/",
    config={
        "max_zoom": 7,        # raster max zoom (vector_max_zoom = max_zoom + 1)
        "id_column": "cell_id",
    },
)
print(result["layer_counts"])   # e.g. {"cells": 12345, "nuclei": 12000, "transcripts": 980000}
print(result["tile_count"])
```

`convert` looks up the format, instantiates the converter, and calls `converter.convert(input_dir, output_dir, config or {})`. An unknown format raises `ValueError` listing the available names.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `format` | `str` | — | One of `"xenium"`, `"obj"`, `"geojson"`. |
| `input_dir` | `str` | — | Source data directory or file. |
| `output_dir` | `str` | — | Output directory (created with parents). |
| `config` | `dict` \| `None` | `None` | Converter-specific settings; `None` becomes `{}`. |

!!! warning "The return dict shape depends on the converter"
    - **`xenium`** returns `total_time`, `timings`, `layer_counts`, and `tile_count`.
    - **`obj`** and **`geojson`** return `total_time`, `feature_count`, and `timings` (no `layer_counts`).

=== "Python (OBJ)"

    ```python
    from mudm_tools.converters import convert

    result = convert(
        "obj",
        input_dir="meshes/",
        output_dir="out/meshes/",
        config={"max_zoom": 4, "generate_parquet": True},
    )
    print(result["feature_count"])
    # Output: out/meshes/3dtiles/  and  out/meshes/features.parquet/
    ```

=== "CLI"

    There is **no** `mudm convert` console script. The converter CLI runs as a module:

    ```bash
    # List the registered formats:
    uv run python -m mudm_tools.converters.cli list-formats

    # Convert an OBJ directory:
    uv run python -m mudm_tools.converters.cli convert \
        --format obj \
        --input meshes/ \
        --output out/meshes/ \
        --max-zoom 4

    # Convert with a JSON config file:
    uv run python -m mudm_tools.converters.cli convert \
        -f geojson -i my_data.geojson -o out/geo/ \
        --config converter_config.json
    ```

!!! note "GeoJSON bounds"
    The GeoJSON converter's automatic bounds computation is currently a stub and falls back to `(0, 0, 1, 1)`. Pass `bounds` explicitly in `config` (a 4-tuple `(xmin, ymin, xmax, ymax)`) for correct tiling. The OBJ converter takes a 6-tuple `bounds` and auto-derives it via `scan_obj_bounds` when omitted.

!!! tip "Xenium needs an extra"
    Running the Xenium converter requires the optional dependencies: `pip install mudm-tools[xenium]`. Importing the module works without them — only `.convert()` needs `polars`, `tifffile`, and `pillow`.

Full per-converter config tables, output layouts, and the lower-level `xenium_to_mudm` helper are in the **[Converters guide](guides/converters.md)**.

---

## View your tiles

`mudm-tools` installs one console script, `mudm-serve`, which serves your tiles **and** a bundled web viewer over HTTP. Open the printed URL in a browser.

=== "3D viewer (default)"

    Point `--tiles-base` at a directory containing `pyramids.json` and per-pyramid subdirs:

    ```bash
    mudm-serve --tiles-base out/
    # Viewer:  http://localhost:8080 (3d)
    ```

    The 3D viewer (Three.js) decodes both Draco and Meshopt GLBs, with level-of-detail streaming, a feature selector, color-by-attribute, slice planes, and a scale bar. When running `--viewer 3d` (the default), the 2D Leaflet viewer is also auto-mounted at `/2d/` if present.

=== "2D viewer"

    Point `--tiles2d-base` at a directory of 2D datasets (each a subdir with `metadata.json`):

    ```bash
    mudm-serve --tiles2d-base tiles/ --viewer 2d
    # Viewer:  http://localhost:8080 (2d)
    ```

    The 2D viewer (Leaflet) overlays multi-layer MVT vectors on a DAPI raster base layer, with per-layer toggles, gene filtering, and a micron scale bar.

### `mudm-serve` flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--tiles-base` | **required** | Directory with `pyramids.json` and 3D tile subdirs. |
| `--tiles2d-base` | `""` | Directory of 2D datasets; empty disables 2D serving. |
| `--viewer` | `3d` | Which bundled viewer to serve at `/` (`3d` or `2d`). |
| `--viewer-dir` | `None` | Custom viewer directory; overrides `--viewer`. |
| `--port` | `8080` | TCP port to bind. |

!!! note "Compression and offline use"
    `mudm-serve` compresses `.glb` responses on the fly (Brotli if the optional `brotli` package is installed, otherwise gzip). Both bundled viewers load their JS engines (Three.js / Leaflet) from CDNs at runtime, so they need internet access on first load.

More on routes, the directory layouts each viewer expects, and Neuroglancer passthrough is in the **[Neuroglancer guide](guides/neuroglancer.md)** and the **[CLI reference](reference/cli.md)**.

---

## Next steps

Now that you've run the full loop, go deeper:

- **[2D Tiling guide](guides/2d-tiling.md)** — `StreamingTileGenerator2D` in depth: GeoJSON & Parquet ingest, point decimation, PBF and Parquet outputs, partitioning, priming, and reading tiles back.
- **[3D Tiling guide](guides/3d-tiling.md)** — `StreamingTileGenerator`: OBJ/Parquet mesh ingest, 3D Tiles, PBF3, and native Parquet, plus memory tuning.
- **[Converters guide](guides/converters.md)** — Xenium, OBJ, and GeoJSON converters end to end, with full config tables and the converter registry extension point.
- **[Neuroglancer guide](guides/neuroglancer.md)** — Precomputed mesh output (legacy and multi-LOD Draco, sharded or loose) and serving it.
- **[GeoParquet & glTF guide](guides/geoparquet-gltf.md)** — interop with the wider geospatial and 3D ecosystems.
- **[OME-NGFF guide](guides/ome-ngff.md)** — bioimaging raster pyramids.
- **[Legacy pipeline guide](guides/legacy-pipeline.md)** — the pre-Rust Python tiling path.

Reference material:

- **[CLI reference](reference/cli.md)** — every flag for `mudm-serve` and the converter CLI.
- **[Python API reference](reference/python-api.md)** — full module and function docs.
- **[TileJSON reference](reference/tilejson.md)** — the `metadata.json` / `tilejson3d.json` schemas.
- **[Roadmap](roadmap.md)** — where the pipeline is headed.
