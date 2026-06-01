# 2D Tiling

This guide covers the **Rust-accelerated 2D vector tiling pipeline** in `mudm-tools`. You feed it GeoJSON (or read points/polygons straight out of Parquet), it builds a quadtree from `min_zoom` to `max_zoom`, clips every feature into tiles, and writes the result either as **PBF (Mapbox Vector Tiles)** for web viewers or as **tiled Parquet** for ML pipelines.

The engine is the compiled class [`StreamingTileGenerator2D`](#streamingtilegenerator2d) from `mudm_tools._rs`. Thin Python helpers in `mudm_tools.tiling2d` wrap its output methods and provide readers and a few maintenance utilities.

!!! tip "Where things live"
    - Engine class: `from mudm_tools._rs import StreamingTileGenerator2D` (canonical import).
    - Python helpers/readers: `from mudm_tools.tiling2d import generate_pbf, read_pbf, generate_parquet, read_parquet, ...`
    - Runnable end-to-end example: `python -m mudm_tools.examples.tiling_rust` (source: `src/mudm_tools/examples/tiling_rust.py`).

For the legacy pure-Python tiling modules (`mudm2vt`, `tilewriter`, `tilereader`), see [Legacy pipeline](legacy-pipeline.md). For autodoc API listings, see the [Python API reference](../reference/python-api.md) and the [CLI reference](../reference/cli.md).

---

## Quick start

The fastest way to produce tiles is: construct a generator, add GeoJSON, then call a `generate_*` helper.

=== "PBF (web viewers)"

    ```python
    from mudm_tools._rs import StreamingTileGenerator2D
    from mudm_tools.tiling2d import generate_pbf, read_pbf

    # max_zoom and buffer here are EXAMPLE values, not the API defaults
    # (defaults are max_zoom=4, buffer=0.0). buffer is in normalized [0,1] space:
    # 64 px at MVT extent 4096 => 64/4096.
    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64 / 4096)

    geojson_str = open("data.json").read()
    bounds = (0.0, 0.0, 10000.0, 10000.0)   # world bbox: (xmin, ymin, xmax, ymax)
    gen.add_geojson(geojson_str, bounds)

    # Write the {z}/{x}/{y}.pbf tree + metadata.json under tiles/
    n_tiles = generate_pbf(gen, "tiles/", bounds, simplify=True)
    print(f"Wrote {n_tiles} PBF tiles")

    # Read tiles back (decoded to world coordinates)
    features = read_pbf("tiles/", bounds, zoom=0)
    ```

=== "Parquet (ML pipelines)"

    ```python
    from mudm_tools._rs import StreamingTileGenerator2D
    from mudm_tools.tiling2d import generate_parquet, read_parquet

    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64 / 4096)

    geojson_str = open("data.json").read()
    bounds = (0.0, 0.0, 10000.0, 10000.0)
    gen.add_geojson(geojson_str, bounds)

    # Default (partitioned=False) => a SINGLE .parquet file with a `zoom` column
    n_rows = generate_parquet(gen, "output.parquet", bounds, simplify=True)
    print(f"Wrote {n_rows} rows")

    # Read with optional zoom/tile filtering (predicate pushdown)
    rows = read_parquet("output.parquet", zoom=0)
    ```

=== "Command line"

    ```bash
    # Random polygons -> single-file tiled Parquet (tiles_2d.parquet)
    uv run python -m mudm_tools.examples.tiling_rust

    # Your own GeoJSON, higher zoom, Hive-partitioned output
    uv run python -m mudm_tools.examples.tiling_rust my_data.json \
        --max-zoom 6 --partitioned

    # PBF vector tiles instead of Parquet
    uv run python -m mudm_tools.examples.tiling_rust my_data.json --pbf
    ```

!!! note "How the pipeline works"
    For each feature the generator projects world coordinates into `[0,1]²` against `world_bounds`, clips the geometry through the quadtree for every zoom in `min_zoom..=max_zoom`, and writes `Fragment2D` records to per-process temp shard files (`shard_NNN.mf2d`). The `generate_*` methods then read those fragments back, transform positions to **world coordinates as `f32`**, and encode them as MVT and/or Parquet. Because storage is `f32`, output coordinates are precision-limited.

---

## `StreamingTileGenerator2D`

The streaming quadtree generator. Construct it directly from `mudm_tools._rs`.

```python
from mudm_tools._rs import StreamingTileGenerator2D

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=None)
```

!!! warning "These are the real defaults"
    The constructor defaults are **`min_zoom=0`, `max_zoom=4`, `buffer=0.0`, `temp_dir=None`**. The `max_zoom=7` and `buffer=64/4096` you see throughout the examples are *example* values, not defaults. Any claim that the default `max_zoom` is 7 is wrong — the source default is **4**.

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_zoom` | int | `0` | Minimum quadtree zoom level. |
| `max_zoom` | int | `4` | Maximum / leaf zoom level. |
| `buffer` | float | `0.0` | Tile buffer in **normalized `[0,1]` space** (a fraction of the full extent, *not* per-tile units). For PBF, examples pass `buffer=64/4096`. |
| `temp_dir` | str \| None | `None` | Base directory for fragment shard files; falls back to the OS temp dir. The actual fragment directory is `<temp_dir>/microjson_frags2d_<pid>_<genid>`. |

!!! danger "Add features before you generate"
    Calling any `generate_*` (or the internal `_collect_parquet_data` / `_init_parquet_stream`) method **consumes the single-feature writer**. After that, `add_feature` / `add_geojson` raise `RuntimeError: Cannot add features after generate`. Add everything first, then generate once.

### Geometry-type codes

The `geom_type` field is an integer throughout the subsystem:

| Code | Geometry |
|------|----------|
| `1` | Point |
| `2` | LineString |
| `3` | Polygon |

A GeoJSON `MultiPolygon` is flattened into a single `POLYGON` feature carrying all rings; a `GeometryCollection` is recursed.

---

## Ingestion methods

There are several ways to add features. All ingestion must happen **before** any `generate_*` call.

### `add_geojson`

Parse a GeoJSON string (a `Feature`, `FeatureCollection`, or bare `Geometry`), project each geometry, clip through the quadtree, and return the list of assigned feature ids.

```python
def add_geojson(self, json_str: str, bounds: tuple[float, float, float, float]) -> list[int]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `json_str` | str | GeoJSON text. Supports `Point`, `MultiPoint`, `LineString`, `MultiLineString`, `Polygon`, `MultiPolygon`, `GeometryCollection`. |
| `bounds` | tuple | World bbox `(xmin, ymin, xmax, ymax)` used for normalization. A degenerate axis (`max == min`) uses span `1.0`. |

**Returns:** `list[int]` — the assigned feature ids.

```python
gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7)
fids = gen.add_geojson(open("data.json").read(), (0.0, 0.0, 10000.0, 10000.0))
print(f"Added {len(fids)} features")
```

**Properties become tags** with these type mappings: `String → Str`, integer `Number → Int`, float `Number → Float`, `Bool → Bool`. Arrays, objects, and `null` are skipped.

**Geometry minimums:** LineStrings need ≥ 4 coordinate values (≥ 2 vertices); polygon rings need ≥ 3 vertices.

!!! note "Point decimation"
    Point features are thinned at coarse zooms. At a zoom `tz < max_zoom`, a point is kept **only if `fid % (1 << (max_zoom - tz)) == 0`**. So at `max_zoom` every point survives; one level coarser keeps every other point, and so on. This applies to `add_geojson` and [`add_parquet_points`](#add_parquet_points), but **not** to [`add_geojson_files`](#add_geojson_files). It changes point counts at coarse zooms — expect fewer point rows there.

### `add_geojson_files`

Parallel (rayon) bulk ingest of many GeoJSON files, with the GIL released. Reads, parses, projects, clips, and writes each file across threads using per-thread shard writers.

```python
def add_geojson_files(self, paths: list[str], bounds: tuple[float, float, float, float]) -> list[int]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `paths` | list[str] | GeoJSON file paths. Unreadable or invalid-JSON files are **silently skipped** (yield no features). |
| `bounds` | tuple | World bbox `(xmin, ymin, xmax, ymax)` for normalization. |

**Returns:** `list[int]` — assigned feature ids, **sorted ascending**.

```python
fids = gen.add_geojson_files(
    ["roi_001.geojson", "roi_002.geojson", "roi_003.geojson"],
    (0.0, 0.0, 10000.0, 10000.0),
)
```

!!! warning "No point decimation here"
    Unlike `add_geojson`, this bulk path does **not** apply point decimation — every point is kept at every zoom. Feature ids are assigned from an atomic counter, so id *ordering* is not deterministic across runs, but the returned list is sorted. Internally this closes the single-feature writer and opens `rayon_threads + 1` per-thread shard files.

### `add_feature`

Add a single feature that is **already projected to `[0,1]²`**. No projection is applied — you must pre-normalize coordinates yourself (see [`CartesianProjector2D`](#cartesianprojector2d)).

```python
def add_feature(self, feat: dict) -> int
```

The `feat` dict must contain:

| Key | Type | Description |
|-----|------|-------------|
| `xy` | list[float] | Flat `[x, y, x, y, ...]` in `[0,1]`. |
| `geom_type` | int | `1` point, `2` linestring, `3` polygon. |
| `min_x`, `min_y`, `max_x`, `max_y` | float | Geometry bbox in `[0,1]`. |
| `ring_lengths` | list[int] | *Optional* — for polygons; `None`/absent ⇒ `[]`. |
| `tags` | dict | *Optional* — extracted into feature tags. |

**Returns:** `int` — the assigned feature id (sequential, starting at `0`).

```python
from mudm_tools._rs import StreamingTileGenerator2D, CartesianProjector2D

bounds = (0.0, 0.0, 10000.0, 10000.0)
proj = CartesianProjector2D(bounds)
gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=4)

# A single normalized point at world (2500, 5000)
nx, ny = proj.project(2500.0, 5000.0)
fid = gen.add_feature({
    "xy": [nx, ny],
    "geom_type": 1,
    "min_x": nx, "min_y": ny, "max_x": nx, "max_y": ny,
    "tags": {"layer_type": "markers"},
})
```

### `add_parquet_points`

Read point features directly from a Parquet file via arrow-rs (no JSON intermediary). Each row becomes a `POINT`; coordinates are multiplied by `coord_scale`, normalized against `bounds`, clipped, and written with the GIL released.

```python
def add_parquet_points(
    self, path, x_col, y_col, prop_col, prop_name, layer_type, bounds, coord_scale=1.0
) -> int
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | — | Parquet file path. |
| `x_col`, `y_col` | str | — | Coordinate columns (`Float32` or `Float64`). |
| `prop_col` | str | — | Source column for a string property; if missing/non-string it is ignored. Supports `StringArray`, `BinaryArray` (utf8-decoded), `LargeStringArray`. |
| `prop_name` | str | — | Tag key under which the `prop_col` value is stored. |
| `layer_type` | str | — | Value stored under the **fixed** tag key `'layer_type'`. |
| `bounds` | tuple | — | World bbox `(xmin, ymin, xmax, ymax)` **after** scaling. |
| `coord_scale` | float | `1.0` | Multiply raw coordinates by this before normalization (e.g. `1 / um_per_px`). |

**Returns:** `int` — number of point features added.

```python
count = gen.add_parquet_points(
    "transcripts.parquet",
    "x_location", "y_location",   # coordinate columns
    "feature_name", "gene_name",  # prop_col -> output tag key "gene_name"
    "transcripts",                # stored under tag key "layer_type"
    (0.0, 0.0, 8192.0, 8192.0),   # world bounds (after scaling)
    coord_scale=1.0 / 0.2125,     # microns -> pixels
)
```

Each feature gets the tags `[('layer_type', layer_type)]` plus `('<prop_name>', value)` when `prop_col` is present. Point decimation applies (same rule as `add_geojson`).

### `add_parquet_polygons`

Read polygon features from a Parquet file with a **vertex-per-row** layout. Rows are grouped by `id_col` (preserving first-seen order) into rings, scaled, normalized, clipped, and written.

```python
def add_parquet_polygons(
    self, path, id_col, x_col, y_col, layer_type, bounds, coord_scale=1.0
) -> int
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str | — | Parquet file path. |
| `id_col` | str | — | Polygon identifier column. Accepts `Int32`, `Int64`, `String`, `Binary` (utf8), `LargeString` — stringified. |
| `x_col`, `y_col` | str | — | Vertex coordinate columns (`Float32`/`Float64`). |
| `layer_type` | str | — | Value for the `'layer_type'` tag. |
| `bounds` | tuple | — | World bbox `(xmin, ymin, xmax, ymax)` **after** scaling. |
| `coord_scale` | float | `1.0` | Multiply raw coords before normalization. |

**Returns:** `int` — number of polygons added (groups with ≥ 3 vertices).

```python
count = gen.add_parquet_polygons(
    "cell_boundaries.parquet",
    "cell_id", "vertex_x", "vertex_y",  # id + coordinate columns
    "cells",                            # layer_type tag value
    (0.0, 0.0, 8192.0, 8192.0),
    coord_scale=1.0 / 0.2125,
)
```

!!! danger "Hardcoded tag keys"
    Tags are `[('layer_type', layer_type), ('cell_id', <id string>)]`. The polygon id is stored under the **hardcoded** tag key `'cell_id'` regardless of what you pass for `id_col`. (`add_parquet_points` likewise stores `layer_type` under the fixed key `'layer_type'`.)

!!! warning "Keep each polygon within one row group"
    Grouping happens **per Arrow record batch**. A polygon whose vertex rows straddle two batches is split into separate features. Make sure all rows for a given polygon live within a single row group. Groups with fewer than 3 vertices are dropped. No point decimation applies.

### `feature_count_val`

```python
def feature_count_val(self) -> int
```

Returns the number of features added so far (the current feature-id counter). Reflects the atomic counter after bulk methods complete.

---

## Output methods

After ingesting, call exactly one of these to emit tiles. Each consumes the writer.

### `generate_pbf`

Flush fragments, group by tile, encode each tile as an MVT in parallel, write the `{z}/{x}/{y}.pbf` tree, and write `metadata.json` (TileJSON 3.0.0).

```python
def generate_pbf(
    self, output_dir, world_bounds, extent=4096, simplify=True, layer_name="geojsonLayer"
) -> int
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | — | Root directory for the `{z}/{x}/{y}.pbf` tree. |
| `world_bounds` | tuple | — | `(xmin, ymin, xmax, ymax)`. |
| `extent` | int | `4096` | MVT tile extent. |
| `simplify` | bool | `True` | Apply Douglas-Peucker (polygons via `simplify_polygon_rings`, linestrings via `douglas_peucker`) at zooms `tz < max_zoom`. |
| `layer_name` | str | `"geojsonLayer"` | MVT layer id (also the `vector_layers` id in `metadata.json`). |

**Returns:** `int` — number of `.pbf` tiles written. **Empty tiles are skipped.**

```python
n_tiles = gen.generate_pbf("tiles/", bounds, simplify=True, layer_name="features")
```

The written `metadata.json` is TileJSON 3.0.0 with: `tilejson "3.0.0"`, `tiles ["{z}/{x}/{y}.pbf"]`, `name "MicroJSON Vector Tiles"`, `minzoom`/`maxzoom`, `bounds`, `center [0, cx, cy]`, `vector_layers [{id, fields: {}, minzoom, maxzoom}]`, and `tile_count`. See [TileJSON reference](../reference/tilejson.md) for the full schema.

!!! tip "Prefer the Python wrapper for keyword-only args"
    The [`tiling2d.generate_pbf`](#tiling2dgenerate_pbf) helper does `mkdir -p` for you and exposes `extent`/`simplify`/`layer_name` as keyword-only. The Rust method takes them positionally.

### `generate_parquet_native`

Pure-Rust **partitioned** Parquet writer. Flushes fragments, transforms to world-coordinate `f32` in parallel, and writes one or more part files per zoom under `{output_dir}/zoom={z}/part_NNN.parquet`.

```python
def generate_parquet_native(
    self, output_dir, world_bounds, simplify=True, compression="zstd"
) -> int
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | str | — | Root directory; per-zoom subdirs `zoom={z}/` each contain `part_000.parquet`, … (one part per rayon chunk). |
| `world_bounds` | tuple | — | `(xmin, ymin, xmax, ymax)`. |
| `simplify` | bool | `True` | Douglas-Peucker at coarse zooms. |
| `compression` | str | `"zstd"` | One of `"zstd"`, `"lz4"` (→ `LZ4_RAW`), `"snappy"`; any other string ⇒ `UNCOMPRESSED`. |

**Returns:** `int` — total number of rows written.

!!! warning "This always writes a directory, never a single file"
    `generate_parquet_native` **always** produces a Hive-partitioned directory tree (`zoom={z}/part_NNN.parquet`) regardless of the name you pass. Naming the path `features.parquet/` is misleading — it is a directory, not a single `.parquet` file. For single-file output, use the Python helper [`tiling2d.generate_parquet`](#tiling2dgenerate_parquet) with the default `partitioned=False`.

**Arrow schema (8 columns — NO `zoom` column, since zoom is the partition):**

| Column | Arrow type | Notes |
|--------|-----------|-------|
| `tile_x` | `UInt16` | |
| `tile_y` | `UInt16` | |
| `feature_id` | `UInt32` | |
| `geom_type` | `UInt8` | `1`/`2`/`3` |
| `positions` | `LargeBinary` | `f32` LE world `x,y` pairs |
| `indices` | `LargeBinary` | `u32` LE line-segment index pairs; empty for non-linestrings |
| `ring_lengths` | `List<UInt32>` | |
| `tags` | `Map<Utf8,Utf8>` | all tag values stringified |

### `generate_all`

Read fragments **once**, then run PBF encoding and native partitioned-Parquet writing concurrently (`rayon::join`, GIL released). Equivalent to `generate_pbf` + `generate_parquet_native` sharing one fragment read. Also writes `pbf_dir/metadata.json`.

```python
def generate_all(
    self, pbf_dir, parquet_dir, world_bounds,
    extent=4096, simplify=True, layer_name="geojsonLayer", compression="zstd"
) -> tuple[int, int]
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pbf_dir` | str | — | Root for the `{z}/{x}/{y}.pbf` tree + `metadata.json`. |
| `parquet_dir` | str | — | Root for `zoom={z}/part_NNN.parquet`. |
| `world_bounds` | tuple | — | `(xmin, ymin, xmax, ymax)`. |
| `extent` | int | `4096` | MVT tile extent. |
| `simplify` | bool | `True` | |
| `layer_name` | str | `"geojsonLayer"` | |
| `compression` | str | `"zstd"` | |

**Returns:** `tuple[int, int]` — `(tile_count, parquet_rows)` (u32 PBF tile count, u64 Parquet row count). The Parquet output uses the same 8-column partitioned schema as `generate_parquet_native`; `metadata.json` matches `generate_pbf`.

```python
tile_count, parquet_rows = gen.generate_all(
    "out/vectors", "out/features", bounds, layer_name="features"
)
```

!!! note "Internal streaming protocol"
    The methods `_collect_parquet_data`, `_init_parquet_stream`, `_next_parquet_batch`, and `_close_parquet_stream` (all leading-underscore) implement the streaming protocol consumed by `tiling2d.parquet_writer`. They are not intended for direct end-user calls — use [`tiling2d.generate_parquet`](#tiling2dgenerate_parquet) instead.

---

## `CartesianProjector2D`

Normalizes 2D world coordinates to `[0,1]²` and back. Use it to pre-normalize coordinates for [`add_feature`](#add_feature), which expects `[0,1]` input.

```python
from mudm_tools._rs import CartesianProjector2D

proj = CartesianProjector2D((0.0, 0.0, 10000.0, 10000.0))   # (xmin, ymin, xmax, ymax)

nx, ny = proj.project(2500.0, 5000.0)     # world -> normalized [0,1]^2
x, y = proj.unproject(nx, ny)             # normalized -> world
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `project` | `project(self, x: float, y: float) -> tuple[float, float]` | normalized `(nx, ny)` |
| `unproject` | `unproject(self, nx: float, ny: float) -> tuple[float, float]` | world `(x, y)` |

A degenerate axis (`max == min`) uses span `1.0`, so `project` returns `0.0` on that axis.

---

## Python helper functions (`mudm_tools.tiling2d`)

These wrap the engine's output methods, add the readers, and provide partition-maintenance utilities. All are importable directly from `mudm_tools.tiling2d`.

```python
from mudm_tools.tiling2d import (
    generate_pbf, read_pbf,
    generate_parquet, read_parquet,
    prime_parquet, deprime_parquet, repartition_parquet,
)
```

!!! note "Importing the engine class"
    `tiling2d.__init__` imports `StreamingTileGenerator2D` and `CartesianProjector2D` from `mudm_tools._rs` but does **not** list them in `__all__`. `from mudm_tools.tiling2d import StreamingTileGenerator2D` happens to work, but the canonical, documented import is `from mudm_tools._rs import StreamingTileGenerator2D`.

### `tiling2d.generate_pbf`

```python
def generate_pbf(
    generator, output_path, world_bounds, *,
    extent=4096, simplify=True, layer_name="geojsonLayer"
) -> int
```

Thin wrapper: `mkdir -p` the output dir, then call `generator.generate_pbf(...)`. Returns the number of tiles written. `extent`, `simplify`, and `layer_name` are **keyword-only** here (unlike the positional Rust method).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generator` | `StreamingTileGenerator2D` | — | A generator with features added. |
| `output_path` | str \| Path | — | Directory for the `{z}/{x}/{y}.pbf` tree. |
| `world_bounds` | tuple | — | `(xmin, ymin, xmax, ymax)`. |
| `extent` | int | `4096` | MVT extent (kw-only). |
| `simplify` | bool | `True` | (kw-only) |
| `layer_name` | str | `"geojsonLayer"` | (kw-only) |

### `tiling2d.read_pbf`

```python
def read_pbf(path, world_bounds, *, zoom=None, tile_x=None, tile_y=None) -> list[dict]
```

Walk the `{z}/{x}/{y}.pbf` tree, decode each MVT tile, convert tile-local integers back to world `f64` coordinates, and return one dict per feature, sorted by `(z, x, y)`. Returns `[]` if `path` is not a directory.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str \| Path | — | Directory containing the `{z}/{x}/{y}.pbf` tree. |
| `world_bounds` | tuple | — | The `(xmin, ymin, xmax, ymax)` used at generation time. |
| `zoom`, `tile_x`, `tile_y` | int \| None | `None` | Optional filters (kw-only). |

**Output dict keys:** `zoom`, `tile_x`, `tile_y`, `feature_id`, `geom_type`, `positions` (numpy `float32` `[N, 2]`), `ring_lengths`, `tags`.

!!! warning "`read_pbf` has no `indices` key"
    The MVT reader does **not** reconstruct line-segment indices. `read_pbf` dicts contain no `'indices'` key. If you need `indices`, read from Parquet instead (see [`read_parquet`](#tiling2dread_parquet)).

```python
features = read_pbf("tiles/", bounds, zoom=0)
if features:
    f = features[0]
    print(f["geom_type"], f["positions"].shape, f["tags"])
```

### `tiling2d.generate_parquet`

```python
def generate_parquet(
    generator, output_path, world_bounds, *,
    compression="zstd", compression_level=3, batch_size=50_000,
    partitioned=False, max_file_bytes=500_000_000,
    max_batch_bytes=2_000_000_000, simplify=True
) -> int
```

Writes tiled Parquet from a `StreamingTileGenerator2D`. It selects one of three paths: in-memory (if the generator lacks `_init_parquet_stream`), **single-file streaming** (the default), or **partitioned streaming** (`partitioned=True`). Returns rows written.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generator` | `StreamingTileGenerator2D` | — | Generator with fragments added. |
| `output_path` | str \| Path | — | A single `.parquet` file, **or** a directory if `partitioned=True`. |
| `world_bounds` | tuple | — | `(xmin, ymin, xmax, ymax)`. |
| `compression` | str | `"zstd"` | (kw-only) |
| `compression_level` | int | `3` | (kw-only) |
| `batch_size` | int | `50_000` | Fragments per streaming batch (kw-only). |
| `partitioned` | bool | `False` | One Hive `zoom={z}/part_NNN.parquet` tree (no `zoom` column) vs a single sorted-by-zoom file (with `zoom` column). |
| `max_file_bytes` | int | `500_000_000` | Rotating part-file size budget (partitioned mode). |
| `max_batch_bytes` | int | `2_000_000_000` | Per-batch byte budget. |
| `simplify` | bool | `True` | (kw-only) |

```python
# Single file (default): includes a leading `zoom` column
n = generate_parquet(gen, "output.parquet", bounds)

# Hive-partitioned directory: zoom is the partition, no `zoom` column
n = generate_parquet(gen, "output_dir/", bounds, partitioned=True)
```

!!! danger "Single-file and native are DIFFERENT schemas"
    `tiling2d.generate_parquet` (single-file mode, the default) writes a **9-column** table that *includes* a leading `zoom` `UInt8` column. The Rust [`generate_parquet_native`](#generate_parquet_native) method — and `generate_parquet` in `partitioned=True` mode — write an **8-column** table with **no** `zoom` column (zoom is encoded in the directory name `zoom={z}`). Do not conflate them. See [Output structures](#output-structures) below.

### `tiling2d.read_parquet`

```python
def read_parquet(path, *, zoom=None, feature_id=None, tile_x=None, tile_y=None) -> list[dict]
```

Reader for tiled 2D Parquet (single file or Hive-partitioned dir). Uses PyArrow dataset predicate pushdown for the optional filters and decodes binary columns into numpy arrays. For a partitioned dir it auto-detects Arrow IPC (`.arrow`) siblings (the "primed" fast path) vs `.parquet`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | str \| Path | — | `.parquet` file or partitioned directory. |
| `zoom`, `feature_id`, `tile_x`, `tile_y` | int \| None | `None` | Optional filters, combined with AND (kw-only). |

**Output dict keys:** `zoom`, `tile_x`, `tile_y`, `feature_id`, `geom_type`, `positions` (np `float32` `[N, 2]`), `indices` (np `uint32` `[M]`), `ring_lengths` (list[int]), `tags` (dict[str, str]).

!!! note "`read_parquet` includes `indices`"
    Unlike `read_pbf`, `read_parquet` dicts **do** contain an `'indices'` key (numpy `uint32`).

```python
rows = read_parquet("output.parquet", zoom=0)
# Or filter a partitioned dir by tile + feature:
rows = read_parquet("output_dir/", zoom=3, tile_x=2, tile_y=5)
```

---

## Partition maintenance utilities

These operate on a Hive-partitioned Parquet pyramid (`zoom={z}/...`). All raise `FileNotFoundError` / `NotADirectoryError` on a bad path.

### `prime_parquet`

```python
def prime_parquet(path, *, compression="uncompressed") -> int
```

Convert each `zoom={z}/*.parquet` partition file to a sibling Arrow IPC (`.arrow`) file — the `read_parquet` fast path. Drops the `zoom` column if present. Returns the number of `.arrow` files written. `compression` (kw-only) is one of `{"uncompressed", "lz4", "zstd"}` (else `ValueError`).

### `deprime_parquet`

```python
def deprime_parquet(path) -> int
```

Delete all `zoom={z}/*.arrow` IPC sibling files from a partitioned pyramid. Returns the number deleted.

### `repartition_parquet`

```python
def repartition_parquet(
    path, *, max_file_bytes=500_000_000, compression="zstd", compression_level=3
) -> dict[int, int]
```

Split oversized per-zoom partition files into uniformly named `part_NNN.parquet` files capped at `max_file_bytes` (uncompressed binary). Skips zoom dirs already correctly named and small. Drops the `zoom` column, removes `.arrow` siblings, and writes via temp files then renames. Returns `{zoom: num_parts}`.

```python
from mudm_tools.tiling2d import prime_parquet, deprime_parquet, repartition_parquet

# Re-balance part files, then add IPC fast-path siblings
repartition_parquet("output_dir/", max_file_bytes=250_000_000)
n_arrow = prime_parquet("output_dir/")          # now read_parquet uses the .arrow files
# ... later, to reclaim space:
deprime_parquet("output_dir/")
```

---

## Output structures

### PBF (MVT) — `generate_pbf` / `generate_all`

```text
output_dir/
  metadata.json          # TileJSON 3.0.0 (tilejson, tiles=["{z}/{x}/{y}.pbf"],
                         #   minzoom, maxzoom, bounds, center, vector_layers, tile_count)
  {z}/
    {x}/
      {y}.pbf            # one MVT per non-empty tile; layer id = layer_name
                         #   (default "geojsonLayer"); MVT extent = extent (default 4096)
```

### Tiled Parquet — partitioned (8-col, no `zoom` column)

Produced by the Rust `generate_parquet_native`, by `generate_all`, and by `generate_parquet(partitioned=True)`:

```text
output_dir/
  zoom=0/
    part_000.parquet
    part_001.parquet     # native: one part per rayon chunk;
                         #   python partitioned: rotated by max_file_bytes
  zoom=1/
    part_000.parquet
  ...
```

Columns: `tile_x`(u16), `tile_y`(u16), `feature_id`(u32), `geom_type`(u8), `positions`(large_binary, `f32` LE world x,y pairs), `indices`(large_binary, `u32` LE seg-index pairs; empty for non-linestrings), `ring_lengths`(list&lt;u32&gt;), `tags`(map&lt;utf8,utf8&gt;).

### Tiled Parquet — single file (9-col, with `zoom` column)

Produced by `generate_parquet(partitioned=False)`, the default. One file, sorted by zoom (one row group per zoom):

```text
output.parquet           # single file
```

Same columns as above **plus** a leading `zoom`(u8) column.

### Primed fast path

`prime_parquet` adds sibling Arrow IPC files inside each partition:

```text
output_dir/
  zoom=0/
    part_000.parquet
    part_000.arrow       # written by prime_parquet; removed by deprime_parquet
```

---

## End-to-end example

The repository ships a complete, runnable script at `src/mudm_tools/examples/tiling_rust.py`. It generates random polygons with `polygen`, builds a `StreamingTileGenerator2D`, adds them with `add_geojson`, then writes and reads either Parquet or PBF.

```bash
# Default: single-file Parquet at tiles_2d.parquet, max-zoom 7
uv run python -m mudm_tools.examples.tiling_rust

# Hive-partitioned Parquet, lower max-zoom
uv run python -m mudm_tools.examples.tiling_rust --max-zoom 6 --partitioned

# PBF vector tiles (writes to tiles/ when the output path ends in .parquet)
uv run python -m mudm_tools.examples.tiling_rust my_data.json --pbf
```

The script accepts `--min-zoom`, `--max-zoom` (default `7`), `--output`, `--partitioned`, `--pbf`, `--no-simplify`, `--buffer` (pixels at extent 4096, default `64`), `--grid-size`, and `--cell-size`. It converts the pixel buffer to normalized space with `buffer = args.buffer / 4096.0` before constructing the generator.

!!! note "Run modules, not file paths"
    Run the example as a module: `python -m mudm_tools.examples.tiling_rust`. The script's own docstring shows a stale `src/mudm/examples/...` path — the real location is `src/mudm_tools/examples/tiling_rust.py`.

---

## See also

- [Python API reference](../reference/python-api.md) — autodoc listings for the `mudm_tools.tiling2d` package and helpers.
- [CLI reference](../reference/cli.md) — `mudm-serve` and the converter CLI (`python -m mudm_tools.converters.cli`).
- [TileJSON reference](../reference/tilejson.md) — the `metadata.json` schema written alongside PBF tiles.
- [3D tiling](3d-tiling.md) — the octree-based pipeline for OBJ meshes.
- [Converters](converters.md) — Xenium / OBJ / GeoJSON converters that drive this pipeline end to end.
- [Legacy pipeline](legacy-pipeline.md) — the original pure-Python `mudm2vt` / `tilewriter` / `tilereader` modules.

## API reference (autodoc)

::: mudm_tools.tiling2d
    options:
      show_root_heading: true

::: mudm_tools.tiling2d.generate_pbf
    options:
      show_root_heading: true

::: mudm_tools.tiling2d.read_pbf
    options:
      show_root_heading: true

::: mudm_tools.tiling2d.generate_parquet
    options:
      show_root_heading: true

::: mudm_tools.tiling2d.read_parquet
    options:
      show_root_heading: true

::: mudm_tools.tiling2d.prime_parquet
    options:
      show_root_heading: true

::: mudm_tools.tiling2d.deprime_parquet
    options:
      show_root_heading: true

::: mudm_tools.tiling2d.repartition_parquet
    options:
      show_root_heading: true
