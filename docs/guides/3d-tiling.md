# 3D Tiling

muDM represents 3D mesh data with dedicated geometry types and turns it into multi-resolution tile pyramids you can stream to a browser, an ML pipeline, or a Neuroglancer client. This guide covers the full 3D workflow: the 3D geometry types, the two independent tiling engines (the Rust `StreamingTileGenerator` and the pure-Python `TileGenerator3D`), every output format, and the bundled Three.js viewer.

!!! tip "Which engine should I use?"
    There are **two distinct 3D tiling implementations**, and they do **not** share code:

    - **`StreamingTileGenerator`** (Rust, `mudm_tools._rs`) — the disk-streaming, out-of-core path. Ingests OBJ or Parquet files plus world bounds, accumulates on-disk fragment shards, and emits 3D Tiles / PBF3 / Parquet / Neuroglancer. Built for very large datasets (thousands of meshes). Start here for most production work.
    - **`TileGenerator3D`** (pure Python, `mudm_tools.tiling3d`) — the in-memory path driven by a `MuDMFeatureCollection`. Simple `add_features()` → `generate()` → `write_metadata()` flow for moderate datasets. See [Pure-Python alternative: TileGenerator3D](#pure-python-alternative-tilegenerator3d).

    Both produce the same on-disk `{z}/{x}/{y}/{d}.*` tile layout and `tilejson3d.json` / `tileset.json` metadata, so they are interchangeable at the output level — they just take different inputs.

For installation and a first end-to-end run, see [Getting Started](../getting-started.md). For 2D vector tiling see [2D Tiling](2d-tiling.md).

## 3D geometry types

muDM extends GeoJSON with two 3D mesh geometry types based on ISO 19107. Both are accepted on ingest and emitted in tiled output (their integer GeomType codes are `PolyhedralSurface = 4` and `TIN = 5`).

### TIN (Triangulated Irregular Network)

A triangle mesh surface. Each face is a closed ring of exactly four positions (3 vertices + the repeated first vertex). This is the primary geometry type for tiled 3D mesh output.

```json
{
    "type": "Feature",
    "geometry": {
        "type": "TIN",
        "coordinates": [
            [[[0, 0, 0], [10, 0, 0], [5, 8, 3], [0, 0, 0]]],
            [[[10, 0, 0], [10, 8, 0], [5, 8, 3], [10, 0, 0]]],
            [[[0, 0, 0], [0, 8, 0], [5, 8, 3], [0, 0, 0]]]
        ]
    },
    "properties": {
        "featureClass": "neuron",
        "cellType": "pyramidal",
        "region": "CA1"
    }
}
```

### PolyhedralSurface

A closed surface mesh of polygonal faces (not limited to triangles). Each face follows the same structure as a Polygon ring.

```json
{
    "type": "Feature",
    "geometry": {
        "type": "PolyhedralSurface",
        "coordinates": [
            [[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]]],
            [[[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5], [0, 0, 5]]],
            [[[0, 0, 0], [10, 0, 0], [10, 0, 5], [0, 0, 5], [0, 0, 0]]],
            [[[0, 10, 0], [10, 10, 0], [10, 10, 5], [0, 10, 5], [0, 10, 0]]],
            [[[0, 0, 0], [0, 10, 0], [0, 10, 5], [0, 0, 5], [0, 0, 0]]],
            [[[10, 0, 0], [10, 10, 0], [10, 10, 5], [10, 0, 5], [10, 0, 0]]]
        ]
    },
    "properties": {
        "featureClass": "organelle"
    }
}
```

Both types also support a `tiles` field as an alternative to inline `coordinates`, referencing external tiled data (e.g. a 3D Tiles tileset).

## StreamingTileGenerator (Rust)

`StreamingTileGenerator` is a compiled Rust `#[pyclass]`. Accumulate features into on-disk fragment shards (octree-clipped per zoom level), then call one `generate_*` method to emit a format.

!!! danger "Canonical import"
    The compiled extension module is **`mudm_tools._rs`** — never `mudm._rs` (that module does not exist).

    ```python
    from mudm_tools._rs import StreamingTileGenerator
    ```

### Constructor

```python
from mudm_tools._rs import StreamingTileGenerator

gen = StreamingTileGenerator(
    min_zoom=0,        # minimum octree zoom level
    max_zoom=4,        # maximum (finest) octree zoom level
    extent=4096,       # XY tile extent in integer tile-local coords
    extent_z=4096,     # Z tile extent in integer coords
    buffer=0.0,        # tile buffer as a fraction of tile size (normalized)
    base_cells=10,     # grid resolution at zoom 0 (vertex clustering / QEM target)
    temp_dir=None,     # base dir for fragment files; None -> system temp dir
    num_buckets=256,   # bucket count for redistribution
)
```

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `min_zoom` | `int` | `0` | Minimum octree zoom level. |
| `max_zoom` | `int` | `4` | Maximum (finest) octree zoom level. |
| `extent` | `int` | `4096` | XY tile extent in integer tile-local coordinates (used by the PBF3 encoders). |
| `extent_z` | `int` | `4096` | Z tile extent in integer coordinates. |
| `buffer` | `float` | `0.0` | Tile buffer as a fraction of tile size in normalized space. |
| `base_cells` | `int` | `10` | Grid resolution at zoom 0 for vertex clustering / the QEM simplify target. Use `10` for solid regions, `50`–`200` for thin branching structures (neurons). A passed value of `0` is silently replaced by the internal default. |
| `temp_dir` | `str \| None` | `None` | Base dir for intermediate fragment files; defaults to the system temp dir. Fragments live in `{temp_dir}/microjson_frags_{pid}_{genid}/`. Use a large volume for big datasets. |
| `num_buckets` | `int` | `256` | Bucket count for redistribution; a passed `0` is replaced by `256`. |

!!! note "Auto-resolved settings at construction"
    On construction the generator resolves two values automatically:

    - **I/O threads** — `MUDM_IO_THREADS` env var, else `available_parallelism()`, else `8`.
    - **Memory ceiling** — see [memory budget](#memory-budget-auto-detection) below.

    The object implements `Drop`: when it is garbage-collected it removes its temp fragment directory.

### Ingesting OBJ meshes

The `add_obj_*` methods parse OBJ files entirely in Rust and normalize them to `[0,1]³` using the supplied world `bounds`, avoiding Python intermediate objects. A handy companion, `scan_obj_bounds`, computes the combined world bounding box across a list of OBJ files.

=== "Single file"

    ```python
    feature_id = gen.add_obj_file(
        "neuron_001.obj",
        (xmin, ymin, zmin, xmax, ymax, zmax),  # world bbox
        {"cellType": "pyramidal", "region": "CA1"},  # tags
    )
    ```

    All three arguments are required and positional. Degenerate axes (`max == min`) default their span to `1.0`.

=== "Batch (parallel)"

    ```python
    from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

    paths = ["neuron_001.obj", "neuron_002.obj", "neuron_003.obj"]
    bounds = scan_obj_bounds(paths)  # (xmin, ymin, zmin, xmax, ymax, zmax)

    gen = StreamingTileGenerator(max_zoom=5, base_cells=100)

    feature_ids = gen.add_obj_files(
        paths,
        bounds,
        tags_list=[
            {"cellType": "pyramidal"},
            {"cellType": "basket"},
            {"cellType": "pyramidal"},
        ],
        ingest_threads=0,  # 0 = global rayon pool (all cores); N>0 = scoped N-thread pool
    )
    ```

| Method | Signature | Returns |
|--------|-----------|---------|
| `scan_obj_bounds` | `scan_obj_bounds(paths: list[str])` | `(xmin, ymin, zmin, xmax, ymax, zmax)` |
| `add_obj_file` | `add_obj_file(path, bounds, tags)` | `int` — assigned feature ID |
| `add_obj_files` | `add_obj_files(paths, bounds, tags_list, ingest_threads=0)` | `list[int]` — feature IDs of the files that ingested successfully (ascending, deterministic) |

!!! note "Batch ingestion details"
    - `tags_list` must be the same length as `paths` (otherwise `ValueError`). Supported tag value types: `bool`, `str`, `float`, `int` — `None` values are skipped.
    - Files are size-sorted (largest first) for load balancing; large meshes (≥ 500k triangles) use a cascaded pre-simplify for coarse zooms.
    - A file that fails to parse is skipped non-fatally; only the genuinely successful feature IDs are returned. A fatal write error raises `IOError`.

`scan_obj_bounds` is exactly how the bundled OBJ converter derives bounds before ingestion — see [Converters](converters.md) and `src/mudm_tools/converters/obj.py`, the best end-to-end usage example for this subsystem.

### Ingesting from Parquet

For pre-processed mesh data, bypass OBJ parsing and read flat binary geometry columns directly. Each row is one mesh feature; rows are processed in parallel.

```python
count = gen.add_parquet_meshes(
    "meshes.parquet",
    positions_col="positions",   # f32 LE [x,y,z,x,y,z,...]
    indices_col="indices",       # u32 LE [i0,i1,i2,...]
    prop_col=None,               # optional string property column
    prop_name=None,              # output tag name (None -> "name")
    layer_type="meshes",         # stored under the literal tag key 'layer_type'
    bounds=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),  # world bbox for normalization
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `path` | *(required)* | Parquet file path. |
| `positions_col` | `"positions"` | `large_binary`/`binary` column of f32 LE vertex positions. |
| `indices_col` | `"indices"` | `large_binary`/`binary` column of u32 LE triangle indices. |
| `prop_col` | `None` | Optional string property column for per-feature tags. |
| `prop_name` | `None` | Output tag name for that property; **falls back to `"name"`** when `None`. |
| `layer_type` | `"meshes"` | Value written under the literal tag key `layer_type` on **every** feature. |
| `bounds` | `(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)` | World bbox for normalization. |

Returns the number of features ingested. A `('layer_type', layer_type)` tag pair is always added to each feature; if `prop_col` is set, a `(prop_name or 'name', value)` pair is added too.

!!! warning "No more adds after a generate"
    Calling any `generate_*` method takes the fragment writer. Adding features afterward raises `RuntimeError`. Add everything first, then emit.

## Output formats

After ingesting, emit one or more formats. The methods below return integer counts and write directly to disk.

### 3D Tiles (GLB)

OGC 3D Tiles: per-tile `.glb` files plus a `tileset.json` manifest. The reader runs a per-zoom, memory-bounded pipeline (GIL released), encodes each tile, prunes failed tiles, and writes `tileset.json` from the surviving keys.

```python
n_tiles = gen.generate_3dtiles(
    "output/3dtiles",
    (xmin, ymin, zmin, xmax, ymax, zmax),  # world_bounds
    layer_name="default",      # accepted but currently unused
    compression="none",        # "draco" | "meshopt" | anything else -> uncompressed
    use_draco=False,           # backward-compat alias (see below)
    max_concurrent_buckets=8,  # accepted but currently unused
    max_memory_gb=0,           # 0 = auto (see "Memory budget")
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `output_dir` | *(required)* | Output root. Layout: `output_dir/{z}/{x}/{y}/{d}.glb` + `tileset.json`. |
| `world_bounds` | *(required)* | `(xmin, ymin, zmin, xmax, ymax, zmax)`. Degenerate axis spans default to `1.0`. |
| `layer_name` | `"default"` | Accepted for signature compatibility; **currently unused** by this method. |
| `compression` | `"none"` | Selects the GLB encoder — see the exact dispatch below. |
| `use_draco` | `False` | Backward-compat alias — see below. |
| `max_concurrent_buckets` | `8` | Accepted for signature compatibility; **currently unused**. |
| `max_memory_gb` | `0` | Per-output memory ceiling; `0` falls back to the generator's resolved budget. See [memory budget](#memory-budget-auto-detection). |

!!! warning "Exact compression semantics — read carefully"
    The GLB encoder match has **exactly three arms**:

    | `compression` value | Encoder |
    |---------------------|---------|
    | `"draco"` | Draco (`KHR_draco_mesh_compression`) |
    | `"meshopt"` | meshopt (`EXT_meshopt_compression`) |
    | anything else, including `"none"` | uncompressed |

    `use_draco` is a **backward-compat alias only**. Its *sole* effect: if `use_draco=True` **and** `compression == "none"`, the effective compression is promoted to `"draco"`. If `compression` is already set to any non-`"none"` value, `use_draco` is ignored — **`compression` always wins.** They are not independent knobs.

    A single-tile memory-ceiling floor violation raises `MemoryError`.

```python
# Equivalent ways to request Draco:
gen.generate_3dtiles("out", bounds, compression="draco")
gen.generate_3dtiles("out", bounds, use_draco=True)              # promoted (compression was "none")

# use_draco is ignored here — meshopt wins:
gen.generate_3dtiles("out", bounds, compression="meshopt", use_draco=True)
```

#### Compression comparison

| Compression | Encode speed | Browser decode | Lossless |
|-------------|-------------|----------------|----------|
| `"meshopt"` | Fast | Very fast (~1 GB/s) | Yes |
| `"draco"` | Slow | Slow (~50–100 MB/s) | No (quantizes positions) |
| `"none"` | Fastest | N/A | Yes |

!!! tip "Recommendation for web delivery"
    Prefer `meshopt`. The bundled `mudm-serve` server applies Brotli/gzip HTTP transport compression to `.glb` responses on the fly, so meshopt GLBs approach Draco transfer sizes while decoding far faster in the browser. The 3D viewer decodes **both** Draco and meshopt GLBs.

### Tiled Parquet

Columnar output partitioned by zoom level — ideal for ML training pipelines. Two methods are available.

```python
# Streaming-friendly native writer
n_rows = gen.generate_parquet_native(
    "output/parquet",
    bounds,
    simplify=True,        # accepted but inert in this 3D writer (the 2D engine's same-named method DOES honor it)
    compression="zstd",   # "zstd" | "lz4" | "snappy" | anything else -> uncompressed
)

# Fully-Rust partitioned consolidation (one GIL-free region, rotating per-part writers)
n_rows = gen.generate_parquet_native_partitioned(
    "output/parquet",
    bounds,
    compression="zstd",            # "zstd" | "lz4" | "snappy" | else uncompressed
    compression_level=3,           # ZSTD level (only used when compression="zstd")
    max_batch_bytes=2_000_000_000, # uncompressed read-chunk byte budget
    max_file_bytes=500_000_000,    # per-part-file rotation threshold
    overlap_read=True,             # producer/consumer read || transform pipeline
)
```

!!! warning "Parquet codec strings are different from GLB compression"
    For both Parquet methods the accepted `compression` strings are **`"zstd"`, `"lz4"`, `"snappy"`** (anything else → uncompressed). `"draco"` and `"meshopt"` are **not** valid here — those are GLB-only encoders. An invalid ZSTD level raises `ValueError`.

Output layout for both methods:

```text
output_dir/
  zoom={z}/part_NNN.parquet
```

The schema fields are `tile_x` / `tile_y` / `tile_d` (`UInt16`), `feature_id` (`UInt32`), `geom_type` (`UInt8`), `positions` / `indices` (`LargeBinary`), and `tags` (map). For `generate_parquet_native_partitioned`, the `overlap_read` pipeline is ~23% faster on a cold cache and byte-identical; the env vars `MUDM_OVERLAP` / `MUDM_NO_OVERLAP` (disable wins) and `MUDM_OVERLAP_CAP` (in-flight depth, default 1) tune it.

### PBF3

A compact protobuf format for 3D tiles. Two flavors:

```python
# Tile-centric: one .pbf3 per tile
n_tiles = gen.generate_pbf3("output/pbf3", layer_name="neurons")

# Feature-centric: one {feature_id}.pbf3 per feature for O(1) segment retrieval
n_files = gen.generate_feature_pbf3(
    "output/feature_pbf3",
    bounds,
    layer_name="default",  # used only in single-LOD (multilod=False) encoding
    multilod=True,         # each file holds one Layer per zoom with real coarse-LOD reduction
)
```

| Method | Output layout | Notes |
|--------|---------------|-------|
| `generate_pbf3` | `output_dir/{z}/{x}/{y}/{d}.pbf3` | Tile-centric: groups fragments by tile key, transforms to tile-local integers. |
| `generate_feature_pbf3` | `output_dir/{feature_id}.pbf3` + `manifest.json` | Feature-centric. `manifest.json` has `format = mudm_feature_pbf3`, version `2` when `multilod=True` (with a `multilod: true` flag and `world_bounds`), else version `1` (single-LOD, only `max_zoom` fragments). |

### Neuroglancer precomputed

Two methods emit the Neuroglancer precomputed mesh format. The full guide — including sharding, LOD, and serving — lives on its own page.

```python
# Legacy single-resolution mesh (one mesh per feature, at max_zoom only)
n_segments = gen.generate_neuroglancer("output/ng", bounds)

# Multi-resolution Draco mesh (all zoom levels; optional sharding)
n_segments = gen.generate_neuroglancer_multilod(
    "output/ng_multilod",
    bounds,
    vertex_quantization_bits=10,
    sharded=False,   # True -> neuroglancer_uint64_sharded_v1 (holds whole corpus in RAM)
)
```

!!! tip "Full Neuroglancer detail is on its own page"
    Sharding parameters (`minishard_bits`, `shard_bits`), the per-bucket memory bound, info-file contents, and the `/neuroglancer/<id>/` serving route are all covered in **[Neuroglancer Export](neuroglancer.md)**. Use that page for anything Neuroglancer-specific.

### Memory budget (auto-detection)

`max_memory_gb` / `max_memory_bytes` arguments default to `0`, which means **auto**. The resolution order is:

1. An explicit value passed to the method (`> 0` → that many GiB).
2. Otherwise the env var `MUDM_MAX_MEMORY_GB`.
3. Otherwise **0.8 × physical RAM** (cross-platform: macOS via `sysctl hw.memsize`, Linux via `/proc/meminfo`).
4. Otherwise an **8 GiB fallback**.

!!! note
    This is fully cross-platform — the older "auto-detect from `/proc/meminfo` only" description is stale. On macOS the detector reads `sysctl hw.memsize`.

### Bucketed redistribution

For datasets that exceed RAM (thousands of meshes, hundreds of GB of fragments), the pipeline redistributes work into buckets so peak memory stays bounded:

1. **Shard → bucket.** Each ingestion shard is read on its own thread; fragments are hashed by tile key and written to per-shard bucket files — no locks.
2. **Bucket processing.** Each bucket is loaded and processed independently, with tile encoding parallelized via rayon. Memory per bucket is bounded by the resolved budget above.

The constructor's `num_buckets` controls this redistribution knob (default `256`; raise to `512`–`1024` for very large datasets).

!!! note "About `max_concurrent_buckets`"
    `generate_3dtiles`' bucketing is now driven by the per-zoom, memory-bounded pipeline (`max_memory_gb` / the resolved budget). Its `max_concurrent_buckets` argument is accepted for signature compatibility but **currently unused**. The constructor's `num_buckets` is the separate redistribution knob.

### Full example: OBJ folder to 3D Tiles + Parquet

```python
from pathlib import Path
from mudm_tools._rs import StreamingTileGenerator, scan_obj_bounds

paths = [str(p) for p in Path("neurons/").glob("*.obj")]
bounds = scan_obj_bounds(paths)

gen = StreamingTileGenerator(max_zoom=5, base_cells=100, num_buckets=256)
gen.add_obj_files(paths, bounds, tags_list=[{} for _ in paths])

n_tiles = gen.generate_3dtiles("out/3dtiles", bounds, compression="meshopt")
n_rows = gen.generate_parquet_native("out/parquet", bounds, compression="zstd")
print(f"{n_tiles} GLB tiles, {n_rows} Parquet rows")
```

This mirrors `src/mudm_tools/converters/obj.py` — the production OBJ → 3D Tiles
path. The simplest way to run it end-to-end is the [OBJ converter](converters.md)
(`convert("obj", ...)`), which wraps exactly this sequence and also writes the
tiled Parquet sidecar.

!!! note "Example scripts are 2D"
    The bundled `mudm_tools.examples.tiling` / `examples.readtiles` scripts
    demonstrate the **legacy 2D** `TileWriter` pipeline, not 3D meshes — see the
    [Legacy Pipeline guide](legacy-pipeline.md). For the Rust 2D path, see
    `mudm_tools.examples.tiling_rust`.

!!! note "Status helpers"
    `gen.tile_count()` returns the number of tiles/segments written by the last `generate_*` call; `gen.feature_count_val()` returns the number of features added so far.

## Pure-Python alternative: TileGenerator3D

`TileGenerator3D` is a **fully independent, pure-Python** 3D tiling pipeline (`convert → octree index → transform → encode → write`). It is **not** a wrapper around `StreamingTileGenerator` and **not** a "Rust unavailable" fallback — it has no import of the Rust extension. The only place this subsystem touches Rust is the `transform_tile_3d` helper, which is transparently accelerated when the extension is present.

Use it when you already have an in-memory `MuDMFeatureCollection` and want a simple `add_features → generate → write_metadata` flow.

```python
from pathlib import Path
from mudm_tools import TileGenerator3D, OctreeConfig

config = OctreeConfig(
    min_zoom=0,
    max_zoom=4,
    extent=4096,
    extent_z=4096,
)
gen = TileGenerator3D(config=config, output_format="3dtiles")  # or "pbf3"
gen.add_features(collection, layer_name="neurons")   # MuDMFeatureCollection
n_tiles = gen.generate(Path("out/3dtiles"))
gen.write_metadata("out/3dtiles")  # tileset.json (3dtiles) or tilejson3d.json (pbf3)
print(f"{n_tiles} tiles")
```

### TileGenerator3D constructor

```python
TileGenerator3D(
    config: OctreeConfig | None = None,
    output_format: Literal["pbf3", "3dtiles"] = "pbf3",
    workers: int | None = None,
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `config` | `None` | Octree configuration; defaults to `OctreeConfig()`. |
| `output_format` | `"pbf3"` | `"pbf3"` → `.pbf3` tiles + `tilejson3d.json`; `"3dtiles"` → `.glb` + `tileset.json`. |
| `workers` | `None` | Parallel worker processes. `None` or `0` = auto (`os.cpu_count()`); `1` = single-threaded; `N>1` = explicit count. The parallel path only triggers when the tile count is ≥ 16 and `workers > 1` (fork on Unix, spawn on Windows). |

!!! warning "Call `add_features()` first"
    `TileGenerator3D` is stateful. Calling `generate()`, `write_metadata()`, `write_tilejson()`, or `write_tileset_json()` before `add_features()` raises `RuntimeError`. `add_features()` also **overwrites `config.bounds` in place** with the collection's computed world bounds.

### OctreeConfig

```python
OctreeConfig(
    bounds=(0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    min_zoom=0,
    max_zoom=4,
    extent=4096,
    extent_z=4096,
    tolerance=0.0,
    buffer=0.0,
)
```

| Field | Default | Notes |
|-------|---------|-------|
| `bounds` | `(0.0, 0.0, 0.0, 1.0, 1.0, 1.0)` | World bounds `(xmin,ymin,zmin,xmax,ymax,zmax)`. **Overwritten by `add_features()`.** |
| `min_zoom` | `0` | Minimum zoom level to write. |
| `max_zoom` | `4` | Maximum zoom level (octree split depth). |
| `extent` | `4096` | XY tile extent in integer coordinates. |
| `extent_z` | `4096` | Z tile extent in integer coordinates. |
| `tolerance` | `0.0` | Simplification tolerance in normalized space (currently not consumed by the octree build). |
| `buffer` | `0.0` | Tile buffer as a fraction of tile size; when `> 0`, expands clip ranges per octant. |

### Output layout

=== "pbf3"

    ```text
    output_dir/
      tilejson3d.json          # TileModel3D metadata
      {z}/{x}/{y}/{d}.pbf3      # protobuf 3D vector tiles (z >= config.min_zoom)
    ```

=== "3dtiles"

    ```text
    output_dir/
      tileset.json             # OGC 3D Tiles 1.1 manifest
      {z}/{x}/{y}/{d}.glb       # glTF/GLB tile content (z >= config.min_zoom)
    ```

Tile keys are 4-tuples `(z, x, y, d)` where `d` is the depth (3rd spatial) axis of the octree. For `"3dtiles"` output, per-zoom mesh simplification applies: the leaf level (`z == max_zoom`) keeps ratio `1.0`, and each level above keeps `1/(4**levels_above)` of its faces.

### Reading pure-Python tiles

```python
from mudm_tools import TileReader3D

reader = TileReader3D("out/pbf3/tilejson3d.json")  # parent dir is the tile root
layers = reader.read_tile(z=4, x=2, y=3, d=1)       # list[dict] | None
all_at_z = reader.tiles_at_zoom(4)                  # [(z, x, y, d, layers), ...]
collection = reader.tiles2microjson(4)              # rebuild a MuDMFeatureCollection
```

`TileReader3D` reads the `.pbf3` output of `TileGenerator3D`. For OGC 3D Tiles output, use `TileReader3DTiles` from `mudm_tools.tiling3d.reader_3dtiles` (not exported at the top level).

!!! warning "depthsize gotcha"
    `TileModel3D.depthsize` has a Pydantic default of **`256`**, but `TileGenerator3D.write_tilejson()` writes `depthsize = config.extent_z` (default **`4096`**). So an emitted `tilejson3d.json` shows `4096`, not `256` — a common point of confusion when comparing the model default against a generated file.

### Top-level exports vs. submodule imports

These names are re-exported from `mudm_tools` directly:

```python
from mudm_tools import (
    TileGenerator3D, OctreeConfig, TileReader3D, TileModel3D,
    TileEncoding, KnownTileFormat, KnownCompression,
)
```

These live only in submodules (import explicitly):

```python
from mudm_tools.tiling3d.octree import Octree
from mudm_tools.tiling3d.reader_3dtiles import TileReader3DTiles
from mudm_tools.tiling3d.projector3d import CartesianProjector3D
from mudm_tools.tiling3d.reader3d import decode_tile
```

### API reference (autodoc)

::: mudm_tools.tiling3d.generator3d.TileGenerator3D
    options:
      show_root_heading: true

::: mudm_tools.tiling3d.octree.OctreeConfig
    options:
      show_root_heading: true

::: mudm_tools.tiling3d.reader3d.TileReader3D
    options:
      show_root_heading: true

::: mudm_tools.tiling3d.tilejson3d.TileModel3D
    options:
      show_root_heading: true

::: mudm_tools.tiling3d.tilejson3d.TileEncoding
    options:
      show_root_heading: true

For the rest of the pure-Python helpers — `Octree`, `TileReader3DTiles`, `CartesianProjector3D`, `decode_tile`, `convert_collection_3d`, `convert_feature_3d`, `compute_bounds_3d`, `create_tile_3d`, `transform_tile_3d_py` — see the [Python API reference](../reference/python-api.md).

## 3D viewer

The bundled Three.js viewer provides interactive visualization of 3D Tiles output. It uses a **Z-up** coordinate system (`camera.up = (0,0,1)`) matching muDM world coordinates, and decodes both Draco (`KHR_draco_mesh_compression`) and meshopt (`EXT_meshopt_compression`) GLBs.

### Starting the viewer

```bash
mudm-serve --tiles-base out/ --port 8080
# Open http://localhost:8080/
```

`--tiles-base` is the only required flag. When you run with the default `--viewer 3d`, the 2D Leaflet viewer is **also** auto-mounted (best-effort) at `/2d/`. For the full flag reference and directory layout, see [CLI reference](../reference/cli.md#mudm-serve).

The server serves `.glb` tiles with on-the-fly Brotli (quality 5) or gzip (compresslevel 6) `Content-Encoding`, cached per (file, encoding) up to 512 MB total. It also passes through the Neuroglancer precomputed format under `/neuroglancer/<pyramid_id>/`.

!!! note "Internet access for viewer engines"
    The 3D viewer fetches three.js `0.171.0`, the Draco decoder (gstatic `1.5.7`), and the meshopt decoder from CDNs at runtime, so the bundled viewers need network access to load their JS engines.

### Viewer features

- **Level of detail (two modes):** a `dynamic` mode (screen-space-error per-feature zoom) and a `forced` mode where a zoom slider picks a single level (0..maxZoom). Each feature is shown at exactly one zoom; old tiles stay visible during transitions.
- **GPU memory budget slider:** 256–16384 MB, with LRU + stale-tile eviction so the scene respects the budget.
- **Feature selection:** a searchable, filterable checkbox list (from `features.json`) with select-all / clear and per-attribute categorical/numeric filters.
- **Color by attributes:** dropdown auto-discovered from feature properties; categorical (color palette) or numeric (min/max range), with a color legend overlay. "Original" restores per-feature colors.
- **Slice plane:** a global renderer clipping plane with selectable axis (x/y/z) and a flip toggle for examining internal structure.
- **Orbit camera:** Z-up `OrbitControls` with damping.
- **Axis gizmo:** inset X=red, Y=green, Z=blue arrows.
- **Micron scale bar:** adaptive nm/µm/mm bar.
- **Overview panel:** orthographic projection panels; clicking sets a spatial filter cube (with a neighbor "ring" slider) and recenters the main camera.
- **Info panel:** hover/click raycast shows feature properties (name, acronym, cell type, vertex/face counts, …).
- **Background selector:** dark / light / white, persisted in `localStorage`.
- **Screenshot:** PNG download with the scale bar composited in.
- **Keyboard shortcuts:** `R` reset camera, `A` select all visible, `Esc` clear + hide info, `S` screenshot, `F` focus/fit selected.

!!! note "Neuroglancer is server-only"
    The bundled viewer contains no Neuroglancer client code. `/neuroglancer/` is a server passthrough route that serves the precomputed files for an *external* Neuroglancer client. See [Neuroglancer Export](neuroglancer.md).

## Dataset examples

| Dataset | Features | Output | Notes |
|---------|----------|--------|-------|
| MouseLight (38 brains) | ~876K rows | meshopt 3D Tiles | Web delivery via `mudm-serve` |
| Hemibrain (5,000 neurons) | 95 cell types | Tiled Parquet | ML training pipeline; raise `num_buckets` to 512–1024 |

## See also

- [Neuroglancer Export](neuroglancer.md) — full detail on `generate_neuroglancer` / `generate_neuroglancer_multilod`, sharding, and serving.
- [GeoParquet & glTF](geoparquet-gltf.md) — more on the Parquet and GLB outputs.
- [Converters](converters.md) — the high-level OBJ/SWC → tiles conversion paths.
- [2D Tiling](2d-tiling.md) — the 2D vector tiling pipeline and Leaflet viewer.
- [CLI reference](../reference/cli.md) — `mudm-serve` flags.
- [Python API reference](../reference/python-api.md) — every public symbol.
- [TileJSON-3D reference](../reference/tilejson.md) — the `tilejson3d.json` schema.
