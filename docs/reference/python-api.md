# Python API Reference

This page is the **map** of the `mudm_tools` public API. Each symbol is fully
documented — with rendered signatures, parameters, and return types — in the
**guide** where it is used; this reference links you straight there. The
Rust-accelerated tiling engines (`mudm_tools._rs`) are hand-documented below,
since they are a compiled extension and cannot be introspected by the autodoc
tooling.

!!! tip "Is the Rust engine available?"
    The tiling engines require the compiled extension. Check at runtime:

    ```python
    import mudm_tools
    print(mudm_tools.RUST_AVAILABLE)  # True when the native extension is installed
    ```

    The pure-Python APIs (GeoParquet/Arrow, glTF, Neuroglancer writers, the
    `TileGenerator3D` pipeline, converters, and the legacy pipeline) work even
    when `RUST_AVAILABLE` is `False`. See [Installation](../installation.md).

## Top-level exports

Everything below is importable directly from the `mudm_tools` package
(`from mudm_tools import ...`):

| Symbol | Kind | Documented in |
| --- | --- | --- |
| `to_geoparquet`, `from_geoparquet`, `to_arrow_table`, `from_arrow_table`, `ArrowConfig` | GeoParquet / Arrow I/O | [GeoParquet & glTF](../guides/geoparquet-gltf.md) |
| `to_glb`, `to_gltf`, `GltfConfig` | glTF / GLB export | [GeoParquet & glTF](../guides/geoparquet-gltf.md) |
| `to_neuroglancer`, `write_annotations` | Neuroglancer export | [Neuroglancer Export](../guides/neuroglancer.md) |
| `TileGenerator3D`, `OctreeConfig`, `TileReader3D`, `TileModel3D`, `TileEncoding`, `KnownTileFormat`, `KnownCompression` | Pure-Python 3D tiling | [3D Tiling](../guides/3d-tiling.md#pure-python-alternative-tilegenerator3d) |
| `mudm2vt` | Legacy intermediate vector tiles | [Legacy Pipeline](../guides/legacy-pipeline.md) |
| `RUST_AVAILABLE` | `bool` flag | this page |

The compiled tiling engines live under `mudm_tools._rs`:

```python
from mudm_tools._rs import StreamingTileGenerator2D, StreamingTileGenerator
```

## Rust tiling engines (`mudm_tools._rs`) { #rust-engines }

These two classes are the high-performance core. They are documented in full —
with worked examples, output layouts, and tuning notes — in the
[2D Tiling](../guides/2d-tiling.md) and [3D Tiling](../guides/3d-tiling.md)
guides. The signatures below are the quick reference.

!!! warning "Requires the native extension"
    `StreamingTileGenerator` and `StreamingTileGenerator2D` are only importable
    when `mudm_tools.RUST_AVAILABLE` is `True`.

### `StreamingTileGenerator2D`

Quadtree-based 2D vector tiling → PBF (MVT) and tiled Parquet. Full guide:
[2D Tiling](../guides/2d-tiling.md).

```python
StreamingTileGenerator2D(min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=None)
```

| Method | Signature (defaults as in source) |
| --- | --- |
| `add_geojson` | `add_geojson(json_str, bounds) -> list[int]` |
| `add_geojson_files` | `add_geojson_files(paths, bounds) -> list[int]` (parallel; no point decimation) |
| `add_feature` | `add_feature(feat) -> int` (feature must be pre-normalized to `[0, 1]²`) |
| `add_parquet_points` | `add_parquet_points(path, x_col, y_col, prop_col, prop_name, layer_type, bounds, coord_scale=1.0) -> int` |
| `add_parquet_polygons` | `add_parquet_polygons(path, id_col, x_col, y_col, layer_type, bounds, coord_scale=1.0) -> int` |
| `generate_pbf` | `generate_pbf(output_dir, world_bounds, extent=4096, simplify=True, layer_name="geojsonLayer") -> int` |
| `generate_parquet_native` | `generate_parquet_native(output_dir, world_bounds, simplify=True, compression="zstd") -> int` (writes a **partitioned directory**) |
| `generate_all` | `generate_all(pbf_dir, parquet_dir, world_bounds, extent=4096, simplify=True, layer_name="geojsonLayer", compression="zstd")` |

!!! note "`buffer` is in normalized space"
    `buffer` is a fraction of the full `[0, 1]` extent, not a per-tile pixel
    count. For PBF output, the examples pass `buffer=64/4096`. The constructor
    default `max_zoom` is **4** — the examples deliberately raise it to 7.

### `StreamingTileGenerator`

Octree-based 3D mesh tiling → 3D Tiles (GLB), tiled Parquet, PBF3, and
Neuroglancer precomputed. Full guide: [3D Tiling](../guides/3d-tiling.md);
Neuroglancer specifics in [Neuroglancer Export](../guides/neuroglancer.md).

```python
StreamingTileGenerator(
    min_zoom=0, max_zoom=4, extent=4096, extent_z=4096,
    buffer=0.0, base_cells=10, temp_dir=None, num_buckets=256,
)
```

| Method | Signature (defaults as in source) |
| --- | --- |
| `add_obj_file` | `add_obj_file(path, bounds, tags) -> int` (all three required, positional) |
| `add_obj_files` | `add_obj_files(paths, bounds, tags_list, ingest_threads=0) -> list[int]` (parallel) |
| `add_parquet_meshes` | `add_parquet_meshes(path, positions_col="positions", indices_col="indices", prop_col=None, prop_name=None, layer_type="meshes", bounds=(0,0,0,1,1,1)) -> int` |
| `generate_3dtiles` | `generate_3dtiles(output_dir, world_bounds, layer_name="default", compression="none", use_draco=False, max_concurrent_buckets=8, max_memory_gb=0) -> int` |
| `generate_parquet_native` | `generate_parquet_native(output_dir, world_bounds, simplify=True, compression="zstd") -> int` |
| `generate_parquet_native_partitioned` | `generate_parquet_native_partitioned(output_dir, world_bounds, compression="zstd", compression_level=3, max_batch_bytes=2_000_000_000, max_file_bytes=500_000_000, overlap_read=True) -> int` |
| `generate_pbf3` | `generate_pbf3(output_dir, layer_name="default") -> int` |
| `generate_feature_pbf3` | `generate_feature_pbf3(output_dir, world_bounds, layer_name="default", multilod=True) -> int` |
| `generate_neuroglancer` | `generate_neuroglancer(output_dir, world_bounds) -> int` |
| `generate_neuroglancer_multilod` | `generate_neuroglancer_multilod(output_dir, world_bounds, vertex_quantization_bits=10, max_memory_bytes=0, sharded=False, minishard_bits=6, shard_bits=0) -> int` |

!!! warning "`generate_3dtiles` compression semantics"
    `compression` selects the GLB encoder with exactly three outcomes:
    `"draco"` → Draco, `"meshopt"` → meshopt, anything else (including the
    default `"none"`) → uncompressed. The legacy `use_draco=True` flag **only**
    promotes to Draco when `compression == "none"`; a non-`"none"` compression
    wins. The Parquet methods take a different codec set —
    `"zstd"` / `"lz4"` / `"snappy"` (else uncompressed), never `draco`/`meshopt`.

## GeoParquet & Arrow

`from mudm_tools import to_geoparquet, from_geoparquet, to_arrow_table, from_arrow_table, ArrowConfig`

Round-trip muDM ↔ Apache Arrow / GeoParquet. The full field reference for
`ArrowConfig` and the round-trip caveats are in the
**[GeoParquet & glTF guide](../guides/geoparquet-gltf.md)**.

## glTF / GLB

`from mudm_tools import to_glb, to_gltf, GltfConfig`

Export muDM 3D geometries to glTF 2.0 / GLB, with optional Draco compression.
The full 13-field reference for `GltfConfig` is in the
**[GeoParquet & glTF guide](../guides/geoparquet-gltf.md)**.

## Neuroglancer

`from mudm_tools import to_neuroglancer, write_annotations`

The Python writers (annotations, skeletons, legacy meshes, segment properties,
and viewer-state/URL builders) and the scalable Rust multi-LOD mesh path are all
covered in the **[Neuroglancer Export guide](../guides/neuroglancer.md)**.
Lower-level helpers live in `mudm_tools.neuroglancer` (e.g. `write_skeleton`,
`write_mesh`, `write_segment_properties`, `viewer_state_to_url`).

## Pure-Python 3D tiling

`from mudm_tools import TileGenerator3D, OctreeConfig, TileReader3D, TileModel3D, TileEncoding, KnownTileFormat, KnownCompression`

An **independent**, pure-Python 3D tiling implementation (not a wrapper or
fallback of the Rust `StreamingTileGenerator`). When to use which, plus the
`TileModel3D.depthsize` gotcha, is explained in the
**[3D Tiling guide](../guides/3d-tiling.md#pure-python-alternative-tilegenerator3d)**.

## 2D tiling helpers

`from mudm_tools.tiling2d import generate_pbf, read_pbf, generate_parquet, read_parquet, prime_parquet, deprime_parquet, repartition_parquet`

High-level wrappers around the Rust 2D engine, plus Parquet priming/repartition
utilities. Documented in the **[2D Tiling guide](../guides/2d-tiling.md)**.

!!! note "Two Parquet schemas"
    `tiling2d.generate_parquet` (single-file mode) emits a 9-column table that
    **includes** a `zoom` column; the partitioned / native paths emit an
    8-column table where zoom is the Hive partition directory instead. `read_pbf`
    results have **no** `indices` key; `read_parquet` results do.

## Converters

`from mudm_tools.converters import convert, list_formats`

The pluggable converter registry (`geojson`, `obj`, `xenium`). Config keys,
result shapes, and output layouts are in the
**[Format Converters guide](../guides/converters.md)**; the command-line form is
in the **[CLI reference](cli.md)**.

## Legacy pipeline & TileJSON models

`from mudm_tools.tilewriter import TileWriter` · `from mudm_tools.tilereader import TileReader` · `from mudm_tools.mudm2vt.mudm2vt import MuDMVt`

The pre-Rust Python tiling pipeline (`TileWriter` / `TileReader` / `mudm2vt`) and
the sample-data helper `mudm_tools.polygen.generate_polygons` are documented in
the **[Legacy Pipeline guide](../guides/legacy-pipeline.md)**. The TileJSON
Pydantic models (`mudm.tilemodel.TileJSON`, `TileModel`, `TileLayer`) live in the
core `mudm` package and are documented in the
**[TileJSON reference](tilejson.md)**.

## See also

- [Getting Started](../getting-started.md) — end-to-end quickstart.
- [Command-Line Reference](cli.md) — `mudm-serve` and the converter CLI.
- [TileJSON for muDM](tilejson.md) — the tileset metadata specification.
