# Format Converters

mudm-tools ships a small **converter registry** that turns common source formats into ready-to-serve muDM tiled output — MVT vector tiles, partitioned Parquet, and (for imaging data) a PNG raster pyramid. Three converters are built in:

| Format | Source | Output | Backend |
|--------|--------|--------|---------|
| `xenium` | 10x Genomics Xenium output bundle | MVT + Parquet + raster pyramid | `StreamingTileGenerator2D` |
| `obj` | Wavefront OBJ mesh files | octree 3D Tiles (GLB/Meshopt) + Parquet | `StreamingTileGenerator` |
| `geojson` | GeoJSON file or directory | quadtree MVT + Parquet | `StreamingTileGenerator2D` |

All three delegate the heavy lifting to the Rust extension `mudm_tools._rs` and to `mudm_tools.tiling2d.generate_pbf`. You drive them through one of two interfaces — the Python `convert()` function or the `python -m mudm_tools.converters.cli` command line.

!!! note "No `mudm convert` console script"
    The CLI module's prog name and in-code docstrings say `mudm convert ...`, but **no `mudm` console script is installed**. The only console entry point in this package is `mudm-serve`. Always invoke the converter CLI as `python -m mudm_tools.converters.cli ...`.

## Quick start

=== "Python"

    ```python
    from mudm_tools.converters import convert, list_formats

    # Discover what's registered
    print(list_formats())  # ['geojson', 'obj', 'xenium']

    # Convert a GeoJSON file to MVT + Parquet
    result = convert(
        "geojson",
        input_dir="annotations.geojson",
        output_dir="tiles/annotations",
        config={"max_zoom": 7, "bounds": (0.0, 0.0, 10000.0, 10000.0)},
    )
    print(result["feature_count"])
    ```

=== "CLI"

    ```bash
    # Discover what's registered
    uv run python -m mudm_tools.converters.cli list-formats

    # Convert a GeoJSON file to MVT + Parquet
    uv run python -m mudm_tools.converters.cli convert \
        --format geojson \
        --input annotations.geojson \
        --output tiles/annotations \
        --max-zoom 7
    ```

## The Python API

Two top-level functions live in `mudm_tools.converters`.

### `convert`

```python
convert(
    format: str,
    input_dir: str,
    output_dir: str,
    config: dict[str, Any] | None = None,
) -> dict
```

Looks up `format` in the registry, instantiates the registered converter class, and calls `converter.convert(input_dir, output_dir, config or {})`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | `str` | — | One of `"xenium"`, `"obj"`, `"geojson"`. |
| `input_dir` | `str` | — | Path to the source data directory or file. |
| `output_dir` | `str` | — | Path for tiled output (parents created as needed). |
| `config` | `dict[str, Any] \| None` | `None` | Converter-specific settings; `None` is coerced to `{}`. |

If `format` is not registered, `convert()` raises `ValueError` with the message
`Unknown format '<format>'. Available: geojson, obj, xenium`.

!!! warning "The return dict shape is converter-dependent"
    There is **no uniform result schema**. All three converters return `total_time` (float seconds) and a `timings` dict, but:

    - **Xenium** returns `layer_counts` (`dict[str, int]`) and `tile_count` (`int`).
    - **OBJ** and **GeoJSON** return `feature_count` (`int`) — and have **no** `layer_counts` or `tile_count`.

    Always read the keys for the converter you called. See the per-converter sections below.

### `list_formats`

```python
list_formats() -> list[str]
```

Returns the sorted list of registered format names. With the three built-in converters this is exactly `['geojson', 'obj', 'xenium']`.

### Autodoc

::: mudm_tools.converters
    options:
      show_root_heading: true
      members:
        - convert
        - list_formats
        - register

## The CLI

The CLI exposes two subcommands.

```text
python -m mudm_tools.converters.cli convert  --format <fmt> -i <in> -o <out> [options]
python -m mudm_tools.converters.cli list-formats
```

### `convert`

| Flag | Alias | Type | Default | Description |
|------|-------|------|---------|-------------|
| `--format` | `-f` | str | *required* | Source format: `xenium`, `obj`, or `geojson`. |
| `--input` | `-i` | str | *required* | Path to source data directory or file. |
| `--output` | `-o` | str | *required* | Path for tiled output. |
| `--config` | `-c` | str | `None` | Path to a JSON config file with converter-specific settings. |
| `--temp-dir` | | str | `None` | Temp directory; injected as `config["temp_dir"]` when set. |
| `--max-zoom` | | int | `None` | Override max zoom; injected as `config["max_zoom"]` when set. |

On success, `convert` prints the result dict as `Result: <pretty JSON>`.

!!! tip "Config precedence"
    The `--config` JSON file is loaded **first**, then `--temp-dir` and `--max-zoom` overwrite the corresponding keys. Use the JSON file for converter-specific keys (`bounds`, `tags`, `point_zoom_offset`, …) that have no dedicated flag.

```bash
# Inline overrides only
uv run python -m mudm_tools.converters.cli convert \
    --format xenium \
    --input data/Xenium_outs \
    --output tiles/xenium_sample \
    --temp-dir /data/tmp \
    --max-zoom 8

# Rich config from a JSON file
uv run python -m mudm_tools.converters.cli convert \
    --format obj \
    --input data/meshes \
    --output tiles/brain \
    --config obj_config.json
```

### `list-formats`

```bash
$ uv run python -m mudm_tools.converters.cli list-formats
  geojson
  obj
  xenium
```

### Autodoc

::: mudm_tools.converters.cli
    options:
      show_root_heading: true

---

## Xenium converter

`XeniumConverter` (registered as `"xenium"`) converts a 10x Genomics Xenium output bundle into a full muDM tiled tree: MVT vector tiles for cells/nuclei polygons and transcripts points, partitioned Parquet, and a PNG raster pyramid built from the DAPI morphology image.

The three layers are fixed:

| Layer | Source file | Geometry | Color |
|-------|-------------|----------|-------|
| `cells` | `cell_boundaries.parquet` | polygon | `#00ffff` |
| `nuclei` | `nucleus_boundaries.parquet` | polygon | `#00ff00` |
| `transcripts` | `transcripts.parquet` | point (`x_location`/`y_location`/`feature_name`) | `#ff4444` |

Missing layer files are skipped with a printed message rather than raising. Polygon layers tile from min zoom `0`; the transcripts layer starts deeper (see `point_zoom_offset`).

!!! danger "Extra dependencies required"
    The Xenium converter's runtime deps are gated behind the `[xenium]` extra: **polars**, **tifffile**, and **pillow**. Install them with:

    ```bash
    pip install mudm-tools[xenium]
    # or, for development:
    uv run --extra xenium pytest
    ```

    These are imported *lazily* inside the converter's methods, so `import mudm_tools.converters.xenium` (and hence `import mudm_tools.converters`) succeeds **without** the extra. You only need it to actually run `.convert()`. Note that `numpy` is imported at module top level, so it is a hard dependency of importing the module (it ships as a core dependency).

### Config keys

All keys are read from `config` via `config.get(...)`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temp_dir` | str | system temp (`tempfile.gettempdir()`) | Temp dir for MVT/Parquet fragments. |
| `max_zoom` | int | derived from image size | Override raster max zoom. Default is `ceil(log2(max(h, w) / 256))`, or `7` if no morphology image. `vector_max_zoom = max_zoom + 1`. |
| `point_zoom_offset` | int | `3` | Transcripts layer min zoom = `max(0, vector_max_zoom - point_zoom_offset)`. |
| `id_column` | str | `"cell_id"` | Boundary polygon ID column for cells and nuclei. |
| `skip_raster` | bool | `false` | Skip raster generation if a `raster/` dir already exists (max_zoom inferred from existing tiles). |

### Return value

```python
{
    "total_time": 412.7,                # float seconds
    "timings": {                        # dict: "raster" + per-layer {ingest, pbf, parquet}
        "raster": 38.1,
        "cells": {"ingest": 12.0, "pbf": 9.4, "parquet": 5.2},
        "nuclei": {"ingest": 11.6, "pbf": 9.1, "parquet": 5.0},
        "transcripts": {"ingest": 80.3, "pbf": 70.2, "parquet": 41.7},
    },
    "layer_counts": {                   # dict[str, int], per present layer
        "cells": 167780,
        "nuclei": 167780,
        "transcripts": 42638083,
    },
    "tile_count": 18421,                # int — merged MVT tiles
}
```

### Example

=== "Python"

    ```python
    from mudm_tools.converters import convert

    result = convert(
        "xenium",
        input_dir="data/Xenium_outs",
        output_dir="tiles/xenium_sample",
        config={
            "temp_dir": "/data/tmp",
            "point_zoom_offset": 3,   # transcripts only at detailed zooms
            "id_column": "cell_id",
            "skip_raster": False,
        },
    )
    print(result["layer_counts"])  # {'cells': 167780, 'nuclei': 167780, 'transcripts': 42638083}
    print(result["tile_count"])
    ```

=== "CLI"

    ```bash
    uv run python -m mudm_tools.converters.cli convert \
        --format xenium \
        --input data/Xenium_outs \
        --output tiles/xenium_sample \
        --temp-dir /data/tmp
    ```

    With a config file for the Xenium-specific keys:

    ```json title="xenium_config.json"
    {
      "temp_dir": "/data/tmp",
      "point_zoom_offset": 3,
      "id_column": "cell_id",
      "skip_raster": false
    }
    ```

    ```bash
    uv run python -m mudm_tools.converters.cli convert \
        --format xenium \
        --input data/Xenium_outs \
        --output tiles/xenium_sample \
        --config xenium_config.json
    ```

### Output structure

```text
output_dir/
  metadata.json            # name, platform, um_per_px, bounds_um, raster{}, vectors{layers}, parquet{}
  gene_list.json           # sorted unique transcript feature names (only if transcripts.parquet present)
  raster/                  # PNG tile pyramid (only if morphology image present / not skipped)
    {z}/{x}/{y}.png        # 256x256 grayscale ("L") DAPI tiles
  vectors/                 # merged multi-layer MVT
    metadata.json          # TileJSON 3.0.0 (vector_layers: cells, nuclei, transcripts)
    {z}/{x}/{y}.pbf
  features.parquet/        # partitioned Parquet
    zoom={z}/<layer>_<part>.parquet
```

The resulting tree is exactly what `mudm-serve` expects. See the [2D Tiling guide](2d-tiling.md) for the tile model and the [TileJSON reference](../reference/tilejson.md) for the `vectors/metadata.json` schema.

### Building a FeatureCollection directly: `xenium_to_mudm`

`mudm_tools.converters.xenium` also exposes a lower-level helper that is **not** part of the registry. Use it when you want an in-memory muDM object instead of tiles.

```python
xenium_to_mudm(
    cell_boundaries_path: Path | str,
    cell_feature_matrix_path: Path | str,
    cells_path: Path | str | None = None,
    cell_type_annotations: Path | str | None = None,
    max_cells: int | None = None,
)  # -> mudm.model.MuDMFeatureCollection
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cell_boundaries_path` | `Path \| str` | — | `cell_boundaries.parquet` (`cell_id`, `vertex_x`, `vertex_y`). |
| `cell_feature_matrix_path` | `Path \| str` | — | The `cell_feature_matrix/` directory (`matrix.mtx.gz` + `barcodes.tsv.gz` + `features.tsv.gz`). A `.zarr.zip` path is accepted only if a sibling `cell_feature_matrix/` dir exists, else raises `ValueError`. |
| `cells_path` | `Path \| str \| None` | `None` | `cells.parquet` summary (centroids/counts/areas). |
| `cell_type_annotations` | `Path \| str \| None` | `None` | 10x `clusters.csv` (`Barcode`, `Cluster`) → `properties["cluster_id"]`. |
| `max_cells` | `int \| None` | `None` | Truncate to the first N cell IDs in sorted order; `None` = all. |

It builds one closed-polygon `MuDMFeature` per cell, stores the per-cell expression vector as a JSON-encoded string under `properties["expression"]` (to survive the `map<utf8,utf8>` Parquet tags schema), and sets collection-level properties `{platform: "xenium", crs: {type: "physical", units: "micrometers"}, gene_panel_dimension, gene_panel}`. Coordinates stay in physical micrometres (Xenium native, not normalized). Cells whose ring has fewer than 4 positions after closure are skipped.

!!! note "Dependencies for `xenium_to_mudm`"
    `xenium_to_mudm` relies only on packages that are already **core** dependencies of `mudm-tools` — `mudm`, `geojson_pydantic`, `pyarrow`, `numpy`, and `scipy` (all declared in `[project].dependencies`) — so it needs **no** extra install beyond the base package. The `[xenium]` extra (`polars`, `tifffile`, `pillow`) is required only by the full `XeniumConverter.convert()` raster / gene-list paths, not by `xenium_to_mudm` itself.

---

## OBJ converter

`ObjConverter` (registered as `"obj"`) converts a directory of Wavefront OBJ meshes into octree-tiled 3D Tiles (GLB/Meshopt) plus optional partitioned Parquet, using `mudm_tools._rs.StreamingTileGenerator`. Ingest is parallelized across files with Rayon, and world bounds are auto-derived via `scan_obj_bounds` when not supplied.

### Config keys

All keys are read from `config` via `config.get(...)`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temp_dir` | str | `None` | Temp dir for fragments (passed straight to the Rust generator; `None` uses the generator's own default). |
| `max_zoom` | int | `4` | Octree max zoom level. |
| `min_zoom` | int | `0` | Octree min zoom level. |
| `bounds` | tuple | auto-scan | World bounds — **6-tuple** `(xmin, ymin, zmin, xmax, ymax, zmax)`. Auto-derived via `scan_obj_bounds` when omitted. |
| `tags` | dict | `{}` | Map of filename-stem → property dict; files with no entry get `{"name": <stem>}`. |
| `glob` | str | `"*.obj"` | Glob pattern for selecting OBJ files inside `input_dir`. |
| `generate_parquet` | bool | `true` | Also emit Parquet (`generate_parquet_native` with `simplify=True`). |

!!! note "Bounds are 3D for OBJ"
    OBJ `bounds` is a **6-tuple** (it includes `zmin`/`zmax`), unlike the 4-tuple used by the GeoJSON converter. Leaving it unset triggers a full bounds scan of every OBJ file.

### Return value

```python
{
    "total_time": 22.4,        # float seconds
    "feature_count": 237,      # int — number of feature ids from add_obj_files
    "timings": {               # dict
        "ingest": 6.1,
        "tiles": 12.8,
        "parquet": 3.5,        # 0 when generate_parquet is False
    },
}
```

There is **no** `layer_counts` or `tile_count` key. If no files match the glob, `convert()` raises `FileNotFoundError`.

### Example

=== "Python"

    ```python
    from mudm_tools.converters import convert

    result = convert(
        "obj",
        input_dir="data/meshes/",
        output_dir="tiles/brain",
        config={
            "max_zoom": 4,
            "min_zoom": 0,
            "temp_dir": "/data/tmp",
            "tags": {
                "neuron_001": {"name": "L5 pyramidal", "region": "MOp"},
                "neuron_002": {"name": "interneuron", "region": "MOp"},
            },
            "generate_parquet": True,
        },
    )
    print(result["feature_count"])
    ```

=== "CLI"

    ```bash
    uv run python -m mudm_tools.converters.cli convert \
        --format obj \
        --input data/meshes \
        --output tiles/brain \
        --max-zoom 4 \
        --temp-dir /data/tmp
    ```

    Per-file `tags` and 3D `bounds` have no dedicated flags — pass them via `--config`:

    ```json title="obj_config.json"
    {
      "max_zoom": 4,
      "min_zoom": 0,
      "bounds": [0.0, 0.0, 0.0, 8192.0, 8192.0, 8192.0],
      "tags": {
        "neuron_001": {"name": "L5 pyramidal", "region": "MOp"}
      },
      "generate_parquet": true
    }
    ```

### Output structure

```text
output_dir/
  3dtiles/                 # octree 3D Tiles (tileset.json + GLB/Meshopt content) from generate_3dtiles
  features.parquet/        # partitioned Parquet (only when generate_parquet=True)
    zoom={z}/...
```

For the full 3D Tiles model, compression options, and viewer, see the [3D Tiling guide](3d-tiling.md).

---

## GeoJSON converter

`GeoJsonConverter` (registered as `"geojson"`) converts a single GeoJSON file *or* a directory of GeoJSON files into quadtree-tiled MVT vector tiles plus partitioned Parquet, using `mudm_tools._rs.StreamingTileGenerator2D` and `mudm_tools.tiling2d.generate_pbf`. A single file is ingested with `add_geojson(text, bounds)`; a directory is ingested with `add_geojson_files([paths], bounds)`.

### Config keys

All keys are read from `config` via `config.get(...)`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temp_dir` | str | system temp (`tempfile.gettempdir()`) | Temp dir for fragments. |
| `max_zoom` | int | `7` | Quadtree max zoom level. |
| `min_zoom` | int | `0` | Quadtree min zoom level. |
| `bounds` | tuple | auto-compute (stub) | World bounds — **4-tuple** `(xmin, ymin, xmax, ymax)`. |
| `layer_name` | str | `"features"` | MVT layer name. |
| `glob` | str | `"*.geojson"` | Glob pattern, only used when `input_dir` is a directory. |

!!! warning "Pass `bounds` explicitly"
    The GeoJSON converter's internal bounds scanner is currently a **stub** — `_update_bounds_from_coords` does nothing — so when `bounds` is omitted the auto-computed bounds fall back to `(0, 0, 1, 1)`. That will mis-tile any real dataset. **Always supply `bounds` explicitly** as a 4-tuple covering your data's world extent.

### Return value

```python
{
    "total_time": 3.9,         # float seconds
    "feature_count": 5821,     # int — len of ids from add_geojson / add_geojson_files
    "timings": {               # dict
        "ingest": 0.8,
        "pbf": 2.4,
        "parquet": 0.7,
    },
}
```

There is **no** `layer_counts` or `tile_count` key. If no GeoJSON files are found, `convert()` raises `FileNotFoundError`.

### Example

=== "Python"

    ```python
    from mudm_tools.converters import convert

    # Single file — supply bounds explicitly!
    result = convert(
        "geojson",
        input_dir="annotations.geojson",
        output_dir="tiles/annotations",
        config={
            "max_zoom": 7,
            "min_zoom": 0,
            "layer_name": "annotations",
            "bounds": (0.0, 0.0, 10000.0, 10000.0),
        },
    )
    print(result["feature_count"])

    # A directory of GeoJSON files
    result = convert(
        "geojson",
        input_dir="data/regions/",
        output_dir="tiles/regions",
        config={"glob": "*.geojson", "bounds": (0.0, 0.0, 10000.0, 10000.0)},
    )
    ```

=== "CLI"

    ```bash
    uv run python -m mudm_tools.converters.cli convert \
        --format geojson \
        --input annotations.geojson \
        --output tiles/annotations \
        --max-zoom 7 \
        --config geojson_config.json
    ```

    `bounds` and `layer_name` have no dedicated flags — pass them via `--config`:

    ```json title="geojson_config.json"
    {
      "max_zoom": 7,
      "min_zoom": 0,
      "layer_name": "annotations",
      "bounds": [0.0, 0.0, 10000.0, 10000.0]
    }
    ```

### Output structure

```text
output_dir/
  vectors/                 # MVT quadtree tiles
    {z}/{x}/{y}.pbf
  features.parquet/        # partitioned Parquet
    zoom={z}/...
```

See the [2D Tiling guide](2d-tiling.md) for the tile model and [TileJSON reference](../reference/tilejson.md) for vector metadata.

---

## Extending the registry

Adding a new converter is the same pattern the built-ins use: write a class with a `convert(self, input_dir, output_dir, config) -> dict` method, decorate it with `@register("name")`, and import the module so registration runs at import time.

```python
from mudm_tools.converters import register

@register("myformat")
class MyConverter:
    def convert(self, input_dir: str, output_dir: str, config: dict) -> dict:
        # ... do work, write tiles to output_dir ...
        return {"total_time": 0.0, "feature_count": 0, "timings": {}}
```

The `register` decorator stores the class in the module-level `_REGISTRY` and returns the class unchanged. Once the module is imported, `list_formats()` will include `"myformat"` and `convert("myformat", ...)` will dispatch to it. The built-in converters register themselves via the `from . import xenium / obj / geojson` lines at the bottom of `mudm_tools/converters/__init__.py`.

## Module reference

::: mudm_tools.converters.xenium
    options:
      show_root_heading: true
      members:
        - XeniumConverter
        - xenium_to_mudm

::: mudm_tools.converters.obj
    options:
      show_root_heading: true
      members:
        - ObjConverter

::: mudm_tools.converters.geojson
    options:
      show_root_heading: true
      members:
        - GeoJsonConverter

## See also

- [2D Tiling guide](2d-tiling.md) — the MVT + Parquet tile model the GeoJSON and Xenium converters produce.
- [3D Tiling guide](3d-tiling.md) — the octree 3D Tiles model the OBJ converter produces.
- [CLI reference](../reference/cli.md) — full command-line reference, including `mudm-serve`.
