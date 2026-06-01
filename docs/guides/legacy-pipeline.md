# Legacy Python Pipeline

The original muDM tiling pipeline is a pure-Python quadtree slicer built around
`TileWriter`, `TileReader`, and the `mudm2vt` intermediate format. It converts a
muDM / MicroJSON `FeatureCollection` into a pyramid of vector tiles (JSON, PBF, or
GeoParquet) and reads them back into a `FeatureCollection`.

!!! warning "This is the legacy path"

    For production workloads, use the **Rust-accelerated pipeline** instead. It is
    dramatically faster (parallel tile encoding, native code) and is the recommended
    approach for all new projects. See **[2D Tiling (Rust)](2d-tiling.md)**.

    The Python pipeline described on this page remains available for backwards
    compatibility, small datasets, debugging, and learning the tiling data model.

## When to use the legacy pipeline

- You want a dependency-light, pure-Python reference implementation you can step
  through in a debugger.
- You are working with small `FeatureCollection`s where raw throughput does not matter.
- You need the JSON intermediate tiles for inspection, or want to read tiles back into
  a muDM `FeatureCollection` with `TileReader`.

If none of these apply, prefer the [Rust 2D pipeline](2d-tiling.md).

## How it fits together

```text
FeatureCollection (.json)
        │
        ▼
 TileWriter.microjson2tiles()        # mudm_tools.tilewriter
        │
        ▼
 mudm2vt() → MuDMVt quadtree slicer  # mudm_tools.mudm2vt.mudm2vt
        │
        ▼
 per-tile encode (JSON | PBF | Parquet)   # chosen by TileWriter flags
        │
        ▼
 tiles/{z}/{x}/{y}.{json|pbf|parquet}
        │
        ▼
 TileReader.tiles2microjson()        # mudm_tools.tilereader → FeatureCollection
```

The tile pyramid is described by a [`TileModel`](../reference/tilejson.md) instance from
the core `mudm` package. Both the writer and the reader take that instance directly.

## Core API

### TileWriter and TileReader

`TileWriter` and `TileReader` are both subclasses of `TileHandler` and **inherit its
constructor**. There is no `tilejson_path` argument — you pass a fully built
`mudm.tilemodel.TileModel` **instance**, plus two independent output-format flags.

```python
class TileHandler:
    def __init__(self, tileobj: TileModel, pbf: bool = False, parquet: bool = False)

class TileWriter(TileHandler): ...
class TileReader(TileHandler): ...
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tileobj` | `mudm.tilemodel.TileModel` | — | Tile pyramid configuration (an instance, not a path). |
| `pbf` | `bool` | `False` | Encode tiles as Mapbox Vector Tile PBF (`vt2pbf`). |
| `parquet` | `bool` | `False` | Encode tiles as GeoParquet (`GeoDataFrame.to_parquet`). |

!!! note "Two format flags, not one"

    Older docs listed only a `pbf` flag. The constructor takes **both** `pbf` and
    `parquet`. The output format is chosen in `microjson2tiles` by precedence:
    if `pbf` is set → PBF; else if `parquet` is set → Parquet; otherwise plain JSON.

After construction, a handler exposes `tile_json` (the `TileModel`), `pbf`, `parquet`,
and internal `id_counter` / `id_set` used for integer id assignment during slicing.

!!! note "One vector layer only"

    `TileWriter` currently uses `self.tile_json.vector_layers[0]` (per an in-source
    `TODO`); only the first vector layer is tiled. Per-layer `minzoom`/`maxzoom` are
    clamped against the global `TileModel` zoom range.

### TileWriter.microjson2tiles

```python
def microjson2tiles(
    self,
    microjson_data_path: Union[str, Path],
    validate: bool = False,
    tolerance_key: str = "default",
) -> List[str]
```

Loads a muDM JSON file, optionally validates it, slices it into vector tiles via
`mudm2vt`, encodes each tile per the handler flags, writes them to the path template in
`tile_json.tiles[0]`, and returns the list of written tile paths.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `microjson_data_path` | `str \| pathlib.Path` | — | Path to the muDM / MicroJSON `FeatureCollection` file. |
| `validate` | `bool` | `False` | If `True`, validates with `MuDM.model_validate` and returns `[]` on a `ValidationError`. |
| `tolerance_key` | `str` | `"default"` | Simplification tolerance-function key passed to `mudm2vt` as `options['tolerance_function']`; must be one of the `AVAILABLE_TOLERANCE_FUNCTIONS` keys. |

**Returns:** `List[str]` — paths to the generated tile files (empty list if validation fails).

!!! tip "Tolerance functions"

    `tolerance_key` selects how aggressively geometry is simplified per zoom level. The
    valid keys are: `default`, `linear`, `constant`, `slow_exponential`, `logarithmic`,
    and `step`. An unknown string raises `ValueError`. See
    [`mudm2vt` options](#the-mudm2vt-intermediate-format) below.

### TileReader.tiles2microjson

```python
def tiles2microjson(self, zlvl: int = 0) -> dict[str, Any]
```

Reads every tile at the given zoom level, reprojects each feature's local tile
coordinates (extent `4096`) back into the global coordinate space defined by
`tile_json.bounds`, and aggregates them into one `FeatureCollection` dict.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `zlvl` | `int` | `0` | Zoom level to read. Returns `{}` if `zlvl` is outside `[minzoom, maxzoom]` or if `bounds` is `None`. |

**Returns:** `dict` of the form `{"type": "FeatureCollection", "features": [...]}` (empty
`{}` on out-of-range zoom or missing bounds).

!!! note "Layer name and decoding"

    `TileReader` decodes PBF via `mapbox_vector_tile.decode` when `pbf=True`, otherwise
    `json.loads`. It expects the layer name `geojsonLayer` inside each decoded tile, and
    reads from the same `{z}/{x}/{y}` template as the writer.

### Helper functions

These live in `mudm_tools.tilewriter` and `mudm_tools.polygen`.

```python
from mudm_tools.tilewriter import getbounds, extract_fields_ranges_enums
from mudm_tools.polygen import generate_polygons
```

#### getbounds

```python
def getbounds(microjson_file: str, square: bool = False) -> List[float]
```

Scans `Polygon` and `MultiPolygon` feature coordinates and returns the bounding box
`[minx, miny, maxx, maxy]`. With `square=True`, expands `maxx`/`maxy` so the box is
square (side = max of width/height, anchored at `minx`/`miny`). `Point` and `LineString`
geometries do not contribute to the bounds.

#### extract_fields_ranges_enums

```python
def extract_fields_ranges_enums(microjson_file: str)
```

Inspects feature properties to derive a TileJSON-style schema. Returns a 3-tuple:

- `field_names: dict[str, str]` — field name → type string. Emitted types: `String`
  (includes `None`), `Boolean`, `Number` (int/float), `Object` (dict), `Array` (list).
- `field_ranges: dict[str, [min, max]]` — numeric field ranges.
- `field_enums: dict[str, set[str]]` — string field enum sets.

!!! warning "field_enums returns Python sets"

    The `field_enums` values are Python `set` objects and are **not** JSON-serializable
    as-is. Convert them to lists (e.g. `sorted(values)`) before dumping to JSON, or rely
    on the `TileLayer` pydantic model to handle serialization for you.

## The mudm2vt intermediate format

The quadtree slicer is `mudm2vt`. Mind the import layout — it is the most common
mistake when using this module:

- The **top-level** `mudm_tools.mudm2vt` resolves to the **factory function**
  (re-exported in `mudm_tools/__init__.py`).
- The `MuDMVt` **class** must be imported from the inner module
  `mudm_tools.mudm2vt.mudm2vt`. The subpackage `mudm_tools.mudm2vt.__init__`
  contains only a license/attribution comment and exports nothing.

```python
# Factory function (top-level convenience re-export)
from mudm_tools import mudm2vt

# The class and helpers (inner module)
from mudm_tools.mudm2vt.mudm2vt import (
    MuDMVt,
    mudm2vt as mudm2vt_factory,
    get_default_options,
    AVAILABLE_TOLERANCE_FUNCTIONS,
)
```

```python
def mudm2vt(data, options, log_level=logging.INFO)  # -> MuDMVt

class MuDMVt:
    def __init__(self, data, options, log_level=logging.INFO)
    def get_tile(self, z, x, y)            # -> dict | None
    def split_tile(self, features, z, x, y, cz=None, cx=None, cy=None)  # -> None
```

`MuDMVt` projects the data, builds per-zoom simplified geometries, and recursively
slices a quadtree of vector tiles. Useful public attributes: `.tiles` (dict id → tile),
`.tile_coords` (list of `{z, x, y}`), `.stats`, and `.total`. `get_tile(z, x, y)` returns
the extent-scaled tile, drilling down from the nearest cached parent when the exact tile
was not pre-sliced (returns `None` for `z < 0` or `z > 24`).

### Default options

`get_default_options()` returns the option dict that user options are merged over:

| Key | Default | Notes |
|-----|---------|-------|
| `maxZoom` | `8` | Must be `0`–`24` (else raises `Exception`). |
| `indexMaxZoom` | `5` | Highest zoom that is pre-sliced. |
| `indexMaxPoints` | `100000` | Point budget before splitting further. |
| `tolerance` | `50` | Base simplification tolerance. |
| `extent` | `4096` | Tile coordinate extent. |
| `buffer` | `64` | Tile buffer (in extent units). |
| `lineMetrics` | `False` | |
| `promoteId` | `None` | Mutually exclusive with `generateId`. |
| `generateId` | `False` | Setting both `promoteId` and `generateId` raises `Exception`. |
| `projector` | `None` | |
| `bounds` | `None` | |
| `tolerance_function` | `default_tolerance_func` | Callable, or a key in `AVAILABLE_TOLERANCE_FUNCTIONS`. |

`AVAILABLE_TOLERANCE_FUNCTIONS` maps the string keys `default`, `linear`, `constant`,
`slow_exponential`, `logarithmic`, `step` to callables of signature `func(z, options) -> float`.

!!! danger "Error semantics"

    - `maxZoom` outside `0`–`24` raises `Exception`.
    - Setting both `promoteId` and `generateId` raises `Exception`.
    - An invalid `tolerance_function` string raises `ValueError`.
    - A non-callable, non-string `tolerance_function` raises `TypeError`.

## Walkthrough: generate, tile, and read back

This end-to-end example mirrors `src/mudm_tools/examples/tiling.py` and
`src/mudm_tools/examples/readtiles.py`. Run the snippets in order from a project checkout.

!!! example "Step 1 — Generate sample polygon data"

    `generate_polygons` writes a random grid of convex polygons to disk and returns the
    `MuDMFeatureCollection`.

    !!! warning "Correct import path"

        Import from `mudm_tools.polygen`, **not** `microjson.polygen`. The demo notebook
        ships a stale `from microjson.polygen import generate_polygons`; that path is
        wrong and will fail.

    ```python
    from mudm_tools.polygen import generate_polygons

    GRID_SIZE = 10000    # overall grid dimension
    CELL_SIZE = 100      # per-cell size (num_cells = GRID_SIZE // CELL_SIZE)
    MIN_VERTICES = 10    # min vertices per polygon
    MAX_VERTICES = 100   # max vertices per polygon

    meta_types = {"num_vertices": "int"}
    meta_values_options = {"polytype": ["Type1", "Type2", "Type3", "Type4"]}

    mudm_data_path = "example_generated.json"

    generate_polygons(
        GRID_SIZE,
        CELL_SIZE,
        MIN_VERTICES,
        MAX_VERTICES,
        meta_types,
        meta_values_options,
        mudm_data_path,
    )
    print(f"Generated polygon data saved to {mudm_data_path}")
    ```

    `generate_polygons` takes its arguments **positionally**, validates the resulting
    `MuDMFeatureCollection`, and writes it via `model_dump_json(indent=2)`.

!!! example "Step 2 — Extract fields, ranges, and enums"

    ```python
    from mudm_tools.tilewriter import extract_fields_ranges_enums

    field_names, field_ranges, field_enums = extract_fields_ranges_enums(mudm_data_path)
    print(field_names)   # {'num_vertices': 'Number', 'polytype': 'String'}
    print(field_ranges)  # {'num_vertices': [10, 24]}
    print(field_enums)   # {'polytype': {'Type1', 'Type2', 'Type3', 'Type4'}}
    ```

!!! example "Step 3 — Define the vector layer"

    Build a [`TileLayer`](../reference/tilejson.md) from the extracted schema. (The
    pydantic model serializes the enum `set` to a JSON array for you.)

    ```python
    from mudm.tilemodel import TileLayer

    vector_layers = [
        TileLayer(
            id="polygon-layer",
            fields=field_names,
            minzoom=0,
            maxzoom=10,
            description="Layer containing polygon data",
            fieldranges=field_ranges,
            fieldenums=field_enums,
        )
    ]
    ```

!!! example "Step 4 — Compute bounds and build the TileModel"

    ```python
    import os
    from pathlib import Path
    from mudm.tilemodel import TileJSON, TileModel
    from mudm_tools.tilewriter import getbounds

    os.makedirs("tiles", exist_ok=True)

    # square=True makes the bounding box square (good for a tile pyramid)
    maxbounds = getbounds(mudm_data_path, square=True)
    center = [0, (maxbounds[0] + maxbounds[2]) / 2, (maxbounds[1] + maxbounds[3]) / 2]

    tile_model = TileModel(
        tilejson="3.0.0",
        tiles=[Path("tiles/{z}/{x}/{y}.pbf")],   # path template; {z}/{x}/{y} are filled in
        name="Example Tile Layer",
        description="A TileJSON example incorporating muDM data",
        version="1.0.0",
        attribution="Polus AI",
        minzoom=0,
        maxzoom=7,
        bounds=maxbounds,
        center=center,
        vector_layers=vector_layers,
    )

    # Optional: persist the TileJSON metadata for viewers / the reader
    tileobj = TileJSON(root=tile_model)
    with open("tiles/metadata.json", "w") as f:
        f.write(tileobj.model_dump_json(indent=2))
    ```

    The `tiles[0]` template is what `TileWriter` formats with each tile's `{z}`, `{x}`,
    and `{y}`. The `.pbf` suffix here matches the `pbf=True` writer flag below — see the
    [output layout](#output-layout).

!!! example "Step 5 — Write the tiles"

    ```python
    from mudm_tools.tilewriter import TileWriter

    writer = TileWriter(tile_model, pbf=True)      # PBF output
    written = writer.microjson2tiles(mudm_data_path, validate=False)
    print(f"Wrote {len(written)} tiles")
    ```

    Swap the flags to change the format:

    === "PBF (Mapbox Vector Tile)"

        ```python
        writer = TileWriter(tile_model, pbf=True)
        writer.microjson2tiles(mudm_data_path)
        ```

    === "GeoParquet"

        ```python
        writer = TileWriter(tile_model, parquet=True)
        writer.microjson2tiles(mudm_data_path)
        ```

    === "JSON (default)"

        ```python
        writer = TileWriter(tile_model)            # no flags → JSON tiles
        writer.microjson2tiles(mudm_data_path)
        ```

!!! example "Step 6 — Read the tiles back into a FeatureCollection"

    This mirrors `src/mudm_tools/examples/readtiles.py`: load the saved metadata,
    rebuild a `TileModel`, and reproject one zoom level back to global coordinates.

    ```python
    import json
    from mudm.tilemodel import TileModel
    from mudm_tools.tilereader import TileReader

    with open("tiles/metadata.json", "r") as f:
        tilejson_data = json.load(f)

    tile_model = TileModel.model_validate(tilejson_data)

    # Match the writer's format flag (pbf=True here because we wrote PBF tiles)
    reader = TileReader(tile_model, pbf=True)
    fc = reader.tiles2microjson(zlvl=0)
    print(fc["type"], len(fc.get("features", [])))
    ```

    !!! tip "Match the format flag"

        Construct `TileReader` with the same `pbf` / `parquet` flag you used for the
        writer, so it decodes the on-disk tiles correctly.

## Running the bundled examples

The reference scripts live under `src/mudm_tools/examples/` and run as modules:

=== "Write tiles"

    ```bash
    # Generate sample data + write tiles into ./tiles
    uv run python -m mudm_tools.examples.tiling

    # Or tile an existing muDM file
    uv run python -m mudm_tools.examples.tiling path/to/data.json
    ```

=== "Read tiles back"

    ```bash
    # Reads tiles/metadata.json + the tile tree, writes microjson_data_read.json
    uv run python -m mudm_tools.examples.readtiles
    ```

!!! note

    `examples/tiling.py` clears and recreates a local `tiles/` directory, then writes
    `tiles/metadata.json` and the PBF tile tree. `examples/readtiles.py` loads
    `tiles/metadata.json`, calls `tiles2microjson(zlvl=0)`, and saves the result to
    `tiles/microjson_data_read.json`.

## Output layout

`TileWriter.microjson2tiles` writes one file per tile using the path template from
`tile_json.tiles[0]`, with `{z}`, `{x}`, `{y}` filled in. The handler flags pick the
file format:

```text
<tiles dir from tile_json.tiles[0] template>/
  {z}/{x}/{y}.json      # default — json.dumps of the tile dict
  {z}/{x}/{y}.pbf       # pbf=True     — vt2pbf-encoded Mapbox Vector Tile
  {z}/{x}/{y}.parquet   # parquet=True — GeoDataFrame.to_parquet (geometry-only)
```

`TileReader` reads back from the same `{z}/{x}/{y}` template.

## Migrating to the Rust pipeline

When you outgrow the legacy path, the [Rust 2D pipeline](2d-tiling.md) is a near-drop-in
replacement: it slices the same `FeatureCollection` data into the same `{z}/{x}/{y}.pbf`
and tiled-Parquet layouts, but in parallel native code. The
[TileJSON reference](../reference/tilejson.md) describes the shared `TileModel` /
`TileLayer` / `TileJSON` configuration models used by both pipelines.

## API reference (autodoc)

::: mudm_tools.tilewriter.TileWriter
    options:
      show_root_heading: true

::: mudm_tools.tilewriter.getbounds
    options:
      show_root_heading: true

::: mudm_tools.tilewriter.extract_fields_ranges_enums
    options:
      show_root_heading: true

::: mudm_tools.tilereader.TileReader
    options:
      show_root_heading: true

::: mudm_tools.tilehandler.TileHandler
    options:
      show_root_heading: true

::: mudm_tools.mudm2vt.mudm2vt.MuDMVt
    options:
      show_root_heading: true

::: mudm_tools.mudm2vt.mudm2vt.mudm2vt
    options:
      show_root_heading: true

::: mudm_tools.mudm2vt.mudm2vt.get_default_options
    options:
      show_root_heading: true

::: mudm_tools.polygen.generate_polygons
    options:
      show_root_heading: true

## See also

- [2D Tiling (Rust)](2d-tiling.md) — the recommended, faster pipeline.
- [TileJSON reference](../reference/tilejson.md) — `TileModel`, `TileLayer`, `TileJSON` models.
