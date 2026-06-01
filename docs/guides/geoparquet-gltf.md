# GeoParquet & glTF

This guide covers two interchange formats for muDM feature collections:

- **GeoParquet / Arrow** — a columnar, analytics-friendly format for storing features as a table with a WKB geometry column. Ideal for round-tripping muDM through the broader geospatial and data-science ecosystem (pyarrow, GeoPandas, DuckDB).
- **glTF / GLB** — the standard 3D scene format for the web. Export muDM geometry to a `.gltf` (JSON, base64-embedded buffers) or `.glb` (binary, optionally Draco-compressed) file you can drop into any glTF viewer.

Both live in `mudm_tools` and operate on `MuDMFeature` / `MuDMFeatureCollection` objects from the core `mudm` package. The four GeoParquet/Arrow functions and the two glTF functions, plus their config models, are all re-exported at the top level:

```python
from mudm_tools import (
    to_geoparquet, from_geoparquet,
    to_arrow_table, from_arrow_table, ArrowConfig,
    to_glb, to_gltf, GltfConfig,
)
```

!!! tip "Where the data comes from"
    These exporters take an already-built muDM collection. If you need to *build* one first — e.g. from a DataFrame — see the `df_to_microjson` example script (`python -m mudm_tools.examples.df_to_microjson`) or load and validate an existing file. To turn meshes into a tiled 3D scene instead of a single file, see the [3D Tiling guide](3d-tiling.md).

---

## GeoParquet & Arrow

### Export a collection to GeoParquet

`to_geoparquet` builds an Arrow table from your features and writes it to a single `.parquet` file. Parent directories are created automatically, and the written `pyarrow.Table` is returned (not `None`), which is handy if you want to inspect it.

```python
from mudm_tools import to_geoparquet, ArrowConfig

# `fc` is a MuDMFeatureCollection (or a single MuDMFeature)
table = to_geoparquet(fc, "output.parquet")

# Optionally customise the geometry column name
table = to_geoparquet(fc, "output.parquet", ArrowConfig(primary_geometry_column="geom"))
```

The resulting schema has three reserved columns followed by one column per distinct feature property:

```text
id            : string        (str(feature.id) or null)
featureClass  : string
<geometry>    : binary (WKB)   (column name = ArrowConfig.primary_geometry_column, default "geometry")
<prop columns>: bool | int64 | float64 | string   (one per distinct property, inferred type)
```

GeoParquet 1.1 metadata is attached to the schema under the `b"geo"` key (containing `version`, `primary_column`, and per-column `encoding`/`geometry_types`/`bbox`). If the collection carries vocabularies, they are stored under `b"mudm:vocabularies"`.

!!! note "Property type inference"
    Each property becomes one Arrow column whose type is inferred from its values:

    | Values across rows | Arrow column type |
    |--------------------|-------------------|
    | all `bool` | `bool` |
    | all `int` (non-bool) | `int64` |
    | all numeric (mix of int/float) | `float64` |
    | all `str` | `string` |
    | mixed / `dict` / `list` | JSON-serialized into a `string` column |

    3D geometries record a ` Z` suffix in the `geo` metadata (e.g. `"LineString Z"`). muDM `TIN` and `PolyhedralSurface` geometries are converted to a Shapely `MultiPolygon` before WKB encoding.

### Read GeoParquet back into muDM

`from_geoparquet` reads the file with `pyarrow.parquet.read_table` and reconstructs a `MuDMFeatureCollection`. The geometry column is auto-detected from the GeoParquet `geo` metadata's `primary_column` (falling back to `"geometry"`), so you usually don't pass `geometry_column` at all.

```python
from mudm_tools import from_geoparquet

fc = from_geoparquet("output.parquet")

# Override the auto-detected geometry column if needed
fc = from_geoparquet("output.parquet", geometry_column="geom")
```

Columns other than `id`, `featureClass`, and the geometry column are restored as each feature's `properties`. Collection vocabularies are read back from `b"mudm:vocabularies"` if present.

### Working with the in-memory Arrow table

If you want the `pyarrow.Table` directly — to hand off to DuckDB, GeoPandas, or your own pipeline — use `to_arrow_table` / `from_arrow_table` instead of touching disk.

```python
from mudm_tools import to_arrow_table, from_arrow_table

table = to_arrow_table(fc)            # build the table, no file written
fc2 = from_arrow_table(table)         # reconstruct the collection
```

!!! warning "Different `geometry_column` defaults"
    `from_arrow_table` defaults `geometry_column` to the literal string `"geometry"`, whereas `from_geoparquet` defaults it to `None` (auto-detect from `geo` metadata). If your table uses a non-default geometry column name and you read it with `from_arrow_table`, pass the column name explicitly:

    ```python
    fc2 = from_arrow_table(table, geometry_column="geom")
    ```

### Round-trip caveats

A round trip is not always byte-for-byte identical at the geometry-type level. Keep these in mind:

!!! danger "Geometry and id round-trip behaviour"
    - **3D triangle-only MultiPolygon → TIN.** On read, a 3D `MultiPolygon` whose sub-polygons are all closed triangles (4 coordinates, no holes) is reconstructed as a muDM `TIN`, not a `MultiPolygon`.
    - **PolyhedralSurface is not preserved.** On write it is encoded as a `MultiPolygon`; on read it is *never* reconstructed as a `PolyhedralSurface` (only `TIN`/`MultiPolygon`), so it round-trips back as a `MultiPolygon`.
    - **Feature ids are stringified.** `id` is written as `str(feature.id)` (or `null`). A non-string id (e.g. an integer) will not round-trip back to its original type — it returns as a string.
    - Unsupported Shapely geometry types raise `TypeError` on read.

### `ArrowConfig` reference

`ArrowConfig` is a Pydantic model with exactly **one** field:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `primary_geometry_column` | `str` | `"geometry"` | Name of the WKB geometry column, used both in the table schema and in the GeoParquet `geo` metadata `primary_column`. |

```python
from mudm_tools import ArrowConfig

ArrowConfig()                                        # primary_geometry_column="geometry"
ArrowConfig(primary_geometry_column="geom")
```

::: mudm_tools.arrow.models.ArrowConfig
    options:
      show_root_heading: true

---

## glTF / GLB export

### Export a binary GLB

`to_glb` is the recommended path for most 3D exports: it produces a compact binary `.glb` and is the only path that supports Draco compression (Draco data is appended to the GLB binary buffer). It always returns the GLB content as `bytes`, and writes the file if `output_path` is given (parent directories are created).

```python
from mudm_tools import to_glb, GltfConfig

# Write a .glb file (and capture the bytes)
glb_bytes = to_glb(fc, "output.glb")

# In-memory only (no file written)
glb_bytes = to_glb(fc)
```

### Export a text glTF

`to_gltf` produces a `pygltflib` `GLTF2` object. When `output_path` is given, buffers are embedded as base64 data URIs and a `.gltf` JSON file is written; when `output_path` is `None`, nothing is written but the `GLTF2` object is still returned.

```python
from mudm_tools import to_gltf

gltf = to_gltf(fc, "output.gltf")   # writes a .gltf with embedded base64 buffers
gltf = to_gltf(fc)                  # returns the GLTF2 object, writes nothing
```

!!! note "Geometry-to-primitive mapping"
    | muDM geometry | glTF primitive |
    |---------------|----------------|
    | `Polygon`, `MultiPolygon`, `TIN`, `PolyhedralSurface` | `TRIANGLES` |
    | `LineString`, `MultiLineString` | `LINES` |
    | `Point`, `MultiPoint` | `POINTS` |

    Polygons are triangulated via Shapely Delaunay. Material index 0 is always the default-color PBR material (`metallicFactor=0.1`, `roughnessFactor=0.8`, `doubleSided=True`) using `GltfConfig.default_color` as its `baseColorFactor`. A collection produces one `Scene`, one `Node` per feature (named `feature_<i>`), and one extra material per `color_map` entry.

### Per-feature colors

To color features by a property value, set both `color_by` (the property key) and `color_map` (a mapping from stringified property values to RGBA tuples). Each `color_map` entry becomes an extra material; lookups use `str(property_value)` and fall back to material 0 when the value is absent.

```python
from mudm_tools import to_glb, GltfConfig

config = GltfConfig(
    color_by="cell_type",
    color_map={
        "neuron": (0.2, 0.4, 0.9, 1.0),
        "glia":   (0.9, 0.5, 0.1, 1.0),
    },
)
to_glb(fc, "colored.glb", config)
```

!!! tip "`color_by` needs `color_map`"
    Setting `color_by` alone has no effect — you must also provide `color_map`. Values not found in `color_map` use the default material (index 0).

### Laying out a collection

By default, features keep their original coordinates. To spread features apart for inspection, use `feature_spacing` and the `grid_max_*` limits. These are forwarded verbatim to `mudm.layout.apply_layout`, which is applied to the collection *before* mesh generation, so the resulting glTF nodes carry no translation offsets.

```python
from mudm_tools import to_glb, GltfConfig

# Auto spacing (20% of the widest feature), wrap to a new row after 5 columns
config = GltfConfig(feature_spacing=0, grid_max_x=5)
to_glb(fc, "grid.glb", config)
```

!!! note "Layout details"
    - `feature_spacing`: `None` = no layout (coords kept as-is); `0` = auto (20% of the widest feature); a positive value = fixed gap in source coordinate units.
    - `grid_max_x` / `grid_max_y` / `grid_max_z` cap the number of columns / rows / layers before wrapping; `apply_layout` raises `ValueError` early if grid capacity is exceeded.
    - Layout is applied only to a `MuDMFeatureCollection`. A single `MuDMFeature` is never laid out.

### Draco compression

Set `draco=True` (GLB path only) to enable `KHR_draco_mesh_compression` on triangle meshes. Lines and points are never Draco-compressed.

```python
from mudm_tools import to_glb, GltfConfig

config = GltfConfig(draco=True, draco_quantization_position=14, draco_compression_level=1)
to_glb(fc, "compressed.glb", config)
```

!!! warning "Draco requirements and limits"
    - Draco requires the optional `DracoPy` package. If it is missing, an `ImportError` is raised at encode time.
    - Because Draco data is appended to the GLB binary buffer, `draco=True` effectively requires the GLB path (`to_glb`), not `to_gltf`.
    - Draco applies only to triangle meshes.
    - `draco_quantization_normal` is declared on the config but is **not currently forwarded** by the encoder — only `draco_quantization_position` and `draco_compression_level` reach DracoPy.

### `GltfConfig` reference

`GltfConfig` is a Pydantic model with 13 fields:

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `include_metadata` | `bool` | `True` | Store `feature.properties` on the glTF `Node.extras` (and `collection.properties` on `Scene.extras`). |
| `y_up` | `bool` | `True` | Apply a Z-up → Y-up rotation to vertices/normals (glTF is Y-up); swaps Y↔Z and negates the new Z. |
| `default_color` | `tuple[float, float, float, float]` | `(0.8, 0.8, 0.8, 1.0)` | RGBA `baseColorFactor` of material index 0. |
| `feature_spacing` | `float \| None` | `None` | Gap between features in a collection. `None` = no layout; `0` = auto (20% of widest feature); positive = fixed gap in source units. |
| `grid_max_x` | `int \| None` | `None` | Max columns (X) before wrapping to a new row; `None` = no limit. |
| `grid_max_y` | `int \| None` | `None` | Max rows (Y) before wrapping to a new layer; `None` = no limit. |
| `grid_max_z` | `int \| None` | `None` | Max layers (Z); `None` = no limit. |
| `color_by` | `str \| None` | `None` | Property key used to look up a per-feature material color via `color_map`. |
| `color_map` | `dict[str, tuple[float, float, float, float]] \| None` | `None` | Maps property values (looked up as `str(value)`) to RGBA tuples; each entry becomes an extra material. |
| `draco` | `bool` | `False` | Enable `KHR_draco_mesh_compression` (triangle meshes only; requires DracoPy). |
| `draco_quantization_position` | `int` | `14` | Quantization bits for vertex positions (1–30). |
| `draco_quantization_normal` | `int` | `10` | Quantization bits for normals (1–30). *Currently unused by the encoder.* |
| `draco_compression_level` | `int` | `1` | Draco compression level (0–10). |

::: mudm_tools.gltf.models.GltfConfig
    options:
      show_root_heading: true

---

## End-to-end example

The snippet below loads a validated collection, exports it both ways, and reads the GeoParquet back.

=== "Python"

    ```python
    from mudm_tools import (
        to_geoparquet, from_geoparquet, ArrowConfig,
        to_glb, GltfConfig,
    )

    # `fc` is a MuDMFeatureCollection you have already built/validated.

    # 1. GeoParquet round trip
    to_geoparquet(fc, "out/features.parquet", ArrowConfig())
    fc_back = from_geoparquet("out/features.parquet")

    # 2. Draco-compressed GLB, colored by a property
    config = GltfConfig(
        draco=True,
        color_by="cell_type",
        color_map={"neuron": (0.2, 0.4, 0.9, 1.0)},
    )
    to_glb(fc, "out/scene.glb", config)
    ```

=== "Shell"

    ```bash
    # Run any of the above inside the project environment
    uv run python my_export_script.py
    ```

!!! example "Sample inputs"
    The example scripts at `src/mudm_tools/examples/` produce suitable inputs. Run them with `python -m mudm_tools.examples.<name>` — for instance `python -m mudm_tools.examples.load_validate` loads and validates a collection ready to feed into `to_geoparquet` or `to_glb`, and `python -m mudm_tools.examples.df_to_microjson` builds one from a DataFrame.

---

## See also

- [3D Tiling](3d-tiling.md) — turn meshes into a tiled, zoomable 3D Tiles / Parquet / Neuroglancer scene instead of a single file.
- [Installation](../installation.md) — installing the optional `mudm-tools[draco]` extra used by `GltfConfig(draco=True)`.
- [Format Converters](converters.md) — go straight from Xenium / OBJ / GeoJSON sources to tiled output.
- [Python API reference](../reference/python-api.md) — full signatures for the Arrow and glTF functions and their config models.
