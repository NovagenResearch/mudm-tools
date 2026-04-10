# Using muDM

This page covers common usage patterns for the muDM library (`mudm` Python package), from basic validation to high-performance tiling pipelines.

## Requirements

- Python >= 3.11, < 3.14
- Install: `uv add mudm` (or `pip install mudm`)
- For Rust acceleration (tiling pipelines): built automatically via maturin when installing from source

## Validating muDM and GeoJSON

```python
import mudm.model as mj
import json

# Validate a muDM file
with open("annotations.json") as f:
    data = json.load(f)
mudm_obj = mj.MuDM.model_validate(data)

# Validate a GeoJSON file (any GeoJSON is valid muDM)
with open("features.geojson") as f:
    data = json.load(f)
geojson_obj = mj.GeoJSON.model_validate(data)
```

## Creating muDM from a DataFrame

The `df_to_microjson` function converts a pandas DataFrame into a muDM FeatureCollection.

::: mudm_tools.examples.df_to_microjson.df_to_microjson
    :docstring:

```python
import pandas as pd
from mudm_tools.examples.df_to_microjson import df_to_microjson

data = [
    {
        "type": "Feature",
        "geometryType": "Point",
        "coordinates": [0, 0],
        "name": "Point 1",
        "value": 1,
        "values": [1, 2, 3],
    },
    {
        "type": "Feature",
        "geometryType": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
        "name": "Polygon 1",
        "value": 2,
        "values": [4, 5, 6],
    },
]

df = pd.DataFrame(data)
feature_collection = df_to_microjson(df)
print(feature_collection.model_dump_json(indent=2, exclude_unset=True))
```

## 2D Tiling (Rust-Accelerated)

Generate vector tiles from GeoJSON data using the Rust pipeline. See the [Tiling](tiling.md) page for the full specification.

### PBF (MVT) Output

```python
from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d import generate_pbf, read_pbf

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096)

geojson_str = open("data.json").read()
bounds = (0.0, 0.0, 10000.0, 10000.0)
gen.add_geojson(geojson_str, bounds)

# Write PBF tiles to a directory tree ({z}/{x}/{y}.pbf)
n_tiles = generate_pbf(gen, "tiles/", bounds, simplify=True)

# Read tiles back
features = read_pbf("tiles/", bounds, zoom=0)
```

### Tiled Parquet Output

```python
from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d import generate_parquet, read_parquet

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096)

geojson_str = open("data.json").read()
bounds = (0.0, 0.0, 10000.0, 10000.0)
gen.add_geojson(geojson_str, bounds)

# Write tiled Parquet (ZSTD compressed)
n_rows = generate_parquet(gen, "output.parquet", bounds, simplify=True)

# Read with optional zoom/tile filtering
rows = read_parquet("output.parquet", zoom=0)
```

A complete example script is at `src/mudm/examples/tiling_rust.py`.

### Rust-Native Parquet Ingestion

For large datasets, bypass GeoJSON entirely and ingest directly from Parquet:

```python
from mudm_tools._rs import StreamingTileGenerator2D

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096, temp_dir="/data/tmp")

# Points: reads x/y columns and a property column directly via arrow-rs
# No JSON serialization — Rayon-parallel clipping
count = gen.add_parquet_points(
    "transcripts.parquet",
    "x_location", "y_location",     # coordinate columns
    "feature_name", "gene_name",    # property column → output tag name
    "transcripts",                  # layer_type tag value
    (0.0, 0.0, 8192.0, 8192.0),    # world bounds
    coord_scale=1.0 / 0.2125,      # microns → pixels
)

# Polygons: reads vertex columns, groups by ID to build rings
count = gen.add_parquet_polygons(
    "cell_boundaries.parquet",
    "cell_id", "vertex_x", "vertex_y",  # id and coordinate columns
    "cells",                             # layer_type tag value
    (0.0, 0.0, 8192.0, 8192.0),
    coord_scale=1.0 / 0.2125,
)
```

### Rust-Native Parquet Output

Write Parquet entirely in Rust (parallel per-zoom part files):

```python
# After adding features, generate both MVT and Parquet:
from mudm_tools.tiling2d import generate_pbf

n_tiles = generate_pbf(gen, "vectors/", bounds, simplify=True, layer_name="features")
n_rows = gen.generate_parquet_native("features.parquet/", bounds, simplify=True)
```

## Format Converters

The converter registry provides standardized entry points for common source formats.

### Python API

```python
from mudm_tools.converters import convert, list_formats

# List available formats
print(list_formats())  # ['geojson', 'obj', 'xenium']

# Convert Xenium spatial transcriptomics data
result = convert("xenium",
    input_dir="data/Xenium_outs",
    output_dir="tiles/xenium_sample",
    config={
        "temp_dir": "/data/tmp",
        "point_zoom_offset": 3,  # transcripts only at detailed zooms
        "skip_raster": False,
    })
print(result["layer_counts"])  # {'cells': 167780, 'nuclei': 167780, 'transcripts': 42638083}

# Convert GeoJSON
result = convert("geojson",
    input_dir="annotations.geojson",
    output_dir="tiles/annotations",
    config={"max_zoom": 7})

# Convert OBJ meshes to 3D tiles + Parquet
result = convert("obj",
    input_dir="data/meshes/",
    output_dir="tiles/brain",
    config={"max_zoom": 4, "temp_dir": "/data/tmp"})
```

### CLI

```bash
# List formats
python -m mudm_tools.converters.cli list-formats

# Convert Xenium data
python -m mudm_tools.converters.cli convert \
    --format xenium \
    --input data/Xenium_outs \
    --output tiles/xenium_sample \
    --temp-dir /data/tmp

# Convert with JSON config file
python -m mudm_tools.converters.cli convert \
    --format xenium \
    --input data/Xenium_outs \
    --output tiles/sample \
    --config xenium_config.json
```

### Converter Config Keys

**Xenium** (`--format xenium`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temp_dir` | str | system temp | Temp directory for fragment files |
| `max_zoom` | int | from image | Override max zoom level |
| `point_zoom_offset` | int | 3 | Transcripts start at max_zoom - offset |
| `id_column` | str | "cell_id" | Boundary polygon ID column |
| `skip_raster` | bool | false | Skip raster tile generation if tiles exist |

**OBJ** (`--format obj`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temp_dir` | str | None | Temp directory for fragment files |
| `max_zoom` | int | 4 | Max zoom level |
| `bounds` | tuple | auto-scan | World bounds (xmin,ymin,zmin,xmax,ymax,zmax) |
| `tags` | dict | filename | Per-file properties |
| `glob` | str | "*.obj" | File pattern |
| `generate_parquet` | bool | true | Also generate Parquet output |

**GeoJSON** (`--format geojson`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `temp_dir` | str | system temp | Temp directory |
| `max_zoom` | int | 7 | Max zoom level |
| `bounds` | tuple | auto-compute | World bounds (xmin,ymin,xmax,ymax) |
| `layer_name` | str | "features" | MVT layer name |
| `glob` | str | "*.geojson" | File pattern (if input is a directory) |

## 2D Viewer

The Leaflet-based 2D viewer displays raster images with MVT vector overlays.

### Starting the Viewer

```bash
mudm-serve --tiles-base data/tiles --tiles2d-base tiles/ --viewer 2d --port 8080
# Open http://localhost:8080/
```

### Features

- **DAPI raster** base layer with PNG tile pyramid
- **Multi-layer MVT** overlays: cell boundaries, nucleus boundaries, transcripts
- **Layer toggles** and **gene category checkboxes** (oncogenes, immune, stromal, etc.)
- **Gene filter** dropdown populated from `gene_list.json` sidecar
- **Color-by-gene** from `gene_colormap.json` with configurable category colors
- **Hover/click** info panel showing feature properties
- **Micron scale bar** (not meters)
- **Dataset selector** for switching between multiple tiled datasets
- **Restyle-in-place** — toggling layers/categories updates styles without reloading tiles

### Output File Structure

```
tiles/<dataset>/
├── raster/{z}/{x}/{y}.png       # DAPI image pyramid
├── vectors/{z}/{x}/{y}.pbf      # Multi-layer MVT (cells + nuclei + transcripts)
├── vectors/metadata.json        # TileJSON 3.0.0
├── features.parquet/            # Partitioned Parquet for ML
│   ├── zoom=0/part_*.parquet
│   ├── ...
│   └── zoom=9/part_*.parquet
├── metadata.json                # Dataset metadata (bounds, layers, um_per_px)
├── gene_list.json               # Gene names for dropdown (Xenium)
└── gene_colormap.json           # Gene category colors (Xenium)
```

## 3D Tiling

Generate 3D Tiles (GLB), tiled Parquet, or Neuroglancer output from OBJ mesh files using `StreamingTileGenerator`. Supports meshopt and Draco compression, bucketed redistribution for datasets exceeding RAM, and parallel ingestion/encoding.

```python
from mudm_tools._rs import StreamingTileGenerator

gen = StreamingTileGenerator(min_zoom=0, max_zoom=5, extent=4096, extent_z=4096)
gen.add_obj_files(["neuron_001.obj", "neuron_002.obj"], bounds)
gen.generate_3dtiles("output/3dtiles", bounds, compression="meshopt")
```

For the full API reference, geometry types, compression options, viewer features, and large-dataset handling, see the [3D Data and Tiling](3d.md) page.

## GeoParquet Import/Export

```python
from mudm_tools import to_geoparquet, from_geoparquet, ArrowConfig

# Export muDM to GeoParquet
config = ArrowConfig()
to_geoparquet(feature_collection, "output.parquet", config)

# Import from GeoParquet
fc = from_geoparquet("output.parquet")
```

## glTF/GLB Export

```python
from mudm_tools import to_glb, GltfConfig

config = GltfConfig()
to_glb(feature_collection, "output.glb", config)
```
