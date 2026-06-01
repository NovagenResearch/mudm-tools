# Home

**mudm-tools** is the processing layer for [muDM](https://github.com/NovagenResearch/mudm) spatial data: a set of streaming **tiling engines**, **format converters**, and **import/export writers** for microscopy and bioimaging vector + 3D-mesh data, with optional **Rust acceleration** delivered through prebuilt wheels. It turns large GeoJSON / Parquet / OBJ / Xenium datasets into web-ready tile pyramids and interchange formats — PBF/MVT vector tiles, tiled Parquet (ZSTD), OGC 3D Tiles (GLB with meshopt/Draco), PBF3, and Neuroglancer precomputed meshes/skeletons/annotations.

!!! note "Two packages, one ecosystem"
    There are two related packages, and they are **not** the same thing:

    - **`mudm`** — the core data model (Pydantic), e.g. `mudm.MuDM`, `mudm.model.MuDMFeatureCollection`, `mudm.tilemodel`. This is a **separate package** you install alongside.
    - **`mudm_tools`** — *this* package: the pipelines, tiling engines, converters, and writers documented on this site. The compiled Rust extension is imported as **`mudm_tools._rs`** (never `mudm._rs`).

## Capabilities

| Capability | Entry point | Guide |
|------------|-------------|-------|
| 2D vector tiling (quadtree → PBF/MVT + tiled Parquet) | `mudm_tools._rs.StreamingTileGenerator2D` | [2D tiling](guides/2d-tiling.md) |
| 3D mesh tiling (octree → 3D Tiles, PBF3, Parquet, Neuroglancer) | `mudm_tools._rs.StreamingTileGenerator` | [3D tiling](guides/3d-tiling.md) |
| Format converters (Xenium / OBJ / GeoJSON) | `mudm_tools.converters.convert()` | [Converters](guides/converters.md) |
| Neuroglancer precomputed (skeletons, meshes, annotations) | `mudm_tools.to_neuroglancer()` + `mudm_tools.neuroglancer` | [Neuroglancer](guides/neuroglancer.md) |
| Neuron morphologies (SWC → skeletons / meshes) | `mudm_tools.examples.swc_to_neuroglancer` | [Neuroglancer](guides/neuroglancer.md) |
| GeoParquet & glTF/GLB export | `mudm_tools.to_geoparquet()` / `to_glb()` | [GeoParquet & glTF](guides/geoparquet-gltf.md) |
| OME-NGFF integration *(design note)* | conceptual integration spec | [OME-NGFF](guides/ome-ngff.md) |
| Web viewers + CLI (`mudm-serve`, converter CLI) | `mudm-serve`, `python -m mudm_tools.converters.cli` | [CLI reference](reference/cli.md) |
| Legacy Python pipeline | pure-Python tiling fallback | [Legacy pipeline](guides/legacy-pipeline.md) |

!!! tip "Where the heavy lifting happens"
    The 2D and 3D tiling engines and the converters delegate to the compiled Rust extension `mudm_tools._rs` (parallel ingest with the GIL released, on-disk fragment sharding, ZSTD Parquet, Draco/meshopt GLB). The GeoParquet, glTF/GLB, and most Neuroglancer writers are pure Python.

## Install

```bash
pip install mudm-tools
```

On supported platforms (Linux x86_64, macOS x86_64/arm64, Windows x86_64) the prebuilt wheel bundles the Rust-accelerated tiling engine. Inside a project, prefer `uv`:

=== "uv (project)"

    ```bash
    uv add mudm-tools
    ```

=== "pip (end user)"

    ```bash
    pip install mudm-tools
    ```

Check whether the Rust extension is available at runtime:

```python
import mudm_tools
print(mudm_tools.RUST_AVAILABLE)  # True when the prebuilt wheel is installed
```

!!! note "Xenium extra"
    The Xenium converter's runtime dependencies (polars, tifffile, pillow) are gated behind an optional extra and imported lazily. Install them with `pip install "mudm-tools[xenium]"` (or `uv add "mudm-tools[xenium]"`). `import mudm_tools.converters.xenium` works without the extra; only calling `.convert()` needs it.

See [Installation](installation.md) for the full matrix (Python versions, optional extras, building from source) and [Getting Started](getting-started.md) for a guided first run.

## 30-second example

Turn a GeoJSON file into a `{z}/{x}/{y}.pbf` vector-tile pyramid:

```python
from mudm_tools._rs import StreamingTileGenerator2D
from mudm_tools.tiling2d import generate_pbf

# World bounds of your data: (xmin, ymin, xmax, ymax)
bounds = (0.0, 0.0, 10000.0, 10000.0)

# max_zoom=7 here is an example value; the constructor default is 4.
# buffer is in NORMALIZED [0,1] space (here 64/4096 of the full extent).
gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64 / 4096)

# Ingest GeoJSON text, projecting world coords into [0,1]^2 against `bounds`.
gen.add_geojson(open("data.json").read(), bounds)

# Write the MVT pyramid + a TileJSON 3.0.0 metadata.json. Returns tile count.
n_tiles = generate_pbf(gen, "tiles/", bounds)
print(f"wrote {n_tiles} tiles")
```

This writes:

```text
tiles/
  metadata.json          # TileJSON 3.0.0
  {z}/{x}/{y}.pbf        # one Mapbox Vector Tile per non-empty tile
```

!!! warning "Coordinate and default gotchas"
    - The `StreamingTileGenerator2D` constructor defaults are `min_zoom=0, max_zoom=4, buffer=0.0, temp_dir=None`. The `max_zoom=7` above is just an example, **not** the API default.
    - `buffer` is a fraction of the **full extent in normalized `[0,1]` space**, not a per-tile pixel value.
    - After any `generate_*` call the generator's writer is consumed — further `add_*` calls raise `RuntimeError`. Build one generator per output run.

Read tiles back, or emit tiled Parquet instead:

=== "Read PBF back"

    ```python
    from mudm_tools.tiling2d import read_pbf

    features = read_pbf("tiles/", bounds, zoom=0)
    # each dict: zoom, tile_x, tile_y, feature_id, geom_type,
    #            positions (np.float32 [N,2]), ring_lengths, tags
    ```

=== "Tiled Parquet"

    ```python
    from mudm_tools.tiling2d import generate_parquet, read_parquet

    gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64 / 4096)
    gen.add_geojson(open("data.json").read(), bounds)

    # Single ZSTD-compressed file by default (pass partitioned=True for a
    # zoom={z}/part_NNN.parquet directory tree).
    n_rows = generate_parquet(gen, "output.parquet", bounds)
    rows = read_parquet("output.parquet", zoom=0)
    ```

=== "Convert OBJ → 3D Tiles"

    ```python
    from mudm_tools.converters import convert

    result = convert(
        "obj",
        input_dir="data/meshes/",
        output_dir="tiles/brain",
        config={"max_zoom": 4},
    )
    print(result["feature_count"])
    ```

!!! example "Runnable example scripts"
    The package ships example scripts under `src/mudm_tools/examples/`. Run them as modules, e.g.:

    ```bash
    uv run python -m mudm_tools.examples.tiling_rust
    uv run python -m mudm_tools.examples.swc_to_neuroglancer
    ```

## Where to next

- **New here?** Start with [Installation](installation.md), then [Getting Started](getting-started.md).
- **Tiling deep dives:** [2D tiling](guides/2d-tiling.md) and [3D tiling](guides/3d-tiling.md).
- **Bringing in your data:** [Converters](guides/converters.md) (Xenium / OBJ / GeoJSON), [GeoParquet & glTF/GLB](guides/geoparquet-gltf.md), [OME-NGFF](guides/ome-ngff.md).
- **Visualization:** [Neuroglancer](guides/neuroglancer.md) and the web viewers in the [CLI reference](reference/cli.md).
- **Reference:** [Python API](reference/python-api.md), [CLI](reference/cli.md), [TileJSON](reference/tilejson.md), and the [Roadmap](roadmap.md).
- **Coming from the old code path?** See the [Legacy Python pipeline](guides/legacy-pipeline.md).
