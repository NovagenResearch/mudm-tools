# Guides

Task-oriented walkthroughs for each part of `mudm-tools`. Pick the guide that
matches your data and goal:

| Guide | Use it when you want to… |
| --- | --- |
| [2D Tiling](2d-tiling.md) | Turn GeoJSON / Parquet point & polygon data (e.g. spatial transcriptomics) into PBF/MVT vector tiles and tiled Parquet, with the Rust `StreamingTileGenerator2D`. |
| [3D Tiling](3d-tiling.md) | Turn OBJ meshes (neurons, organelles, whole brains) into OGC 3D Tiles (GLB), PBF3, tiled Parquet, or Neuroglancer, with the Rust `StreamingTileGenerator` — or the pure-Python `TileGenerator3D`. |
| [Format Converters](converters.md) | Go straight from a Xenium / OBJ / GeoJSON source directory to tiled output with one `convert()` call or a CLI command. |
| [Neuroglancer Export](neuroglancer.md) | Publish precomputed meshes, skeletons (incl. SWC neuron morphologies), and annotations for the Neuroglancer viewer — loose or `uint64_sharded_v1`. |
| [GeoParquet & glTF](geoparquet-gltf.md) | Round-trip a muDM collection to GeoParquet/Arrow, or export it as a single glTF/GLB file. |
| [OME-NGFF Integration](ome-ngff.md) | Understand the design argument for bridging OME-NGFF arrays with TileJSON + muDM vector tiles *(conceptual note, not a how-to)*. |
| [Legacy Pipeline](legacy-pipeline.md) | Work with the pre-Rust pure-Python `TileWriter`/`TileReader`/`mudm2vt` path. |

!!! tip "Not sure where to start?"
    Run through [Getting Started](../getting-started.md) first — it takes you from
    install to tiles to a live viewer. Then come back here for the deep dive on
    your format. For exact signatures, see the
    [Python API reference](../reference/python-api.md) and the
    [CLI reference](../reference/cli.md).
