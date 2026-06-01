# mudm-tools

[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-526CFE)](https://novagenresearch.github.io/mudm-tools/)
[![Docs build](https://github.com/NovagenResearch/mudm-tools/actions/workflows/docs.yml/badge.svg)](https://github.com/NovagenResearch/mudm-tools/actions/workflows/docs.yml)

Processing pipelines, tiling engines, and format converters for [muDM](https://github.com/NovagenResearch/mudm) spatial data. Includes optional Rust acceleration via prebuilt wheels.

📖 **Documentation:** <https://novagenresearch.github.io/mudm-tools/>

## Install

```bash
pip install mudm-tools
```

On supported platforms (Linux x86_64, macOS x86_64/arm64, Windows x86_64), this automatically includes the Rust-accelerated tiling engine. On other platforms, a pure-Python fallback is used.

```python
import mudm_tools
print(mudm_tools.RUST_AVAILABLE)  # True with prebuilt wheel
```

## Features

- **2D tiling**: Rust-accelerated quadtree pipeline (StreamingTileGenerator2D) + legacy Python pipeline
- **3D tiling**: Octree-based mesh tiling with meshopt/Draco compression (StreamingTileGenerator)
- **Format converters**: Xenium spatial transcriptomics, OBJ meshes, GeoJSON
- **GeoParquet/Arrow**: Import and export via Apache Arrow
- **glTF/GLB export**: 3D mesh export with optional Draco compression
- **Neuroglancer**: Precomputed format for browser-based 3D visualization
- **Multiple output formats**: PBF (MVT), 3D Tiles (GLB), tiled Parquet (ZSTD), Neuroglancer precomputed

## Quick start

```python
from mudm import MuDM  # core data model (separate package)
from mudm_tools._rs import StreamingTileGenerator2D  # Rust-accelerated
from mudm_tools.tiling2d import generate_pbf

gen = StreamingTileGenerator2D(min_zoom=0, max_zoom=7, buffer=64/4096)
gen.add_geojson(open("data.json").read(), bounds)
generate_pbf(gen, "tiles/", bounds)
```

## License

MIT
