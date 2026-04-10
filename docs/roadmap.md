# Roadmap

Below is the development roadmap for muDM, showing completed phases and current/future work. The project has evolved from a lightweight annotation format into a high-performance tiling platform with Rust acceleration.

## Phase 1: Consolidation and Documentation (Complete)

1. **Refinement of Core Model**:
    * Finalized and stabilized the muDM core model with Pydantic v2.
    * Published updated documentation and specifications.
    * Implemented hierarchical references (`parentId`, `ref`).

2. **Community Engagement**:
    * Established communication channels via GitHub issues.
    * Gathered feedback from stakeholders and early adopters.

## Phase 2: Expanded Features and Extensions (Complete)

1. **Harmonization with GeoJSON Pydantic**:
    * Full compatibility with [geojson-pydantic](https://developmentseed.org/geojson-pydantic/) — muDM models extend GeoJSON Pydantic types.

2. **Harmonization with OME Model**:
    * Integrated OME-compatible coordinate systems and multiscale metadata.
    * Provenance tracking module for data lineage.

3. **Tiling with TileJSON and binary formats**:
    * TileJSON 3.0.0 metadata specification for muDM tilesets.
    * 2D vector tile pipeline: GeoJSON to PBF (MVT) and tiled Parquet.
    * Python TileWriter/TileReader for legacy PBF workflows.

## Phase 3: 3D Data and Rust Acceleration (Complete)

1. **Rust-Accelerated Tiling Engine**:
    * Full rewrite of hot-path tiling code in Rust (PyO3 + maturin).
    * 2D pipeline: `StreamingTileGenerator2D` with quadtree-based clipping, Douglas-Peucker simplification, and parallel tile encoding via rayon.
    * 3D pipeline: `StreamingTileGenerator` with octree indexing, QEM mesh simplification, and multi-format output.

2. **3D Geometry and Mesh Support**:
    * Added `PolyhedralSurface` and `TIN` geometry types.
    * OBJ mesh ingestion with parallel parsing.
    * Fragment file format (MJF2) for sharded intermediate storage.

3. **Output Formats**:
    * **3D Tiles** (GLB) with meshopt or Draco compression.
    * **PBF3** — custom protobuf format for 3D tile data.
    * **Tiled Parquet** (ZSTD) for ML training pipelines.
    * **Neuroglancer** precomputed format for web-based 3D visualization.
    * **2D PBF** (MVT) — pure-Rust encoder/decoder, no Python dependencies.

4. **Compression**:
    * Meshopt (lossless, fast decode, Brotli-friendly) — default for viewer output.
    * Draco (lossy quantization, smallest on disk) — optional.
    * ZSTD for Parquet columns.
    * Brotli HTTP transport compression for GLB serving.

5. **Dataset Pipelines**:
    * MouseLight: 38 brains, 876K rows, meshopt 3D Tiles (84 min total).
    * Hemibrain: 5,000 neurons (95 cell types), Parquet tiling complete.

## Phase 4: Adoption and Long-Term Sustainability (In Progress)

1. **Format Converter Registry**:
    * Pluggable converter system (`mudm_tools.converters`) with `convert()` and `list_formats()` API.
    * Built-in converters: GeoJSON, OBJ, Xenium.
    * CLI interface: `python -m mudm_tools.converters.cli convert --format <fmt> ...`.

2. **Xenium 2D Spatial Transcriptomics Pipeline**:
    * End-to-end converter for 10x Genomics Xenium output (transcripts, cell boundaries, nuclei).
    * Rust-native Parquet ingestion (`add_parquet_points`, `add_parquet_polygons`) — bypasses GeoJSON serialization entirely.
    * Rust-native Parquet output (`generate_parquet_native`) — parallel per-zoom part files.
    * Tested at scale: 42.97M features (breast cancer dataset) in ~20 minutes.

3. **2D Viewer**:
    * Leaflet-based viewer (`viewer2d/`) with DAPI raster base layer and multi-layer MVT overlays.
    * Gene category color mapping, layer toggles, hover/click info panel.
    * Dataset selector for switching between multiple tiled datasets.

4. **Case Studies and Benchmarks**:
    * Published benchmark tables: synthetic, MouseLight, retrieval, multi-pyramid.
    * Reproducing paper results with current pipeline (Tables 2-8).

5. **Documentation and Guides**:
    * Updated specification for 3D geometry types.
    * API documentation for Rust-accelerated modules.
    * Example scripts for 2D and 3D pipelines.

6. **Remaining Work**:
    * TypeScript and Java reference implementations.
    * Governance model and standards process.
    * Hemibrain full 3D Tiles with QEM + meshopt.
    * Community engagement: user meetings and feedback sessions.
