# Roadmap

This page tracks how **mudm-tools** has evolved and where it is headed. The
project grew from a lightweight annotation format into a high-performance,
Rust-accelerated tiling platform for 2D and 3D spatial biology data.

!!! note "How to read this page"
    Phases 1–3 are **complete**. Phase 4 (adoption and sustainability) is
    **in progress** as of mid-2026. Where a phase maps onto a shipped feature,
    you'll find a link to the relevant guide so you can try it today. Dates and
    benchmark numbers describe the current pipeline.

## Phase 1: Consolidation and Documentation (Complete)

1. **Refinement of the core model**
    * Finalized and stabilized the [`mudm`](reference/python-api.md) core model
      on Pydantic v2.
    * Published updated specifications and documentation.
    * Implemented hierarchical references (`parentId`, `ref`) for skeletons and
      compartment trees.

2. **Community engagement**
    * Established communication channels via GitHub issues.
    * Gathered feedback from stakeholders and early adopters.

## Phase 2: Expanded Features and Extensions (Complete)

1. **Harmonization with GeoJSON Pydantic**
    * Full compatibility with
      [geojson-pydantic](https://developmentseed.org/geojson-pydantic/) — muDM
      models extend the GeoJSON Pydantic types.

2. **Harmonization with the OME model**
    * Integrated OME-compatible coordinate systems and multiscale metadata.
      See the [OME-NGFF guide](guides/ome-ngff.md).
    * Provenance tracking for data lineage.

3. **Tiling with TileJSON and binary formats**
    * [TileJSON 3.0.0](reference/tilejson.md) metadata for muDM tilesets.
    * 2D vector-tile pipeline: GeoJSON → PBF (MVT) and tiled Parquet.
    * Python `TileWriter` / `TileReader` for the
      [legacy PBF workflow](guides/legacy-pipeline.md).

## Phase 3: 3D Data and Rust Acceleration (Complete)

This phase delivered the Rust extension (`mudm_tools._rs`, built with PyO3 +
maturin) and the full 3D tiling stack. See the
[3D tiling guide](guides/3d-tiling.md) to run it.

1. **Rust-accelerated tiling engine**
    * Hot-path tiling rewritten in Rust.
    * 2D pipeline: `StreamingTileGenerator2D` with quadtree clipping,
      Douglas–Peucker simplification, and parallel tile encoding via rayon
      (see the [2D tiling guide](guides/2d-tiling.md)).
    * 3D pipeline: `StreamingTileGenerator` with octree indexing, QEM mesh
      simplification, and multi-format output.

2. **3D geometry and mesh support**
    * Added `PolyhedralSurface` and `TIN` geometry types.
    * OBJ mesh ingestion with parallel parsing
      (see the [converters guide](guides/converters.md)).
    * Fragment file format (MJF2) for sharded intermediate storage.

3. **Output formats**
    * **3D Tiles** (GLB) with meshopt or Draco compression.
    * **PBF3** — custom protobuf format for 3D tile data.
    * **Tiled Parquet** (ZSTD) for ML training pipelines
      (see the [GeoParquet/glTF guide](guides/geoparquet-gltf.md)).
    * **Neuroglancer** precomputed format for web-based 3D visualization
      (see the [Neuroglancer guide](guides/neuroglancer.md)).
    * **2D PBF** (MVT) — pure-Rust encoder/decoder.

4. **Compression**
    * Meshopt (lossless, fast decode, Brotli-friendly) — default for viewer
      output.
    * Draco (lossy quantization, smallest on disk) — optional.
    * ZSTD for Parquet columns.
    * Brotli HTTP transport compression for GLB serving.

5. **Neuroglancer precomputed path** *(hardened in this phase)*
    * Multi-LOD Draco meshes for level-of-detail streaming.
    * **Opt-in `neuroglancer_uint64_sharded_v1` output** that packs every
      segment into `.shard` files for object-store-scale hosting, instead of
      one loose file per segment.
    * **Deterministic fragment ordering** so repeated runs produce
      byte-identical output (reproducible builds, content-addressable caching).
    * Large-mesh Draco-encode performance improvements.

    The sharded layout is opt-in on `generate_neuroglancer_multilod` (the basic
    `generate_neuroglancer` method takes only `output_dir` and `world_bounds`).
    The compiled Rust method (not introspectable by autodoc — hand-written here)
    has the signature:

    ```python
    generator.generate_neuroglancer_multilod(
        output_dir,
        world_bounds,
        vertex_quantization_bits=10,
        max_memory_bytes=0,
        sharded=False,
        minishard_bits=6,
        shard_bits=0,
    )
    ```

    | Parameter                 | Default | Meaning                                                                 |
    | ------------------------- | ------- | ----------------------------------------------------------------------- |
    | `output_dir`              | —       | Destination directory for the precomputed dataset.                      |
    | `world_bounds`            | —       | World-space bounding box used to place meshes.                          |
    | `vertex_quantization_bits`| `10`    | Draco position-quantization bits.                                       |
    | `max_memory_bytes`        | `0`     | Soft memory ceiling for streaming. `0` = auto: the generator's resolved budget (`MUDM_MAX_MEMORY_GB` env, else 0.8 × physical RAM, else an 8 GiB fallback) — never truly unbounded. |
    | `sharded`                 | `False` | Emit `neuroglancer_uint64_sharded_v1` `.shard` files instead of loose per-segment files. |
    | `minishard_bits`          | `6`     | Minishard index bits (sharded mode only).                               |
    | `shard_bits`              | `0`     | Shard index bits (sharded mode only).                                   |

    !!! tip "When to enable sharding"
        Loose per-segment files (the default) are simplest for local viewing.
        Switch on `sharded=True` when hosting many thousands of segments on an
        object store (S3/GCS), where one `.shard` file per group is far cheaper
        than millions of small objects. See the
        [Neuroglancer guide](guides/neuroglancer.md) for a full walkthrough.

6. **Dataset pipelines**
    * MouseLight: 38 brains, 876K rows, meshopt 3D Tiles (84 min total).
    * Hemibrain: 5,000 neurons (95 cell types), Parquet tiling complete.

## Phase 4: Adoption and Long-Term Sustainability (In Progress)

1. **Format converter registry** *(shipped)*
    * Pluggable converter system in `mudm_tools.converters` with a
      `convert()` / `list_formats()` API.
    * Built-in converters: **GeoJSON**, **OBJ**, **Xenium**.
    * CLI front end: `python -m mudm_tools.converters.cli`.

    === "Python"

        ```python
        from mudm_tools.converters import convert, list_formats

        print(list_formats())  # ['geojson', 'obj', 'xenium']

        # format first; input/output are directories (input may also be a file)
        convert("geojson", input_dir="annotations.geojson", output_dir="tiles/annotations")
        ```

    === "CLI"

        ```bash
        uv run python -m mudm_tools.converters.cli convert \
            --format geojson --input annotations.geojson --output tiles/annotations
        ```

    See the [converters guide](guides/converters.md) and
    [CLI reference](reference/cli.md).

2. **Xenium 2D spatial-transcriptomics pipeline at scale** *(shipped)*
    * End-to-end converter for 10x Genomics Xenium output (transcripts, cell
      boundaries, nuclei).
    * Rust-native Parquet ingestion (`add_parquet_points`,
      `add_parquet_polygons`) — bypasses GeoJSON serialization entirely.
    * Rust-native Parquet output (`generate_parquet_native`) — parallel
      per-zoom part files.
    * Validated at scale: 42.97M features (breast-cancer dataset) in
      ~20 minutes. See the [2D tiling guide](guides/2d-tiling.md).

3. **Web viewers** *(shipped)*
    * **2D Leaflet viewer** (`src/mudm_tools/viewers/viewer2d/`) with a DAPI
      raster base layer and multi-layer MVT overlays, gene-category color
      mapping, layer toggles, a hover/click info panel, and a dataset selector.
    * **3D Three.js viewer** (`src/mudm_tools/viewers/viewer3d/`) for streaming
      3D Tiles, with bounding-volume display, slice planes, an axis gizmo, and
      overview/info panels.
    * Both are served by the `mudm-serve` console script — see the
      [CLI reference](reference/cli.md).

4. **Documentation and guides** *(in progress)*
    * This documentation site (the page you're reading).
    * Task-oriented guides for [2D tiling](guides/2d-tiling.md),
      [3D tiling](guides/3d-tiling.md), [converters](guides/converters.md),
      [Neuroglancer](guides/neuroglancer.md),
      [GeoParquet/glTF](guides/geoparquet-gltf.md),
      [OME-NGFF](guides/ome-ngff.md), and the
      [legacy pipeline](guides/legacy-pipeline.md).
    * A [Python API reference](reference/python-api.md) and runnable example
      scripts under `src/mudm_tools/examples/`, e.g.:

        ```bash
        uv run python -m mudm_tools.examples.tiling_rust
        ```

5. **Remaining work** *(general direction)*
    * Reference implementations in additional languages.
    * A governance model and standards process for the muDM specification.
    * Broader dataset coverage and additional case studies.
    * Continued community engagement: user meetings and feedback sessions.

!!! example "Want to try the current pipeline?"
    Start with [Getting Started](getting-started.md) for a guided tour,
    then pick the guide that matches your data: [2D tiling](guides/2d-tiling.md)
    for spatial transcriptomics, or [3D tiling](guides/3d-tiling.md) for meshes
    and neuron morphologies.
