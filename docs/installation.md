# Installation

`mudm-tools` is the processing-pipeline, tiling-engine, and format-converter
toolkit for [muDM](https://github.com/NovagenResearch/mudm) spatial data. It
ships as a Python package (import name `mudm_tools`) with an optional compiled
Rust extension (`mudm_tools._rs`) that accelerates the streaming tilers.

This page covers everything you need to get a working install: the supported
Python versions, the one-line install, the optional extras, building from
source, and how to verify the result. If you just want the fastest path, jump to
[Quick install](#quick-install).

!!! tip "uv-first"
    This project standardizes on [`uv`](https://docs.astral.sh/uv/) for
    development. End users can still use plain `pip` — both are shown below. For
    anything inside a cloned checkout, prefer `uv` so you get the project's
    pinned environment.

## Requirements

| Requirement | Version | Notes |
| --- | --- | --- |
| Python | `>=3.11,<3.14` | 3.13 is the highest supported. PyO3 does **not** support 3.14 yet (see [Troubleshooting](#troubleshooting)). |
| `mudm` (core data model) | `>=0.5.0` | Installed automatically as a dependency. Provides `mudm.MuDM`, `mudm.tilemodel`, `mudm.layout`, etc. |
| Rust toolchain | 1.70+ | **Only** needed to build the extension from source. Prebuilt wheels include it. |

The core scientific stack (`pydantic`, `shapely`, `pyarrow`, `geopandas`,
`numpy`, `scipy`, `scikit-learn`, `pygltflib`, `mapbox-vector-tile`, …) is pulled
in automatically — you do not install those separately.

## Quick install

=== "pip (end users)"

    ```bash
    pip install mudm-tools
    ```

=== "uv (projects)"

    ```bash
    uv add mudm-tools
    ```

This installs `mudm-tools` together with the `mudm` core package. On supported
platforms a **prebuilt wheel with the Rust extension** is selected automatically:

| Platform | Architecture |
| --- | --- |
| Linux | x86_64 |
| macOS | x86_64, arm64 (Apple Silicon) |
| Windows | x86_64 |

On any other platform pip falls back to a source build, which requires a Rust
toolchain (see [Building from source](#building-from-source)).

### Check whether Rust acceleration is active

The package exposes a `RUST_AVAILABLE` flag. It is `True` only when
`mudm_tools._rs` imported successfully (i.e. `StreamingTileGenerator` and
`StreamingTileGenerator2D` are present):

```python
import mudm_tools
print(mudm_tools.RUST_AVAILABLE)  # True with a prebuilt wheel, False otherwise
```

!!! note "It is `mudm_tools._rs`, never `mudm._rs`"
    The compiled extension lives inside the `mudm_tools` package as
    `mudm_tools._rs`. The separate `mudm` package is the pure-Python core data
    model (`mudm.MuDM`, `mudm.tilemodel`) and has no compiled component.

### What works without the Rust extension

If `RUST_AVAILABLE` is `False`, you still get the full pure-Python surface:

| Capability | Needs Rust? |
| --- | --- |
| GeoParquet / Arrow I/O — [`to_geoparquet`](reference/python-api.md), `from_geoparquet`, `to_arrow_table`, `from_arrow_table` | No |
| glTF / GLB export — `to_gltf`, `to_glb` | No |
| Neuroglancer precomputed export — `to_neuroglancer`, `write_annotations` | No |
| Legacy 2D vector tiling — `mudm2vt` (see the [legacy pipeline guide](guides/legacy-pipeline.md)) | No |
| 3D tiling helpers — `TileGenerator3D`, `OctreeConfig`, `TileReader3D` | No |
| Streaming tilers — `StreamingTileGenerator`, `StreamingTileGenerator2D` | **Yes** |
| All three [converters](guides/converters.md) (`xenium`, `obj`, `geojson`) | **Yes** — they delegate to the streaming tilers |

In short: the format I/O and legacy/3D-tiling helpers are pure Python, while the
high-throughput streaming tilers (and therefore the converter registry that
wraps them) require the compiled extension from a prebuilt wheel or a source
build.

## Optional extras

`mudm-tools` defines two optional extras in `pyproject.toml`. Install only what
you need.

| Extra | Pulls in | Enables |
| --- | --- | --- |
| `draco` | `DracoPy>=1.4` | Draco mesh compression for GLB output (`GltfConfig.draco=True`, used by `to_glb`). |
| `xenium` | `polars>=1.41.2`, `tifffile>=2026.3.3`, `pillow>=12.2.0` | The 10x Genomics **Xenium** converter (`XeniumConverter` / `format="xenium"`). |

=== "pip"

    ```bash
    pip install "mudm-tools[draco]"
    pip install "mudm-tools[xenium]"
    pip install "mudm-tools[draco,xenium]"   # both at once
    ```

=== "uv"

    ```bash
    uv add "mudm-tools[draco]"
    uv add "mudm-tools[xenium]"
    # inside a checkout, to run tests with an extra:
    uv run --extra xenium pytest
    ```

!!! warning "The Xenium converter is gated behind `[xenium]`"
    `import mudm_tools.converters.xenium` (and therefore
    `import mudm_tools.converters`) **succeeds without the extra** — the heavy
    deps (`polars`, `tifffile`, `pillow`) are imported lazily inside
    `XeniumConverter` methods. But actually **running** the Xenium conversion
    (`convert(format="xenium", ...)` or `XeniumConverter.convert(...)`) raises
    `ImportError` unless `mudm-tools[xenium]` is installed. `numpy` is a core
    dependency, so it is always present. See the
    [converters guide](guides/converters.md).

!!! note "Draco is an encode-time requirement"
    `to_glb(..., config=GltfConfig(draco=True))` raises `ImportError` at encode
    time if `DracoPy` is missing. Install `mudm-tools[draco]` first. Details in
    the [GeoParquet & glTF guide](guides/geoparquet-gltf.md).

## Install from GitHub

To install the latest unreleased code directly from the repository:

=== "uv"

    ```bash
    uv pip install "mudm-tools @ git+https://github.com/NovagenResearch/mudm-tools.git"
    ```

=== "pip"

    ```bash
    pip install "mudm-tools @ git+https://github.com/NovagenResearch/mudm-tools.git"
    ```

!!! warning "Source build required"
    A GitHub install compiles the Rust extension from source, so a Rust
    toolchain (1.70+) must be present. See [Building from source](#building-from-source).

## Building from source

Build from source when you are developing `mudm-tools`, your platform has no
prebuilt wheel, or you have changed the Rust code.

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Verify with `rustc --version` (1.70+ required).

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Clone and build

```bash
git clone https://github.com/NovagenResearch/mudm-tools.git
cd mudm-tools
uv venv --python=3.13
uv pip install -e .
.venv/bin/maturin develop --uv   # compile the Rust extension into the venv
```

`maturin` reads `[tool.maturin]` from `pyproject.toml`: it builds the crate at
`rust/Cargo.toml`, takes the Python source from `src/`, and produces the
`mudm_tools._rs` extension module. After the build, `mudm_tools.RUST_AVAILABLE`
will be `True`.

!!! tip "Rebuild after Rust changes"
    Editing anything under `rust/src/` requires a recompile. Re-run:

    ```bash
    .venv/bin/maturin develop --uv
    ```

    Pure-Python edits under `src/mudm_tools/` do **not** need a rebuild (the
    package is installed editable with `-e`).

!!! note "Performance note — the dev profile is already fast"
    `maturin develop` builds the *dev* profile. Dependencies (including the
    Draco encoder) are compiled at `opt-level=3` even in dev (see
    `[profile.dev.package."*"]` in `rust/Cargo.toml`), so the hot encode path is
    fast locally. For benchmarking the **full** pipeline, still build optimized:

    ```bash
    .venv/bin/maturin develop --release --uv
    ```

    Distributed wheels are always release-built.

!!! tip "First build is slow, later builds are cached"
    The initial Rust compile typically takes 2–4 minutes. Subsequent builds
    reuse the cargo cache and are much faster.

## Verify installation

Run this snippet to confirm both packages import, print their versions, and show
whether Rust acceleration is active:

```python
import mudm
import mudm_tools

print(f"mudm       {mudm.__version__}")
print(f"mudm-tools {mudm_tools.__version__}")
print(f"Rust:      {mudm_tools.RUST_AVAILABLE}")

# Round-trip a tiny FeatureCollection through the core model
obj = mudm.MuDM.model_validate({
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {},
        }
    ],
})
print(f"Model OK:  {len(obj.root.features)} feature(s)")
```

Save it and run with your environment of choice:

=== "uv"

    ```bash
    uv run python verify_install.py
    ```

=== "python"

    ```bash
    python verify_install.py
    ```

Expected output resembles:

```text
mudm       0.5.0
mudm-tools 0.5.0
Rust:      True
Model OK:  1 feature(s)
```

A `Rust: False` line is not an error — it just means the streaming tilers are
unavailable in this environment. Everything in
[What works without the Rust extension](#what-works-without-the-rust-extension)
still functions.

## CLI tools

The only installed console script is **`mudm-serve`**, which serves the bundled
3D and 2D tile viewers over HTTP:

```bash
# 3D viewer (default); the 2D Leaflet viewer is auto-mounted at /2d/
mudm-serve --tiles-base output/ --port 8080

# Serve the 2D viewer at /
mudm-serve --tiles-base output/ --tiles2d-base output2d/ --viewer 2d --port 8080
```

See the [CLI reference](reference/cli.md) for every flag and the
[Neuroglancer guide](guides/neuroglancer.md) for the precomputed-format route.

!!! note "There is no `mudm convert` console script"
    Format conversion is run as a module, not a console script:

    ```bash
    uv run python -m mudm_tools.converters.cli list-formats
    uv run python -m mudm_tools.converters.cli convert \
        --format geojson --input data.geojson --output tiles/
    ```

    Likewise, the bundled example scripts run as modules — e.g.
    `python -m mudm_tools.examples.tiling_rust`. Full details are in the
    [converters guide](guides/converters.md) and [CLI reference](reference/cli.md).

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `RUST_AVAILABLE` is `False` after install | On supported platforms, prebuilt wheels include Rust. If your platform has no wheel, [install Rust](#1-install-rust) and run `.venv/bin/maturin develop --uv`. |
| `FileNotFoundError: maturin` | Install maturin: `uv add --dev maturin` (or `pip install maturin`), and ensure Rust is installed. |
| `PyO3's maximum supported version (3.13)` | Use Python 3.13, not 3.14. PyO3 does not support 3.14 yet; `requires-python` is `>=3.11,<3.14`. |
| `error: can't find Rust compiler` | Run `rustup default stable` to set a default toolchain. |
| `ImportError` when running the Xenium converter | Install the extra: `pip install "mudm-tools[xenium]"` (or `uv run --extra xenium ...`). Importing the module works without it, but `convert()` does not. |
| `ImportError` from `to_glb(..., config=GltfConfig(draco=True))` | Install `mudm-tools[draco]` (pulls in `DracoPy`). Draco is enabled via the config, not a direct `to_glb` argument. |
| Slow first build from source | Normal — Rust compilation takes 2–4 minutes. Subsequent builds are cached. |

## Next steps

- [Getting started](getting-started.md) — your first end-to-end pipeline.
- [Converters guide](guides/converters.md) — Xenium, OBJ, and GeoJSON conversion.
- [2D tiling](guides/2d-tiling.md) and [3D tiling](guides/3d-tiling.md).
- [Python API reference](reference/python-api.md) — every public symbol.
