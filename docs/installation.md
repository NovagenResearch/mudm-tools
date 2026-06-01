# Installation

## Requirements

- **Python** >= 3.11, < 3.14
- **mudm** (core data model — installed automatically as a dependency)

## Quick Install

```bash
pip install mudm-tools
```

This installs `mudm-tools` and the `mudm` core package. On supported platforms (Linux x86_64, macOS x86_64/arm64, Windows x86_64), a prebuilt wheel with Rust acceleration is installed automatically.

Check what's available:

```python
import mudm_tools
print(mudm_tools.RUST_AVAILABLE)  # True with prebuilt wheel, False otherwise
```

Without the Rust extension, pure-Python pipelines (legacy 2D tiling, GeoParquet I/O, glTF export) still work. The Rust-accelerated `StreamingTileGenerator` and `StreamingTileGenerator2D` require a prebuilt wheel or building from source.

For optional Draco compression support:

```bash
pip install mudm-tools[draco]
```

## Install from GitHub

```bash
uv pip install "mudm-tools @ git+https://github.com/NovagenResearch/mudm-tools.git"
```

This requires a Rust toolchain since the extension is compiled from source.

## Building from Source

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Verify with `rustc --version` (1.70+ required).

### 2. Install uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Clone and install

```bash
git clone https://github.com/NovagenResearch/mudm-tools.git
cd mudm-tools
uv venv --python=3.13
source .venv/bin/activate
uv pip install -e .
maturin develop --uv   # compile Rust extension
```

After modifying Rust code, rebuild with:

```bash
maturin develop --uv
```

> **Performance note.** `maturin develop` builds the *dev* profile. Dependencies
> (including the Draco encoder) are compiled at `opt-level=3` even in dev (see
> `[profile.dev.package."*"]` in `rust/Cargo.toml`), so the hot encode path is
> fast locally. For benchmarking the full pipeline, still build optimized:
> `maturin develop --release --uv`. Distributed wheels are always release-built.

## Verify Installation

```python
import mudm
import mudm_tools

print(f"mudm {mudm.__version__}")
print(f"mudm-tools {mudm_tools.__version__}")
print(f"Rust: {mudm_tools.RUST_AVAILABLE}")

obj = mudm.MuDM.model_validate({
    "type": "FeatureCollection",
    "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {}}]
})
print(f"Model OK: {len(obj.root.features)} feature(s)")
```

## CLI Tools

`mudm-tools` provides the `mudm-serve` command for viewing tiles:

```bash
mudm-serve --tiles-base output/ --port 8080           # 3D viewer (default)
mudm-serve --tiles-base output/ --viewer 2d --port 8080  # 2D viewer
```

## Troubleshooting

| Problem | Solution |
| --- | --- |
| `RUST_AVAILABLE` is `False` after install | On supported platforms, prebuilt wheels include Rust. If your platform has no wheel, install Rust and run `maturin develop`. |
| `FileNotFoundError: maturin` | Install maturin: `pip install maturin`, and ensure Rust is installed. |
| `PyO3's maximum supported version (3.13)` | Use Python 3.13, not 3.14. PyO3 doesn't support 3.14 yet. |
| `error: can't find Rust compiler` | Run `rustup default stable` to set a default toolchain. |
| Slow first build from source | Normal — Rust compilation takes 2-4 minutes. Subsequent builds are cached. |
