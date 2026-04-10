"""muDM format converters — standardized entry points for source data ingestion.

Each converter transforms a specific source format into muDM tiled output
(MVT + Parquet + optional raster tiles).

Usage:
    from mudm_tools.converters import convert

    convert("xenium", input_dir="data/outs", output_dir="tiles/sample",
            config={"temp_dir": "/data/tmp"})

CLI:
    mudm convert --format xenium --input data/outs --output tiles/sample
"""

from __future__ import annotations

from typing import Any

_REGISTRY: dict[str, type] = {}


def register(name: str):
    """Decorator to register a converter class."""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def convert(
    format: str,
    input_dir: str,
    output_dir: str,
    config: dict[str, Any] | None = None,
) -> dict:
    """Run a converter by format name.

    Args:
        format: Converter name (e.g., "xenium", "obj", "geojson").
        input_dir: Path to source data directory or file.
        output_dir: Path for tiled output.
        config: Optional dict of converter-specific settings.

    Returns:
        Dict with conversion metadata (feature counts, timing, etc.).
    """
    if format not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown format {format!r}. Available: {available}")

    converter = _REGISTRY[format]()
    return converter.convert(input_dir, output_dir, config or {})


def list_formats() -> list[str]:
    """Return registered converter format names."""
    return sorted(_REGISTRY.keys())


# Import converters to trigger registration
from . import xenium  # noqa: F401, E402
from . import obj  # noqa: F401, E402
from . import geojson  # noqa: F401, E402
