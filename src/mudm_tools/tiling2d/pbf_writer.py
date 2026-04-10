"""Thin Python wrapper for PBF tile generation via Rust."""

from __future__ import annotations

from pathlib import Path


def generate_pbf(
    generator,
    output_path: str | Path,
    world_bounds: tuple[float, float, float, float],
    *,
    extent: int = 4096,
    simplify: bool = True,
    layer_name: str = "geojsonLayer",
) -> int:
    """Generate PBF vector tiles from a StreamingTileGenerator2D.

    Args:
        generator: A ``StreamingTileGenerator2D`` instance with features added.
        output_path: Directory to write tiles into ({z}/{x}/{y}.pbf).
        world_bounds: ``(xmin, ymin, xmax, ymax)`` in world coordinates.
        extent: MVT tile extent (default 4096).
        simplify: Whether to apply Douglas-Peucker simplification at coarse zooms.
        layer_name: MVT layer name (default "geojsonLayer").

    Returns:
        Number of tiles written.
    """
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    return generator.generate_pbf(
        str(out), world_bounds, extent, simplify, layer_name,
    )
