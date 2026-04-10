"""Thin Python wrapper for PBF tile reading via Rust."""

from __future__ import annotations

from pathlib import Path


def read_pbf(
    path: str | Path,
    world_bounds: tuple[float, float, float, float],
    *,
    zoom: int | None = None,
    tile_x: int | None = None,
    tile_y: int | None = None,
) -> list[dict]:
    """Read PBF tiles back to feature dicts.

    Args:
        path: Directory containing ``{z}/{x}/{y}.pbf`` tile tree.
        world_bounds: ``(xmin, ymin, xmax, ymax)`` used during tile generation.
        zoom: Filter to a specific zoom level.
        tile_x: Filter to a specific tile X.
        tile_y: Filter to a specific tile Y.

    Returns:
        List of dicts with keys: zoom, tile_x, tile_y, feature_id,
        geom_type, positions (numpy float32 array), ring_lengths, tags.
    """
    from mudm_tools._rs import read_pbf as _read_pbf

    return _read_pbf(str(path), world_bounds, zoom, tile_x, tile_y)
