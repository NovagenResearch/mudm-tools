"""Octree spatial index for 3D vector tiles.

Splits features into 8 octants per level via sequential axis clipping
(X → Y → Z). Uses an iterative stack-based approach to avoid deep
recursion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .clip3d import clip_3d
from .tile3d import create_tile_3d


@dataclass
class OctreeConfig:
    """Configuration for 3D octree tiling.

    Parameters
    ----------
    bounds : tuple of 6 floats
        World bounds (xmin, ymin, zmin, xmax, ymax, zmax).
    min_zoom : int
        Minimum zoom level to generate.
    max_zoom : int
        Maximum zoom level to generate.
    extent : int
        XY tile extent in integer coordinates.
    extent_z : int
        Z tile extent in integer coordinates.
    tolerance : float
        Simplification tolerance (in normalized space) per zoom.
    buffer : float
        Tile buffer in normalized space (fraction of tile size).
    """

    bounds: tuple[float, float, float, float, float, float] = (
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    )
    min_zoom: int = 0
    max_zoom: int = 4
    extent: int = 4096
    extent_z: int = 4096
    tolerance: float = 0.0
    buffer: float = 0.0


class Octree:
    """Octree spatial index for 3D tiling.

    Builds tiles by recursively splitting features into 8 octants
    via axis-parallel clipping. Uses an iterative stack to avoid
    deep recursion.
    """

    def __init__(self, features: list[dict], config: OctreeConfig) -> None:
        self.config = config
        self._tiles: dict[tuple[int, int, int, int], dict] = {}
        self._features = features
        self._build()

    # Small epsilon to ensure points at exactly 1.0 are included
    # in the topmost tile (half-open interval [k1, k2) needs k2 > 1.0)
    _EPS = 1e-10

    def _build(self) -> None:
        """Build octree by iteratively splitting from min to max zoom."""
        # Stack entries: (zoom, x, y, d, features)
        stack: list[tuple[int, int, int, int, list[dict]]] = [
            (0, 0, 0, 0, self._features),
        ]

        while stack:
            z, x, y, d, feats = stack.pop()

            if not feats:
                continue

            # Store tile at this level if within zoom range
            if z >= self.config.min_zoom:
                tile = create_tile_3d(feats, z, x, y, d)
                self._tiles[(z, x, y, d)] = tile

            # Stop splitting beyond max zoom
            if z >= self.config.max_zoom:
                continue

            # Split into 8 octants
            nz = z + 1

            # Actual tile boundaries in normalized space
            n = 1 << nz
            x0 = (x * 2) / n
            x1 = (x * 2 + 1) / n
            x2 = (x * 2 + 2) / n
            y0 = (y * 2) / n
            y1 = (y * 2 + 1) / n
            y2 = (y * 2 + 2) / n
            d0 = (d * 2) / n
            d1 = (d * 2 + 1) / n
            d2 = (d * 2 + 2) / n

            buf = self.config.buffer / n if self.config.buffer > 0 else 0.0
            eps = self._EPS

            # Pad the upper bound of the topmost tile to catch boundary points
            x2_pad = x2 + eps if x * 2 + 2 == n else x2
            y2_pad = y2 + eps if y * 2 + 2 == n else y2
            d2_pad = d2 + eps if d * 2 + 2 == n else d2

            # Clip X → two halves
            left = clip_3d(feats, x0 - buf, x1 + buf, 0)
            right = clip_3d(feats, x1 - buf, x2_pad + buf, 0)

            # Clip Y → four quadrants
            for x_feats, nx in [(left, x * 2), (right, x * 2 + 1)]:
                if not x_feats:
                    continue
                bottom = clip_3d(x_feats, y0 - buf, y1 + buf, 1)
                top = clip_3d(x_feats, y1 - buf, y2_pad + buf, 1)

                # Clip Z → eight octants
                for y_feats, ny in [(bottom, y * 2), (top, y * 2 + 1)]:
                    if not y_feats:
                        continue
                    front = clip_3d(y_feats, d0 - buf, d1 + buf, 2)
                    back = clip_3d(y_feats, d1 - buf, d2_pad + buf, 2)

                    for z_feats, nd in [(front, d * 2), (back, d * 2 + 1)]:
                        if z_feats:
                            stack.append((nz, nx, ny, nd, z_feats))

    def get_tile(self, z: int, x: int, y: int, d: int) -> dict | None:
        """Get a tile by coordinates, or None if empty."""
        return self._tiles.get((z, x, y, d))

    def tiles_at_zoom(self, z: int) -> list[dict]:
        """Return all non-empty tiles at a given zoom level."""
        return [
            tile for (tz, tx, ty, td), tile in self._tiles.items()
            if tz == z
        ]

    @property
    def all_tiles(self) -> dict[tuple[int, int, int, int], dict]:
        """All generated tiles keyed by (z, x, y, d)."""
        return self._tiles
