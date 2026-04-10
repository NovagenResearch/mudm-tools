"""3D Cartesian projector — normalizes coordinates to [0, 1]³."""

from __future__ import annotations


class CartesianProjector3D:
    """Normalize 3D coordinates from world bounds to [0, 1]³.

    Parameters
    ----------
    bounds6 : tuple of 6 floats
        (xmin, ymin, zmin, xmax, ymax, zmax) in world coordinates.
    """

    def __init__(self, bounds6: tuple[float, float, float, float, float, float]) -> None:
        xmin, ymin, zmin, xmax, ymax, zmax = bounds6
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.dx = xmax - xmin if xmax != xmin else 1.0
        self.dy = ymax - ymin if ymax != ymin else 1.0
        self.dz = zmax - zmin if zmax != zmin else 1.0

    def project(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Project world (x, y, z) to normalized [0, 1]³."""
        return (
            (x - self.xmin) / self.dx,
            (y - self.ymin) / self.dy,
            (z - self.zmin) / self.dz,
        )

    def unproject(self, nx: float, ny: float, nz: float) -> tuple[float, float, float]:
        """Unproject normalized [0, 1]³ back to world coordinates."""
        return (
            nx * self.dx + self.xmin,
            ny * self.dy + self.ymin,
            nz * self.dz + self.zmin,
        )
