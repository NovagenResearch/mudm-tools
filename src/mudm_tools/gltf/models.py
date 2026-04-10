"""Pydantic configuration models for glTF export."""

from __future__ import annotations

from pydantic import BaseModel


class GltfConfig(BaseModel):
    """Configuration for MuDM -> glTF conversion.

    Attributes:
        include_metadata: Store MuDM properties in glTF ``extras``.
        y_up: Apply Z-up -> Y-up rotation (glTF standard is Y-up).
        default_color: RGBA color for default PBR material.
        feature_spacing: Gap between features when exporting a
            collection.  ``None`` (default) = no layout, coordinates
            are kept as-is.  ``0`` = auto (20 % of widest feature).
            A positive value sets a fixed gap in source coordinate units.
        grid_max_x: Max number of columns (X direction) before wrapping
            to a new row.  ``None`` = no limit.
        grid_max_y: Max number of rows (Y direction) before wrapping
            to a new layer.  ``None`` = no limit.
        grid_max_z: Max number of layers (Z direction).
        color_by: Property key used to look up per-feature material color.
            When set, ``color_map`` maps property values to RGBA colors.
        color_map: Mapping of property values to RGBA tuples.
        draco: Enable Draco mesh compression (``KHR_draco_mesh_compression``).
        draco_quantization_position: Quantization bits for vertex positions
            (1-30).
        draco_quantization_normal: Quantization bits for normal vectors
            (1-30).
        draco_compression_level: Draco compression level (0-10).
    """

    include_metadata: bool = True
    y_up: bool = True
    default_color: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    feature_spacing: float | None = None
    grid_max_x: int | None = None
    grid_max_y: int | None = None
    grid_max_z: int | None = None
    color_by: str | None = None
    color_map: dict[str, tuple[float, float, float, float]] | None = None
    draco: bool = False
    draco_quantization_position: int = 14
    draco_quantization_normal: int = 10
    draco_compression_level: int = 1
