"""Pydantic configuration models for Arrow/GeoParquet export."""

from __future__ import annotations

from pydantic import BaseModel


class ArrowConfig(BaseModel):
    """Configuration for MuDM -> Arrow/GeoParquet conversion.

    Attributes:
        primary_geometry_column: Name of the WKB geometry column.
    """

    primary_geometry_column: str = "geometry"
