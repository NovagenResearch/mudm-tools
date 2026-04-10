"""Pydantic config model for Neuroglancer precomputed legacy mesh info.

Produces the ``{"@type": "neuroglancer_legacy_mesh"}`` JSON structure
that Neuroglancer expects in the ``info`` file of a mesh data source.

Reference: https://github.com/google/neuroglancer/blob/master/
    src/neuroglancer/datasource/precomputed/meshes.md
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class MeshInfo(BaseModel):
    """Info JSON for a ``precomputed://`` legacy mesh source."""

    at_type: str = "neuroglancer_legacy_mesh"
    segment_properties: Optional[str] = None

    def to_info_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"@type": self.at_type}
        if self.segment_properties is not None:
            d["segment_properties"] = self.segment_properties
        return d
