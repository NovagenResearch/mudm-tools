"""Pydantic config models for Neuroglancer info JSON schemas.

Each model produces the exact JSON structure Neuroglancer expects
via its ``to_info_dict()`` method.
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Skeleton info
# ---------------------------------------------------------------------------

class VertexAttributeInfo(BaseModel):
    """A per-vertex attribute in a skeleton (e.g. radius, type)."""

    id: str
    data_type: Literal["float32", "uint8", "uint32"]
    num_components: int = 1

    def to_info_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "data_type": self.data_type,
            "num_components": self.num_components,
        }


class SkeletonInfo(BaseModel):
    """Info JSON for a ``precomputed://`` skeleton source."""

    at_type: Literal["neuroglancer_skeletons"] = "neuroglancer_skeletons"
    transform: Optional[List[float]] = None  # 12-element column-major 4×3
    vertex_attributes: List[VertexAttributeInfo] = []
    segment_properties: Optional[str] = None

    def to_info_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"@type": self.at_type}
        if self.transform is not None:
            d["transform"] = self.transform
        if self.vertex_attributes:
            d["vertex_attributes"] = [
                va.to_info_dict() for va in self.vertex_attributes
            ]
        if self.segment_properties is not None:
            d["segment_properties"] = self.segment_properties
        return d


# ---------------------------------------------------------------------------
# Annotation info
# ---------------------------------------------------------------------------

class AnnotationDimension(BaseModel):
    """Spatial dimension metadata for annotations."""

    names: List[str] = ["x", "y", "z"]
    units: List[str] = ["nm", "nm", "nm"]
    scales: List[float] = [1.0, 1.0, 1.0]

    def to_info_dict(self) -> dict[str, Any]:
        return {
            "dimensions": {
                name: [scale, unit]
                for name, scale, unit in zip(self.names, self.scales, self.units)
            },
        }


class AnnotationSpatialEntry(BaseModel):
    """A single spatial index entry."""

    chunk_size: List[float]
    grid_shape: List[int] = [1, 1, 1]
    key: str = "spatial0"
    limit: int = 0  # 0 = unlimited

    def to_info_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "grid_shape": self.grid_shape,
            "key": self.key,
            "limit": self.limit,
        }


class AnnotationPropertySpec(BaseModel):
    """A per-annotation property definition."""

    id: str
    type: Literal["float32", "uint8", "uint16", "uint32", "int8", "int16", "int32", "rgb", "rgba"]
    description: Optional[str] = None
    enum_values: Optional[List[str]] = None
    enum_labels: Optional[List[str]] = None

    def to_info_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.description is not None:
            d["description"] = self.description
        if self.enum_values is not None:
            d["enum_values"] = self.enum_values
        if self.enum_labels is not None:
            d["enum_labels"] = self.enum_labels
        return d


class AnnotationRelationship(BaseModel):
    """A relationship between annotations and segments."""

    id: str
    key: str

    def to_info_dict(self) -> dict[str, Any]:
        return {"id": self.id, "key": self.key}


class AnnotationInfo(BaseModel):
    """Info JSON for a ``precomputed://`` annotation source."""

    at_type: Literal["neuroglancer_annotations_v1"] = "neuroglancer_annotations_v1"
    annotation_type: Literal["POINT", "LINE", "AXIS_ALIGNED_BOUNDING_BOX", "ELLIPSOID"]
    dimensions: AnnotationDimension = AnnotationDimension()
    lower_bound: List[float] = [0.0, 0.0, 0.0]
    upper_bound: List[float] = [1.0, 1.0, 1.0]
    properties: List[AnnotationPropertySpec] = []
    relationships: List[AnnotationRelationship] = []
    by_id: dict[str, str] = {"key": "by_id"}
    spatial: List[AnnotationSpatialEntry] = []

    def to_info_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "@type": self.at_type,
            "annotation_type": self.annotation_type,
            **self.dimensions.to_info_dict(),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "properties": [p.to_info_dict() for p in self.properties],
            "relationships": [r.to_info_dict() for r in self.relationships],
            "by_id": self.by_id,
        }
        if self.spatial:
            d["spatial"] = [s.to_info_dict() for s in self.spatial]
        return d


# ---------------------------------------------------------------------------
# Segment properties info
# ---------------------------------------------------------------------------

class SegmentPropertyField(BaseModel):
    """A single property field in segment_properties."""

    id: str
    description: Optional[str] = None
    type: Literal["label", "number", "string", "tags"]
    data_type: Optional[Literal[
        "float32", "int8", "uint8", "int16", "uint16", "int32", "uint32"
    ]] = None
    values: List[Any] = []
    tags: Optional[List[str]] = None

    def to_info_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.description is not None:
            d["description"] = self.description
        if self.type == "number" and self.data_type is not None:
            d["data_type"] = self.data_type
        if self.type in ("label", "number", "string"):
            d["values"] = self.values
        elif self.type == "tags" and self.tags is not None:
            d["tags"] = self.tags
            d["values"] = self.values
        return d


class SegmentPropertiesInfo(BaseModel):
    """Info JSON for ``neuroglancer_segment_properties``."""

    at_type: Literal["neuroglancer_segment_properties"] = (
        "neuroglancer_segment_properties"
    )
    inline: Optional[SegmentPropertiesInline] = None  # type: ignore[name-defined]  # forward ref

    def to_info_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"@type": self.at_type}
        if self.inline is not None:
            d["inline"] = self.inline.to_info_dict()
        return d


class SegmentPropertiesInline(BaseModel):
    """The inline block of segment_properties info JSON."""

    ids: List[str] = []
    properties: List[SegmentPropertyField] = []

    def to_info_dict(self) -> dict[str, Any]:
        return {
            "ids": self.ids,
            "properties": [p.to_info_dict() for p in self.properties],
        }


# Rebuild SegmentPropertiesInfo now that SegmentPropertiesInline is defined
SegmentPropertiesInfo.model_rebuild()
