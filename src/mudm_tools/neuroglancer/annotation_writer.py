"""Neuroglancer precomputed annotation writer.

Converts MuDM Point and LineString features to Neuroglancer's
precomputed annotation binary format.

Binary layout per spatial chunk (little-endian):
    uint64       count
    float32      coordinates[count × D]  (D=3 for point, D=6 for line)
    uint64       annotation_ids[count]

Reference: https://github.com/google/neuroglancer/blob/master/
    src/neuroglancer/datasource/precomputed/annotations.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

from geojson_pydantic import LineString, Point

from mudm.model import MuDMFeature
from ._binary import pack_float32_array, pack_uint64, pack_uint64_array
from .models import AnnotationInfo, AnnotationSpatialEntry


def points_to_annotation_binary(
    points: Sequence[Tuple[float, float, float]],
    annotation_ids: Sequence[int],
) -> bytes:
    """Encode point annotations as Neuroglancer binary.

    Args:
        points: Sequence of (x, y, z) coordinates.
        annotation_ids: Unique ID for each annotation.

    Returns:
        Raw bytes in Neuroglancer annotation binary format.
    """
    count = len(points)
    buf = bytearray()
    buf += pack_uint64(count)

    coords: list[float] = []
    for x, y, z in points:
        coords.extend([x, y, z])
    buf += pack_float32_array(coords)

    buf += pack_uint64_array(list(annotation_ids))
    return bytes(buf)


def lines_to_annotation_binary(
    lines: Sequence[Tuple[float, float, float, float, float, float]],
    annotation_ids: Sequence[int],
) -> bytes:
    """Encode line annotations as Neuroglancer binary.

    Each line is (x1, y1, z1, x2, y2, z2).

    Args:
        lines: Sequence of 6-tuples (start_xyz + end_xyz).
        annotation_ids: Unique ID for each annotation.

    Returns:
        Raw bytes in Neuroglancer annotation binary format.
    """
    count = len(lines)
    buf = bytearray()
    buf += pack_uint64(count)

    coords: list[float] = []
    for line in lines:
        coords.extend(line)
    buf += pack_float32_array(coords)

    buf += pack_uint64_array(list(annotation_ids))
    return bytes(buf)


def _compute_bounds(
    features: Sequence[MuDMFeature],
    annotation_type: Literal["point", "line"],
) -> Tuple[List[float], List[float]]:
    """Compute bounding box from features."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for feat in features:
        geom = feat.geometry
        if annotation_type == "point" and isinstance(geom, Point):
            coords = geom.coordinates
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            zs.append(float(coords[2]) if len(coords) > 2 else 0.0)
        elif annotation_type == "line" and isinstance(geom, LineString):
            for pos in geom.coordinates:
                xs.append(float(pos[0]))
                ys.append(float(pos[1]))
                zs.append(float(pos[2]) if len(pos) > 2 else 0.0)

    if not xs:
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]

    return (
        [min(xs), min(ys), min(zs)],
        [max(xs), max(ys), max(zs)],
    )


_NG_ANNOTATION_TYPE = {"point": "POINT", "line": "LINE"}


def write_annotations(
    output_dir: str | Path,
    features: Sequence[MuDMFeature],
    annotation_type: Literal["point", "line"],
) -> Path:
    """Write annotations to Neuroglancer precomputed directory.

    Creates:
        {output_dir}/info                — JSON info file
        {output_dir}/by_id/              — empty dir (required by Neuroglancer)
        {output_dir}/spatial0/0_0_0      — binary annotation data

    Args:
        output_dir: Directory to write to.
        features: MuDM features with Point or LineString geometry.
        annotation_type: "point" or "line".

    Returns:
        Path to the output directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    lower, upper = _compute_bounds(features, annotation_type)

    # Chunk size = full extent
    extent = [high - low if high > low else 1.0 for low, high in zip(lower, upper)]

    # Build info — Neuroglancer expects UPPERCASE annotation_type
    ng_type = _NG_ANNOTATION_TYPE[annotation_type]
    spatial = AnnotationSpatialEntry(
        chunk_size=extent,
        grid_shape=[1, 1, 1],
        key="spatial0",
    )
    info = AnnotationInfo(
        annotation_type=ng_type,  # type: ignore[arg-type]
        lower_bound=lower,
        upper_bound=upper,
        spatial=[spatial],
    )
    (out / "info").write_text(json.dumps(info.to_info_dict(), indent=2))

    # Create by_id directory (Neuroglancer checks for it)
    (out / "by_id").mkdir(parents=True, exist_ok=True)

    # Build binary
    if annotation_type == "point":
        points: list[Tuple[float, float, float]] = []
        ids: list[int] = []
        for i, feat in enumerate(features):
            if isinstance(feat.geometry, Point):
                c = feat.geometry.coordinates
                z = float(c[2]) if len(c) > 2 else 0.0
                points.append((float(c[0]), float(c[1]), z))
                ids.append(i)
        binary = points_to_annotation_binary(points, ids)
    else:  # line
        line_data: list[Tuple[float, float, float, float, float, float]] = []
        ids = []
        for i, feat in enumerate(features):
            if isinstance(feat.geometry, LineString):
                coords = feat.geometry.coordinates
                # Each consecutive pair of coordinates forms a line segment
                for j in range(len(coords) - 1):
                    c0, c1 = coords[j], coords[j + 1]
                    z0 = float(c0[2]) if len(c0) > 2 else 0.0
                    z1 = float(c1[2]) if len(c1) > 2 else 0.0
                    line_data.append(
                        (
                            float(c0[0]),
                            float(c0[1]),
                            z0,
                            float(c1[0]),
                            float(c1[1]),
                            z1,
                        )
                    )
                    ids.append(i)
        binary = lines_to_annotation_binary(line_data, ids)

    # Write spatial chunk
    spatial_dir = out / "spatial0"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    (spatial_dir / "0_0_0").write_bytes(binary)

    return out
