"""Annotate a neuron by SWC compartment type (soma, axon, dendrites).

Creates separate Neuroglancer skeleton segments for each SWC type so they
render in different colors and can be toggled independently.

Usage:
    python -m mudm.examples.annotate_by_type path/to/neuron.swc

    # Export only (no server):
    python -m mudm.examples.annotate_by_type neuron.swc --no-serve

    # Save the MuDM collection as JSON:
    python -m mudm.examples.annotate_by_type neuron.swc --save-json
"""

from __future__ import annotations

import argparse
import functools
import json
import math
from collections import defaultdict
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from geojson_pydantic import MultiLineString

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
)
from mudm_tools.swc import NeuronMorphology, SWCSample, _parse_swc
from mudm_tools.neuroglancer import write_skeleton
from mudm_tools.neuroglancer.properties_writer import write_segment_properties
from mudm_tools.neuroglancer.annotation_writer import write_annotations
from mudm_tools.neuroglancer.skeleton_writer import build_skeleton_info
from mudm_tools.neuroglancer.state import (
    build_annotation_layer,
    build_skeleton_layer,
    build_viewer_state,
    viewer_state_to_url,
)


# SWC type codes
SWC_TYPES = {
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
}


class CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        super().log_message(format, *args)


def split_by_type(morphology: NeuronMorphology) -> dict[int, NeuronMorphology]:
    """Split a neuron morphology into separate trees per SWC type.

    Each compartment type becomes its own NeuronMorphology with local edges.
    Nodes at type boundaries are duplicated into both compartments to keep
    edges connected.
    """
    by_id = {s.id: s for s in morphology.tree}

    # Group nodes by type
    type_nodes: dict[int, list[SWCSample]] = defaultdict(list)
    for s in morphology.tree:
        type_nodes[s.type].append(s)

    # For each type, include parent nodes at type boundaries so edges connect
    result: dict[int, NeuronMorphology] = {}
    for swc_type, nodes in type_nodes.items():
        node_ids = {s.id for s in nodes}
        extended_nodes = list(nodes)

        # Add parent bridge nodes (nodes from other types that are parents of this type)
        for s in nodes:
            if s.parent != -1 and s.parent not in node_ids:
                parent = by_id[s.parent]
                bridge = SWCSample(
                    id=parent.id,
                    type=swc_type,  # Tag with this type for rendering
                    x=parent.x, y=parent.y, z=parent.z,
                    r=parent.r,
                    parent=-1,  # Make it a root in this sub-tree
                )
                extended_nodes.append(bridge)
                node_ids.add(parent.id)

        # Ensure at least one root exists
        has_root = any(s.parent == -1 for s in extended_nodes)
        if not has_root and extended_nodes:
            # Make the first node a root
            first = extended_nodes[0]
            extended_nodes[0] = SWCSample(
                id=first.id, type=first.type,
                x=first.x, y=first.y, z=first.z,
                r=first.r, parent=-1,
            )

        # Fix parent references — remove parents not in this sub-tree
        final_ids = {s.id for s in extended_nodes}
        cleaned = []
        for s in extended_nodes:
            if s.parent != -1 and s.parent not in final_ids:
                s = SWCSample(
                    id=s.id, type=s.type,
                    x=s.x, y=s.y, z=s.z, r=s.r,
                    parent=-1,
                )
            cleaned.append(s)

        if cleaned:
            result[swc_type] = NeuronMorphology(
                type="NeuronMorphology", tree=cleaned
            )

    return result


def find_structural_points(
    morphology: NeuronMorphology,
) -> list[MuDMFeature]:
    """Find soma, branch points, and terminals as Point annotations."""
    by_id = {s.id: s for s in morphology.tree}
    children_count: dict[int, int] = defaultdict(int)
    for s in morphology.tree:
        if s.parent != -1:
            children_count[s.parent] += 1

    parent_ids = {s.parent for s in morphology.tree}
    features: list[MuDMFeature] = []

    for s in morphology.tree:
        # Soma centers
        if s.type == 1:
            features.append(MuDMFeature(
                type="Feature",
                geometry={"type": "Point", "coordinates": [s.x, s.y, s.z]},
                properties={
                    "annotation": "soma",
                    "node_id": s.id,
                    "radius": s.r,
                },
                featureClass="soma",
            ))

        # Branch points (2+ children)
        if children_count.get(s.id, 0) >= 2:
            features.append(MuDMFeature(
                type="Feature",
                geometry={"type": "Point", "coordinates": [s.x, s.y, s.z]},
                properties={
                    "annotation": "branch_point",
                    "node_id": s.id,
                    "branch_count": children_count[s.id],
                    "compartment": SWC_TYPES.get(s.type, str(s.type)),
                },
                featureClass="branch_point",
            ))

        # Terminals (leaf nodes, not soma)
        if s.id not in parent_ids and s.type != 1:
            features.append(MuDMFeature(
                type="Feature",
                geometry={"type": "Point", "coordinates": [s.x, s.y, s.z]},
                properties={
                    "annotation": "terminal",
                    "node_id": s.id,
                    "compartment": SWC_TYPES.get(s.type, str(s.type)),
                },
                featureClass="terminal",
            ))

    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a neuron by SWC type in Neuroglancer",
    )
    parser.add_argument("swc_file", help="Path to an .swc file")
    parser.add_argument("--output-dir", "-o", default="neuroglancer_output")
    parser.add_argument("--port", "-p", type=int, default=9000)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument("--save-json", action="store_true",
                        help="Save the MuDM collection as JSON")
    args = parser.parse_args()

    swc_path = Path(args.swc_file)
    if not swc_path.exists():
        print(f"Error: {swc_path} not found")
        return

    # Step 1: Load the full morphology
    morphology = _parse_swc(str(swc_path))
    print(f"\n  Loaded: {swc_path.name}")
    print(f"  Total nodes: {len(morphology.tree)}")

    # Count by type
    type_counts: dict[str, int] = defaultdict(int)
    for s in morphology.tree:
        type_counts[SWC_TYPES.get(s.type, f"type_{s.type}")] += 1
    for name, count in sorted(type_counts.items()):
        print(f"    {name}: {count} nodes")

    # Step 2: Split morphology by SWC type
    subtrees = split_by_type(morphology)

    # Step 3: Write each type as a separate skeleton segment
    output_dir = Path(args.output_dir)
    skel_dir = output_dir / "skeletons"
    skel_dir.mkdir(parents=True, exist_ok=True)

    segment_map: dict[int, str] = {}  # segment_id → type name
    seg_features: list[MuDMFeature] = []
    seg_ids: list[int] = []

    print(f"\n  Skeleton segments:")
    for segment_id, (swc_type, subtree) in enumerate(sorted(subtrees.items()), start=1):
        type_name = SWC_TYPES.get(swc_type, f"type_{swc_type}")
        segment_map[segment_id] = type_name

        write_skeleton(skel_dir, segment_id, subtree)

        seg_features.append(MuDMFeature(
            type="Feature",
            geometry={"type": "Point", "coordinates": [0, 0, 0]},
            properties={"name": type_name, "swc_type": swc_type, "node_count": len(subtree.tree)},
        ))
        seg_ids.append(segment_id)

        bbox = subtree.bbox3d()
        span = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
        print(f"    [{segment_id}] {type_name:20s} — {len(subtree.tree):5d} nodes, "
              f"span: {span[0]:.0f} x {span[1]:.0f} x {span[2]:.0f}")

    # Write segment properties (so Neuroglancer shows type names)
    write_segment_properties(skel_dir / "seg_props", seg_features, seg_ids)

    # Rewrite info with seg_props reference and um→nm scaling
    info = build_skeleton_info(segment_properties="seg_props", scale_um_to_nm=True)
    (skel_dir / "info").write_text(json.dumps(info.to_info_dict(), indent=2))

    # Step 4: Write structural annotations (soma, branch points, terminals)
    point_annotations = find_structural_points(morphology)
    ann_counts: dict[str, int] = defaultdict(int)
    for f in point_annotations:
        ann_counts[f.featureClass] += 1

    print(f"\n  Point annotations:")
    for cls, count in sorted(ann_counts.items()):
        print(f"    {cls}: {count}")

    ann_dir = output_dir / "point_annotations"
    if point_annotations:
        write_annotations(ann_dir, point_annotations, "point")

    # Step 5: Build MuDMFeatureCollection (for --save-json)
    all_features: list[MuDMFeature] = []

    # Add each compartment as a MuDMFeature (MultiLineString skeleton edges)
    for swc_type, subtree in sorted(subtrees.items()):
        type_name = SWC_TYPES.get(swc_type, f"type_{swc_type}")
        # Convert NeuronMorphology subtree → MultiLineString3D (edge segments)
        by_id = {s.id: s for s in subtree.tree}
        lines = []
        for s in subtree.tree:
            if s.parent != -1 and s.parent in by_id:
                p = by_id[s.parent]
                lines.append([[p.x, p.y, p.z], [s.x, s.y, s.z]])
        geom = MultiLineString(type="MultiLineString", coordinates=lines) if lines else None
        all_features.append(MuDMFeature(
            type="Feature",
            geometry=geom,
            properties={"compartment": type_name, "swc_type": swc_type, "node_count": len(subtree.tree)},
            featureClass=type_name,
        ))

    all_features.extend(point_annotations)

    collection = MuDMFeatureCollection(
        type="FeatureCollection",
        features=all_features,
        properties={
            "source": swc_path.name,
            "total_nodes": len(morphology.tree),
            "compartments": dict(type_counts),
            "annotation_counts": dict(ann_counts),
        },
    )

    if args.save_json:
        json_path = output_dir / f"{swc_path.stem}_annotated.json"
        json_path.write_text(collection.model_dump_json(indent=2))
        print(f"\n  MuDM saved: {json_path}")

    # Step 6: Build viewer URL
    port = args.port
    layers = []

    # Skeleton layer with all segments pre-selected
    source = f"precomputed://http://localhost:{port}/skeletons"
    skel_layer = build_skeleton_layer("neuron_compartments", source)
    skel_layer["segments"] = [str(sid) for sid in seg_ids]
    layers.append(skel_layer)

    # Annotation layer
    if point_annotations:
        source = f"precomputed://http://localhost:{port}/point_annotations"
        layers.append(build_annotation_layer("structural_markers", source))

    # Position must be in nm (model space) — scale the µm centroid by 1000
    center_um = morphology.centroid3d()
    center_nm = [c * 1000 for c in center_um]
    # Zoom: use the largest span (in nm) × 1.2 to frame the neuron tightly
    bbox = morphology.bbox3d()
    spans_nm = [(bbox[i+3] - bbox[i]) * 1000 for i in range(3)]
    zoom_nm = max(spans_nm) * 1.2
    state = build_viewer_state(layers, position=center_nm, projection_scale=zoom_nm)
    url = viewer_state_to_url(state)

    # Save URL to file (avoids terminal copy-paste corruption on long URLs)
    url_path = output_dir / "viewer_url.txt"
    url_path.write_text(url)

    # Create HTML redirect page (bypasses browser URL encoding issues)
    state_json = json.dumps(state, separators=(",", ":"))
    html = (
        "<!DOCTYPE html><html><head><title>Open Neuroglancer</title></head>"
        "<body><p>Redirecting to Neuroglancer...</p><script>"
        f"var state={state_json};"
        "var url='https://neuroglancer-demo.appspot.com/#!'+JSON.stringify(state);"
        "window.location.href=url;"
        "</script></body></html>"
    )
    html_path = output_dir / "open_viewer.html"
    html_path.write_text(html)

    print(f"\n  Open http://localhost:{port}/open_viewer.html in your browser.\n")

    if args.no_serve:
        return

    handler = functools.partial(CORSHandler, directory=str(output_dir))
    server = HTTPServer(("", port), handler)
    print(f"  Server running on http://localhost:{port}")
    print(f"  Copy the Viewer URL into your browser. Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
