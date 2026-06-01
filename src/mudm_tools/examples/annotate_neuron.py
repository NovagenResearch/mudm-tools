"""Annotate a neuron morphology in 3D using MuDM.

Demonstrates how to use the MuDM data model to:
  1. Load a neuron from SWC
  2. Auto-detect structural features (soma, branch points, terminals)
  3. Add manual annotations (measurement lines, region labels)
  4. Export everything as a single MuDMFeatureCollection
  5. View in Neuroglancer with skeleton + annotations overlaid

Usage:
    python -m mudm.examples.annotate_neuron tests/fixtures/sample_neuron.swc

    # With your own SWC file:
    python -m mudm.examples.annotate_neuron path/to/neuron.swc
"""

from __future__ import annotations

import argparse
import functools
import math
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

from mudm.model import (
    MuDMFeature,
    MuDMFeatureCollection,
)
from mudm_tools.swc import NeuronMorphology, _parse_swc, swc_to_microjson
from mudm_tools.neuroglancer import to_neuroglancer, write_skeleton
from mudm_tools.neuroglancer.state import (
    build_annotation_layer,
    build_skeleton_layer,
    build_viewer_state,
    viewer_state_to_url,
)

# ---------------------------------------------------------------------------
# SWC type codes → human-readable names
# ---------------------------------------------------------------------------
SWC_TYPE_NAMES = {
    0: "undefined",
    1: "soma",
    2: "axon",
    3: "basal_dendrite",
    4: "apical_dendrite",
    5: "fork_point",
    6: "end_point",
    7: "custom",
}


# ---------------------------------------------------------------------------
# Annotation helpers — extract structural features from the morphology
# ---------------------------------------------------------------------------


def find_soma(morphology: NeuronMorphology) -> list[MuDMFeature]:
    """Create Point annotations for all soma nodes."""
    features = []
    for sample in morphology.tree:
        if sample.type == 1:  # soma
            features.append(
                MuDMFeature(
                    type="Feature",
                    geometry={
                        "type": "Point",
                        "coordinates": [sample.x, sample.y, sample.z],
                    },
                    properties={
                        "annotation": "soma",
                        "node_id": sample.id,
                        "radius": sample.r,
                    },
                    featureClass="soma",
                )
            )
    return features


def find_branch_points(morphology: NeuronMorphology) -> list[MuDMFeature]:
    """Create Point annotations at bifurcation points.

    A branch point is any node that has 2+ children.
    """
    # Count children per node
    children_count: dict[int, int] = {}
    for sample in morphology.tree:
        if sample.parent != -1:
            children_count[sample.parent] = children_count.get(sample.parent, 0) + 1

    by_id = {s.id: s for s in morphology.tree}
    features = []
    for node_id, count in children_count.items():
        if count >= 2:
            s = by_id[node_id]
            features.append(
                MuDMFeature(
                    type="Feature",
                    geometry={
                        "type": "Point",
                        "coordinates": [s.x, s.y, s.z],
                    },
                    properties={
                        "annotation": "branch_point",
                        "node_id": s.id,
                        "branch_count": count,
                        "swc_type": SWC_TYPE_NAMES.get(s.type, str(s.type)),
                    },
                    featureClass="branch_point",
                )
            )
    return features


def find_terminals(morphology: NeuronMorphology) -> list[MuDMFeature]:
    """Create Point annotations at terminal (leaf) nodes.

    A terminal is any node that has no children.
    """
    parent_ids = {s.parent for s in morphology.tree}
    features = []
    for s in morphology.tree:
        if s.id not in parent_ids and s.type != 1:  # not a parent, not soma
            features.append(
                MuDMFeature(
                    type="Feature",
                    geometry={
                        "type": "Point",
                        "coordinates": [s.x, s.y, s.z],
                    },
                    properties={
                        "annotation": "terminal",
                        "node_id": s.id,
                        "swc_type": SWC_TYPE_NAMES.get(s.type, str(s.type)),
                    },
                    featureClass="terminal",
                )
            )
    return features


def measure_distance(
    morphology: NeuronMorphology,
    from_id: int,
    to_id: int,
) -> Optional[MuDMFeature]:
    """Create a LineString measurement between two nodes."""
    by_id = {s.id: s for s in morphology.tree}
    if from_id not in by_id or to_id not in by_id:
        return None

    a, b = by_id[from_id], by_id[to_id]
    dist = math.sqrt((b.x - a.x) ** 2 + (b.y - a.y) ** 2 + (b.z - a.z) ** 2)

    return MuDMFeature(
        type="Feature",
        geometry={
            "type": "LineString",
            "coordinates": [[a.x, a.y, a.z], [b.x, b.y, b.z]],
        },
        properties={
            "annotation": "measurement",
            "from_node": from_id,
            "to_node": to_id,
            "distance": round(dist, 2),
            "unit": "um",
        },
        featureClass="measurement",
    )


def trace_path_to_soma(
    morphology: NeuronMorphology,
    start_id: int,
) -> Optional[MuDMFeature]:
    """Create a LineString tracing the path from a node back to the soma."""
    by_id = {s.id: s for s in morphology.tree}
    if start_id not in by_id:
        return None

    path_coords = []
    current = by_id[start_id]
    path_length = 0.0

    while True:
        path_coords.append([current.x, current.y, current.z])
        if current.parent == -1:
            break
        prev = current
        current = by_id[current.parent]
        path_length += math.sqrt(
            (current.x - prev.x) ** 2 + (current.y - prev.y) ** 2 + (current.z - prev.z) ** 2
        )

    if len(path_coords) < 2:
        return None

    return MuDMFeature(
        type="Feature",
        geometry={
            "type": "LineString",
            "coordinates": path_coords,
        },
        properties={
            "annotation": "path_to_soma",
            "start_node": start_id,
            "path_length": round(path_length, 2),
            "unit": "um",
            "num_segments": len(path_coords) - 1,
        },
        featureClass="path_trace",
    )


# ---------------------------------------------------------------------------
# Build the annotated MuDMFeatureCollection
# ---------------------------------------------------------------------------


def annotate_neuron(swc_path: str) -> MuDMFeatureCollection:
    """Load SWC and build a fully annotated MuDMFeatureCollection.

    Returns a collection with:
      - The neuron morphology as a skeleton feature
      - Soma markers (Point)
      - Branch point markers (Point)
      - Terminal markers (Point)
      - Measurement lines between interesting points
      - Path traces from terminals back to soma
    """
    # Load the neuron — TIN geometry for the collection, NeuronMorphology for analysis
    neuron_feature = swc_to_microjson(swc_path)
    morphology = _parse_swc(swc_path)
    neuron_feature.properties = {
        "name": Path(swc_path).stem,
        "file": Path(swc_path).name,
        "node_count": len(morphology.tree),
    }
    neuron_feature.featureClass = "neuron_morphology"

    features: list[MuDMFeature] = [neuron_feature]

    # Auto-detect structural annotations
    somas = find_soma(morphology)
    branches = find_branch_points(morphology)
    terminals = find_terminals(morphology)

    features.extend(somas)
    features.extend(branches)
    features.extend(terminals)

    # Add measurement from soma to each terminal
    soma_ids = [s.id for s in morphology.tree if s.type == 1]
    terminal_ids = [int(f.properties["node_id"]) for f in terminals]

    if soma_ids:
        for tid in terminal_ids:
            m = measure_distance(morphology, soma_ids[0], tid)
            if m:
                features.append(m)

    # Add path traces from each terminal back to soma
    for tid in terminal_ids:
        trace = trace_path_to_soma(morphology, tid)
        if trace:
            features.append(trace)

    return MuDMFeatureCollection(
        type="FeatureCollection",
        features=features,
        properties={
            "source": Path(swc_path).name,
            "annotations": {
                "soma_markers": len(somas),
                "branch_points": len(branches),
                "terminals": len(terminals),
                "measurements": len(terminal_ids),
                "path_traces": len(terminal_ids),
            },
        },
    )


# ---------------------------------------------------------------------------
# CORS server
# ---------------------------------------------------------------------------


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
        if args and "200" not in str(args[1]):
            super().log_message(format, *args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a neuron morphology in 3D using MuDM",
    )
    parser.add_argument("swc_file", help="Path to an .swc file")
    parser.add_argument("--output-dir", "-o", default="neuroglancer_output")
    parser.add_argument("--port", "-p", type=int, default=9000)
    parser.add_argument("--no-serve", action="store_true")
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save the MuDM collection as JSON",
    )
    args = parser.parse_args()

    swc_path = Path(args.swc_file)
    if not swc_path.exists():
        print(f"Error: {swc_path} not found")
        return

    # Step 1: Build annotated MuDMFeatureCollection
    print(f"\n  Loading: {swc_path.name}")
    collection = annotate_neuron(str(swc_path))

    ann = collection.properties["annotations"]
    print("\n  MuDM annotations created:")
    print(f"    Soma markers:     {ann['soma_markers']}")
    print(f"    Branch points:    {ann['branch_points']}")
    print(f"    Terminals:        {ann['terminals']}")
    print(f"    Measurements:     {ann['measurements']}")
    print(f"    Path traces:      {ann['path_traces']}")
    print(f"    Total features:   {len(collection.features)}")

    # Optional: save the raw MuDM
    if args.save_json:
        json_path = Path(args.output_dir) / f"{swc_path.stem}_annotated.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(collection.model_dump_json(indent=2))
        print(f"\n  MuDM saved: {json_path}")

    # Step 2: Export to Neuroglancer
    output_dir = Path(args.output_dir)

    # Write skeleton directly (to_neuroglancer handles Point/LineString only)
    morphology = _parse_swc(str(swc_path))
    skel_dir = output_dir / "skeletons"
    write_skeleton(skel_dir, segment_id=1, morphology=morphology)

    # Write Point/LineString annotations via orchestrator
    result = to_neuroglancer(collection, output_dir)

    print("\n  Neuroglancer output:")
    print(f"    skeletons: {skel_dir}/")
    for key, path in result["paths"].items():
        print(f"    {key}: {path}/")

    # Step 3: Build viewer URL with all layers and segments pre-selected
    layers = []

    source = f"precomputed://http://localhost:{args.port}/skeletons"
    layer = build_skeleton_layer("neuron_skeleton", source)
    layer["segments"] = ["1"]
    layers.append(layer)

    if "point_annotations" in result["paths"]:
        source = f"precomputed://http://localhost:{args.port}/point_annotations"
        layers.append(build_annotation_layer("annotations_3d", source))

    if "line_annotations" in result["paths"]:
        source = f"precomputed://http://localhost:{args.port}/line_annotations"
        layers.append(build_annotation_layer("measurements_and_paths", source))

    # Center on the neuron
    center = list(morphology.centroid3d())
    state = build_viewer_state(layers, position=center)
    url = viewer_state_to_url(state)

    print(f"\n  Viewer URL:\n  {url}\n")

    if args.no_serve:
        return

    handler = functools.partial(CORSHandler, directory=str(output_dir))
    server = HTTPServer(("", args.port), handler)
    print(f"  Server running on http://localhost:{args.port}")
    print("  Copy the Viewer URL above into your browser.")
    print("  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
