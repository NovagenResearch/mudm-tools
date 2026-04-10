"""Convert an SWC file to Neuroglancer precomputed skeleton format and serve it.

Usage:
    python -m mudm.examples.swc_to_neuroglancer neuron.swc

    # Multiple SWC files:
    python -m mudm.examples.swc_to_neuroglancer neuron1.swc neuron2.swc neuron3.swc

    # Export only (no server):
    python -m mudm.examples.swc_to_neuroglancer neuron.swc --no-serve

    # Custom port:
    python -m mudm.examples.swc_to_neuroglancer neuron.swc --port 8080

Downloads:
    NeuroMorpho.Org has 200,000+ reconstructions:
    https://neuromorpho.org → search → download SWC
"""

from __future__ import annotations

import argparse
import functools
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from mudm_tools.swc import _parse_swc, swc_to_microjson
from mudm_tools.neuroglancer import write_skeleton
from mudm_tools.neuroglancer.properties_writer import write_segment_properties
from mudm_tools.neuroglancer.skeleton_writer import build_skeleton_info
from mudm_tools.neuroglancer.state import (
    build_skeleton_layer,
    build_viewer_state,
    viewer_state_to_url,
)


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
        # Only log errors, not every GET request
        if args and "200" not in str(args[1]):
            super().log_message(format, *args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SWC files to Neuroglancer precomputed format",
    )
    parser.add_argument(
        "swc_files",
        nargs="+",
        help="One or more .swc files to convert",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="neuroglancer_output",
        help="Output directory (default: neuroglancer_output)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=9000,
        help="Port to serve on (default: 9000)",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Export only, don't start the HTTP server",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    skel_dir = output_dir / "skeletons"

    # Convert each SWC file
    features = []
    segment_ids = []
    centroid_x, centroid_y, centroid_z = 0.0, 0.0, 0.0

    print(f"\nConverting {len(args.swc_files)} SWC file(s):\n")

    for i, swc_path in enumerate(args.swc_files):
        swc_path = Path(swc_path)
        if not swc_path.exists():
            print(f"  ERROR: {swc_path} not found, skipping")
            continue

        segment_id = i + 1
        feature = swc_to_microjson(str(swc_path))
        morphology = _parse_swc(str(swc_path))

        # Add the filename as a property
        feature.properties = {"name": swc_path.stem, "file": swc_path.name}

        # Write skeleton binary
        write_skeleton(skel_dir, segment_id, morphology)

        # Track centroid for camera positioning
        c = morphology.centroid3d()
        centroid_x += c[0]
        centroid_y += c[1]
        centroid_z += c[2]

        node_count = len(morphology.tree)
        bbox = morphology.bbox3d()
        span = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]

        print(f"  [{segment_id}] {swc_path.name}")
        print(f"      {node_count} nodes, span: {span[0]:.0f} x {span[1]:.0f} x {span[2]:.0f}")

        features.append(feature)
        segment_ids.append(segment_id)

    if not features:
        print("\nNo valid SWC files found.")
        return

    # Write segment properties
    write_segment_properties(skel_dir / "seg_props", features, segment_ids)

    # Rewrite info with segment_properties path
    info = build_skeleton_info(segment_properties="seg_props")
    (skel_dir / "info").write_text(json.dumps(info.to_info_dict(), indent=2))

    n = len(features)
    center = [centroid_x / n, centroid_y / n, centroid_z / n]

    print(f"\n  Output: {skel_dir}/")
    print(f"  Segments: {len(features)}")
    print(f"  Center: [{center[0]:.0f}, {center[1]:.0f}, {center[2]:.0f}]")

    # Build viewer URL
    source = f"precomputed://http://localhost:{args.port}/skeletons"
    layer = build_skeleton_layer("neurons", source)
    layer["segments"] = [str(sid) for sid in segment_ids]
    state = build_viewer_state([layer], position=center)
    url = viewer_state_to_url(state)

    print(f"\n  Viewer URL:\n  {url}\n")

    if args.no_serve:
        print("  Export complete. Run the server manually:")
        print(f"  python -m mudm.examples.neuroglancer_serve {output_dir}")
        return

    # Start server
    handler = functools.partial(CORSHandler, directory=str(output_dir))
    server = HTTPServer(("", args.port), handler)
    print(f"  Server running on http://localhost:{args.port}")
    print(f"  Copy the Viewer URL above into your browser.")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
