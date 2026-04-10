"""Serve Neuroglancer precomputed data with CORS headers.

Neuroglancer runs in the browser and fetches data via HTTP, so the
server must include Access-Control-Allow-Origin headers.

Usage:
    # 1. First generate the data
    python -m mudm.examples.neuroglancer_export --output-dir neuroglancer_output

    # 2. Serve it
    python -m mudm.examples.neuroglancer_serve neuroglancer_output

    # 3. Open the printed URL in your browser
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from mudm_tools.neuroglancer.state import (
    build_skeleton_layer,
    build_annotation_layer,
    build_viewer_state,
    viewer_state_to_url,
)


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler that adds CORS headers for Neuroglancer compatibility."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def detect_layers(directory: Path) -> list[dict]:
    """Auto-detect Neuroglancer layers by looking for info files."""
    layers = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        info_file = child / "info"
        if not info_file.exists():
            continue
        try:
            info = json.loads(info_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        at_type = info.get("@type", "")
        if at_type == "neuroglancer_skeletons":
            layers.append(("segmentation", child.name))
        elif at_type == "neuroglancer_annotations_v1":
            layers.append(("annotation", child.name))

    return layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve Neuroglancer precomputed data with CORS"
    )
    parser.add_argument(
        "directory",
        help="Directory containing precomputed data (output of neuroglancer_export.py)",
    )
    parser.add_argument(
        "--port", type=int, default=9000,
        help="Port to serve on (default: 9000)",
    )
    parser.add_argument(
        "--neuroglancer-url",
        default="https://neuroglancer-demo.appspot.com",
        help="Neuroglancer instance URL",
    )
    args = parser.parse_args()

    directory = Path(args.directory).resolve()
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory")
        return

    base_data_url = f"http://localhost:{args.port}"

    # Auto-detect layers and build viewer URLs
    print(f"\nServing: {directory}")
    print(f"URL:     {base_data_url}\n")

    # Check for subdirectories that might contain separate exports
    # (e.g. mixed_collection/, single_skeleton/)
    export_dirs = []
    if (directory / "info").exists():
        # The directory itself is a single precomputed source
        export_dirs = [directory]
    else:
        for child in sorted(directory.iterdir()):
            if child.is_dir() and any(child.rglob("info")):
                export_dirs.append(child)

    for export_dir in export_dirs:
        rel = export_dir.relative_to(directory)
        layer_infos = detect_layers(export_dir)

        if not layer_infos:
            continue

        layers = []
        for layer_type, name in layer_infos:
            source = f"precomputed://{base_data_url}/{rel}/{name}"
            if layer_type == "segmentation":
                layer = build_skeleton_layer(name, source)
                # Auto-detect segment IDs from binary files in the directory
                seg_dir = export_dir / name
                seg_ids = sorted(
                    f.name for f in seg_dir.iterdir()
                    if f.is_file() and f.name != "info" and not f.name.startswith(".")
                )
                if seg_ids:
                    layer["segments"] = seg_ids
                layers.append(layer)
            else:
                layers.append(build_annotation_layer(name, source))

        if layers:
            state = build_viewer_state(layers)
            url = viewer_state_to_url(state, args.neuroglancer_url)
            print(f"  {rel}/")
            for layer_type, name in layer_infos:
                print(f"    - {name} ({layer_type})")
            print(f"    Open: {url}")
            print()

    # Serve
    handler = functools.partial(CORSHTTPRequestHandler, directory=str(directory))
    server = HTTPServer(("", args.port), handler)
    print(f"Serving on http://localhost:{args.port} (Ctrl+C to stop)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
