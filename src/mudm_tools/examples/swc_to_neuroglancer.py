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

from mudm_tools.neuron_meta import NeuronRecord, load_sidecar
from mudm_tools.swc import _neuron_name_from_path, _parse_swc, swc_to_microjson
from mudm_tools.neuroglancer import write_skeleton
from mudm_tools.neuroglancer.properties_writer import write_segment_properties
from mudm_tools.neuroglancer.skeleton_writer import build_skeleton_info
from mudm_tools.neuroglancer.state import (
    build_skeleton_layer,
    build_viewer_state,
    viewer_state_to_url,
)


def _build_segment_properties_from_sidecar(
    sidecar_path: Path,
    swc_paths: list[Path],
    segment_ids: list[int],
    output_dir: Path,
) -> None:
    """Materialize 12-field Neuroglancer segment_properties from neurons_meta.parquet.

    Joins each SWC to a NeuronRecord by neuron_name (SWC stem with ``.CNG``
    stripped). Writes ``output_dir/info`` with the neuroglancer_segment_properties
    inline JSON schema.

    Missing neurons get empty/zero values (not skipped) so the ids list
    stays aligned with segment_ids.
    """
    from collections import OrderedDict

    sidecar_by_id = load_sidecar(sidecar_path)
    sidecar_by_name: dict[str, NeuronRecord] = {
        rec.neuron_name: rec for rec in sidecar_by_id.values()
    }

    label_keys = [
        "neuron_name", "source", "archive", "species",
        "brain_region_flat", "cell_type_flat", "reference_doi",
    ]
    number_keys = ["surface_m", "volume_m", "length", "n_bifs", "n_branch"]

    # Gather values in aligned order
    label_values: dict[str, list[str]] = {k: [] for k in label_keys}
    number_values: dict[str, list[float]] = {k: [] for k in number_keys}

    for swc in swc_paths:
        name = _neuron_name_from_path(str(swc))
        rec = sidecar_by_name.get(name)
        for k in label_keys:
            if rec is None:
                label_values[k].append("")
                continue
            v = getattr(rec, k, None)
            if isinstance(v, list):
                v = "/".join(str(x) for x in v)
            label_values[k].append("" if v is None else str(v))
        for k in number_keys:
            if rec is None:
                number_values[k].append(0.0)
                continue
            v = getattr(rec, k, None)
            number_values[k].append(float(v) if v is not None else 0.0)

    # Build the neuroglancer_segment_properties JSON
    properties = []
    for k in label_keys:
        properties.append({
            "id": k,
            "type": "label",
            "values": label_values[k],
        })
    for k in number_keys:
        properties.append({
            "id": k,
            "type": "number",
            "data_type": "float32",
            "values": number_values[k],
        })

    info = {
        "@type": "neuroglancer_segment_properties",
        "inline": OrderedDict([
            ("ids", [str(s) for s in segment_ids]),
            ("properties", properties),
        ]),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "info").write_text(json.dumps(info, indent=2))


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
    parser.add_argument(
        "--sidecar", type=Path, default=None,
        help="Path to neurons_meta.parquet; when provided, segment_properties "
             "are built from it (12-field subset) instead of feature.properties.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    skel_dir = output_dir / "skeletons"

    # Convert each SWC file
    features = []
    segment_ids = []
    valid_swc_paths: list[Path] = []
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
        valid_swc_paths.append(swc_path)

    if not features:
        print("\nNo valid SWC files found.")
        return

    # Write segment properties
    if args.sidecar and args.sidecar.exists():
        _build_segment_properties_from_sidecar(
            args.sidecar,
            valid_swc_paths,
            segment_ids,
            skel_dir / "seg_props",
        )
    else:
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
