#!/usr/bin/env python3
"""HTTP server for muDM tile viewers.

Serves 3D Tiles (GLB) or 2D vector tiles (MVT/PBF) with on-the-fly
Brotli/gzip compression. Bundles both the Three.js 3D viewer and the
Leaflet 2D viewer as package data.

Usage:
    mudm-serve --tiles-base output/                    # 3D viewer (default)
    mudm-serve --tiles-base output/ --viewer 2d        # 2D viewer
    mudm-serve --tiles-base output/ --viewer-dir ./my-viewer  # custom viewer
"""

import argparse
import gzip as _gzip
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

try:
    import brotli
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False

_VIEWERS_DIR = Path(__file__).resolve().parent / "viewers"

# Compressed file cache
_COMP_CACHE: dict[tuple, tuple[bytes, str]] = {}
_CACHE_MAX_BYTES = 512 * 1024 * 1024
_cache_total = 0


def get_viewer_dir(name: str) -> Path:
    """Get the path to a bundled viewer by name ('3d' or '2d')."""
    mapping = {
        "3d": _VIEWERS_DIR / "viewer3d",
        "2d": _VIEWERS_DIR / "viewer2d",
    }
    d = mapping.get(name)
    if d is None or not d.exists():
        available = [k for k, v in mapping.items() if v.exists()]
        raise ValueError(f"Unknown viewer '{name}'. Available: {available}")
    return d


class TileHandler(SimpleHTTPRequestHandler):
    tiles_base: str = ""
    tiles2d_base: str = ""
    viewer_dir: str = ""
    viewer_2d_dir: str = ""

    def translate_path(self, path: str) -> str:
        path = path.split("?", 1)[0].split("#", 1)[0]

        # Tile manifest
        if path == "/tiles/pyramids.json":
            return os.path.join(self.tiles_base, "pyramids.json")

        # Neuroglancer routes
        if path.startswith("/neuroglancer/"):
            rel = path[len("/neuroglancer/"):]
            parts = rel.split("/", 1)
            pyramid_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            return os.path.join(self.tiles_base, pyramid_id, "neuroglancer", rest)

        # 3D tile data
        if path.startswith("/tiles/"):
            rel = path[len("/tiles/"):]
            parts = rel.split("/", 1)
            pyramid_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            root_path = os.path.join(self.tiles_base, pyramid_id, rest)
            if os.path.isfile(root_path):
                return root_path
            return os.path.join(self.tiles_base, pyramid_id, "3dtiles", rest)

        # 2D viewer routes
        if path == "/2d" or path == "/2d/":
            if self.viewer_2d_dir:
                return os.path.join(self.viewer_2d_dir, "index.html")

        if path.startswith("/2d/"):
            if self.viewer_2d_dir:
                rel = path[4:]
                asset = os.path.join(self.viewer_2d_dir, rel)
                if os.path.isfile(asset):
                    return asset

        # 2D tile data
        if path == "/tiles2d/datasets.json":
            self._serve_2d_datasets_json()
            return ""

        if path.startswith("/tiles2d/") and self.tiles2d_base:
            rel = path[len("/tiles2d/"):]
            tile_path = os.path.join(self.tiles2d_base, rel)
            if os.path.isfile(tile_path):
                return tile_path

        # Root → primary viewer
        if path == "/":
            return os.path.join(self.viewer_dir, "index.html")

        # Viewer assets
        rel = path.lstrip("/")
        return os.path.join(self.viewer_dir, rel)

    def do_GET(self):
        file_path = self.translate_path(self.path)
        if not file_path:
            return

        if self.path.endswith(".glb") and os.path.isfile(file_path):
            accept = self.headers.get("Accept-Encoding", "")
            if (_HAS_BROTLI and "br" in accept) or "gzip" in accept:
                self._serve_compressed(file_path, accept)
                return

        super().do_GET()

    def _serve_compressed(self, file_path: str, accept: str):
        global _cache_total
        can_br = _HAS_BROTLI and "br" in accept
        preferred = "br" if can_br else "gzip"
        cache_key = (file_path, preferred)

        if cache_key in _COMP_CACHE:
            comp_data, encoding = _COMP_CACHE[cache_key]
        else:
            raw = open(file_path, "rb").read()
            if preferred == "br":
                comp_data, encoding = brotli.compress(raw, quality=5), "br"
            else:
                comp_data, encoding = _gzip.compress(raw, compresslevel=6), "gzip"
            if _cache_total + len(comp_data) < _CACHE_MAX_BYTES:
                _COMP_CACHE[cache_key] = (comp_data, encoding)
                _cache_total += len(comp_data)

        self.send_response(200)
        self.send_header("Content-Type", "model/gltf-binary")
        self.send_header("Content-Length", str(len(comp_data)))
        self.send_header("Content-Encoding", encoding)
        self.send_header("Vary", "Accept-Encoding")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(comp_data)

    def _serve_2d_datasets_json(self):
        import json as _json
        datasets = []
        base = Path(self.tiles2d_base) if self.tiles2d_base else None
        if base and base.exists():
            for d in sorted(base.iterdir()):
                meta_path = d / "metadata.json"
                if meta_path.exists():
                    meta = _json.loads(meta_path.read_text())
                    datasets.append({"id": d.name, "name": meta.get("name", d.name)})

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(_json.dumps(datasets).encode())

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        if self.path.endswith(".glb"):
            self.send_header("Cache-Control", "public, max-age=86400")
        elif self.path.endswith((".js", ".html", ".json", ".css")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Serve muDM tile viewer")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--tiles-base", type=str, required=True, help="Directory containing pyramids.json")
    parser.add_argument("--tiles2d-base", type=str, default="", help="Directory containing 2D tile datasets")
    parser.add_argument("--viewer", type=str, default="3d", choices=["3d", "2d"],
                        help="Bundled viewer to use (default: 3d)")
    parser.add_argument("--viewer-dir", type=str, default=None,
                        help="Custom viewer directory (overrides --viewer)")
    args = parser.parse_args()

    tiles_base = os.path.abspath(args.tiles_base)

    if args.viewer_dir:
        viewer_dir = os.path.abspath(args.viewer_dir)
    else:
        viewer_dir = str(get_viewer_dir(args.viewer))

    viewer_2d_dir = ""
    if args.viewer == "3d":
        # Also make 2D viewer available at /2d/ if it exists
        try:
            viewer_2d_dir = str(get_viewer_dir("2d"))
        except ValueError:
            pass

    manifest = os.path.join(tiles_base, "pyramids.json")
    if not os.path.isfile(manifest):
        print(f"WARNING: No pyramids.json in {tiles_base}")

    if not os.path.isfile(os.path.join(viewer_dir, "index.html")):
        print(f"ERROR: No index.html in {viewer_dir}")
        return 1

    TileHandler.tiles_base = tiles_base
    TileHandler.tiles2d_base = os.path.abspath(args.tiles2d_base) if args.tiles2d_base else ""
    TileHandler.viewer_dir = viewer_dir
    TileHandler.viewer_2d_dir = viewer_2d_dir

    comp = "Brotli" if _HAS_BROTLI else "gzip (install 'brotli' for better compression)"
    print(f"Compression: {comp}")
    print(f"Viewer:      http://localhost:{args.port} ({args.viewer})")
    if viewer_2d_dir:
        print(f"2D Viewer:   http://localhost:{args.port}/2d/")
    print(f"Tiles:       {tiles_base}")
    if TileHandler.tiles2d_base:
        print(f"2D Tiles:    {TileHandler.tiles2d_base}")
    print("Press Ctrl+C to stop")

    server = HTTPServer(("", args.port), TileHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
