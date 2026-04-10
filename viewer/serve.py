#!/usr/bin/env python3
"""Development HTTP server for the Three.js 3D Tiles viewer.

Serves GLB files with Brotli (or gzip) Content-Encoding for transparent
decompression in the browser.  Meshopt-encoded GLBs compress ~2-3x further
with Brotli, approaching Draco file sizes while keeping fast decode.

Routes:
    /                                  → viewer/index.html
    /js/*                              → viewer/js/*
    /tiles/pyramids.json               → manifest file
    /tiles/{pyramid_id}/*              → {tiles_base}/{pyramid_id}/3dtiles/*
    /neuroglancer/{pyramid_id}/info    → Neuroglancer mesh info
    /neuroglancer/{pyramid_id}/<id>    → Neuroglancer binary mesh
    /neuroglancer/{pyramid_id}/<id>:0  → Neuroglancer fragment manifest
    /neuroglancer/{pyramid_id}/segment_properties/info → segment properties
"""

import argparse
import io
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

try:
    import brotli
    _HAS_BROTLI = True
except ImportError:
    _HAS_BROTLI = False

import gzip as _gzip

VIEWER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIEWER_DIR.parent
VIEWER_2D_DIR = PROJECT_ROOT / "viewer2d"

# Pre-compressed file cache: absolute_path → (compressed_bytes, encoding)
_COMP_CACHE: dict[str, tuple[bytes, str]] = {}
_CACHE_MAX_BYTES = 512 * 1024 * 1024  # 512 MB cache limit
_cache_total = 0


def _compress(data: bytes) -> tuple[bytes, str]:
    """Compress with Brotli (preferred) or gzip fallback."""
    if _HAS_BROTLI:
        return brotli.compress(data, quality=5), "br"
    return _gzip.compress(data, compresslevel=6), "gzip"


class ViewerHandler(SimpleHTTPRequestHandler):
    """Route requests to viewer assets or tile data."""

    tiles_base: str = ""
    tiles2d_base: str = ""

    def translate_path(self, path: str) -> str:
        # Strip query string
        path = path.split("?", 1)[0].split("#", 1)[0]

        if path == "/tiles/pyramids.json":
            return os.path.join(self.tiles_base, "pyramids.json")

        if path.startswith("/neuroglancer/"):
            # /neuroglancer/{pyramid_id}/rest → {tiles_base}/{pyramid_id}/neuroglancer/rest
            rel = path[len("/neuroglancer/"):]
            parts = rel.split("/", 1)
            pyramid_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            return os.path.join(self.tiles_base, pyramid_id, "neuroglancer", rest)

        if path.startswith("/tiles/"):
            # /tiles/{pyramid_id}/rest
            rel = path[len("/tiles/"):]
            parts = rel.split("/", 1)
            pyramid_id = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            # Check pyramid root first (features.json, tilejson3d.json),
            # then fall back to 3dtiles/ subdirectory for tile data
            root_path = os.path.join(self.tiles_base, pyramid_id, rest)
            if os.path.isfile(root_path):
                return root_path
            return os.path.join(self.tiles_base, pyramid_id, "3dtiles", rest)

        # --- 2D viewer routes ---
        if path == "/2d" or path == "/2d/":
            return str(VIEWER_2D_DIR / "index.html")

        if path.startswith("/2d/"):
            rel = path[4:]  # strip "/2d/"
            asset = VIEWER_2D_DIR / rel
            if asset.exists():
                return str(asset)

        # 2D tile data
        if path == "/tiles2d/datasets.json":
            self._serve_2d_datasets_json()
            return ""  # response already sent

        if path.startswith("/tiles2d/") and self.tiles2d_base:
            rel = path[len("/tiles2d/"):]
            tile_path = Path(self.tiles2d_base) / rel
            if tile_path.exists():
                return str(tile_path)

        if path == "/":
            return str(VIEWER_DIR / "index.html")

        # Serve viewer assets (js/, etc.)
        rel = path.lstrip("/")
        return str(VIEWER_DIR / rel)

    def do_GET(self):
        file_path = self.translate_path(self.path)

        if not file_path:
            return  # Response already sent (e.g., datasets.json)

        # Compress .glb files on the fly if browser accepts it
        if self.path.endswith(".glb") and os.path.isfile(file_path):
            accept = self.headers.get("Accept-Encoding", "")
            can_br = _HAS_BROTLI and "br" in accept
            can_gz = "gzip" in accept

            if can_br or can_gz:
                self._serve_compressed(file_path, accept)
                return

        # Fall through to default handler
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

            # Cache if under limit
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
        """Scan tiles2d base dir and return list of available 2D datasets."""
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
        # CORS for local development
        self.send_header("Access-Control-Allow-Origin", "*")
        # Cache .glb files aggressively (they don't change)
        if self.path.endswith(".glb"):
            self.send_header("Cache-Control", "public, max-age=86400")
        elif self.path.endswith((".js", ".html", ".json")):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description="Serve 3D Tiles viewer")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--tiles-base",
        default=str(PROJECT_ROOT / "data" / "mouselight" / "tiles"),
        help="Base directory containing pyramid subdirectories (default: data/mouselight/tiles/)",
    )
    parser.add_argument(
        "--tiles2d-base",
        default="",
        help="Base directory containing 2D tile datasets",
    )
    args = parser.parse_args()

    tiles_base = os.path.abspath(args.tiles_base)
    manifest = os.path.join(tiles_base, "pyramids.json")
    if not os.path.isfile(manifest):
        print(f"WARNING: No pyramids.json found in {tiles_base}")

    ViewerHandler.tiles_base = tiles_base
    ViewerHandler.tiles2d_base = os.path.abspath(args.tiles2d_base) if args.tiles2d_base else ""

    if _HAS_BROTLI:
        print("Compression: Brotli (GLB files served with Content-Encoding: br)")
    else:
        print("Compression: gzip (install 'brotli' for better compression: uv add brotli)")

    server = HTTPServer(("", args.port), ViewerHandler)
    print(f"Viewer:  http://localhost:{args.port}")
    print(f"Tiles:   {tiles_base}")
    if ViewerHandler.tiles2d_base:
        print(f"2D Tiles: {ViewerHandler.tiles2d_base}")
        print(f"2D Viewer: http://localhost:{args.port}/2d/")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
