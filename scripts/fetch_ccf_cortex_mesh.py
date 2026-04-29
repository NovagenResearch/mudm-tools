#!/usr/bin/env python3
"""Download + cache the Allen CCF v3 isocortex mesh as OBJ (direct HTTP fetch).

Isocortex structure id is 315 in the Allen Mouse Brain ontology. The mesh
is fetched directly from the Allen Institute's public download endpoint -
allensdk isn't used (it pins matplotlib<3.4.3 which doesn't build on
Python 3.13).

Output:
    data/ccf/isocortex.obj    (OBJ mesh in CCF v3 um coordinates)

Usage:
    uv run python scripts/fetch_ccf_cortex_mesh.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CCF_DIR = ROOT / "data" / "ccf"
OUT_OBJ = CCF_DIR / "isocortex.obj"

ISOCORTEX_ID = 315
# Verified 2026-04-20: HTTP 301 -> HTTPS 200, ~4.3 MB VTK-written OBJ text,
# coordinates in CCF v3 um (range 0-13200 on AP axis).
MESH_URL = (
    "http://download.alleninstitute.org/informatics-archive/current-release/"
    "mouse_ccf/annotation/ccf_2017/structure_meshes/315.obj"
)


def main() -> int:
    CCF_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_OBJ.exists():
        print(f"Already cached: {OUT_OBJ} ({OUT_OBJ.stat().st_size} bytes)")
        return 0

    print(f"Downloading {MESH_URL} ...")
    req = urllib.request.Request(
        MESH_URL,
        headers={"User-Agent": "mudm-tools/0.1"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        if resp.status != 200:
            print(f"ERROR: HTTP {resp.status}")
            return 1
        data = resp.read()

    # Sanity check: OBJ text should start with '#', 'v ', 'o ', or 'g '
    head = data[:100].decode("utf-8", errors="replace").lstrip()
    if not any(head.startswith(p) for p in ("#", "v ", "o ", "g ", "mtllib")):
        print(f"ERROR: response doesn't look like OBJ text; first bytes: {head!r}")
        return 2

    OUT_OBJ.write_bytes(data)
    size = OUT_OBJ.stat().st_size
    print(f"Wrote {OUT_OBJ} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
