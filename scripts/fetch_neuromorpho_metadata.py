#!/usr/bin/env python3
"""Fetch NeuroMorpho.Org metadata for every SWC under data/cnn-nmo/.

For each .swc file, the basename (stripped of `.CNG`) is looked up by exact
match via ``/api/neuron/select?q=neuron_name:<name>``. The resolved neuron_id
is then used to pull morphometrics via ``/api/morphometry/id/<id>``.

Results are persisted as:
  data/cnn-nmo/metadata.json             map: filename -> {neuron, morphometry}
  data/cnn-nmo/metadata_unresolved.json  list of filenames with no match

The script is resumable: entries already present in metadata.json are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://neuromorpho.org/api"
HEADERS = {"Accept": "application/json", "User-Agent": "mudm-tools/0.1"}


def _neuron_name_from_path(swc_path: Path) -> str:
    stem = swc_path.stem
    return re.sub(r"\.CNG$", "", stem, flags=re.IGNORECASE)


def _http_get(path: str, *, timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(BASE + path, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def _get_json(path: str, *, retries: int = 3, delay: float = 0.5) -> dict | None:
    for attempt in range(retries):
        code, body = _http_get(path)
        if code == 200:
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return None
        if code in (429, 500, 502, 503, 504) and attempt + 1 < retries:
            time.sleep(delay * (attempt + 1) * 2)
            continue
        return None
    return None


def _select_exact(name: str) -> dict | None:
    q = urllib.parse.urlencode({"q": f"neuron_name:{name}", "size": "5"})
    obj = _get_json(f"/neuron/select?{q}")
    if obj is None:
        return None
    embedded = obj.get("_embedded", {}) or {}
    neurons = (
        embedded.get("neuronResources")
        or embedded.get("neuronResourceList")
        or []
    )
    for n in neurons:
        if n.get("neuron_name") == name:
            return n
    return None


def _morphometry(neuron_id: int) -> dict | None:
    return _get_json(f"/morphometry/id/{neuron_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/cnn-nmo",
        help="Directory containing .swc files (default: data/cnn-nmo)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N files (useful for testing)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Seconds between requests (default: 0.3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch even if already in metadata.json",
    )
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"ERROR: {data_dir} does not exist")
        return 1

    swc_files = sorted(data_dir.glob("*.swc"))
    if args.limit:
        swc_files = swc_files[: args.limit]
    print(f"Found {len(swc_files)} .swc files in {data_dir}")

    out_path = data_dir / "metadata.json"
    unresolved_path = data_dir / "metadata_unresolved.json"

    metadata: dict[str, dict] = {}
    if out_path.exists() and not args.force:
        try:
            metadata = json.loads(out_path.read_text())
            print(f"Resuming: {len(metadata)} entries already in {out_path.name}")
        except json.JSONDecodeError:
            print(f"WARN: {out_path} exists but is not JSON; starting fresh")

    unresolved: list[str] = []
    resolved = len(metadata)
    new_resolved = 0
    n = len(swc_files)

    for i, swc_path in enumerate(swc_files, 1):
        key = swc_path.name
        if key in metadata and not args.force:
            continue

        name = _neuron_name_from_path(swc_path)
        neuron = _select_exact(name)
        if neuron is None:
            unresolved.append(key)
            print(f"[{i:>3}/{n}] MISS  {key} (query name={name!r})")
            time.sleep(args.delay)
            continue

        time.sleep(args.delay)
        nid = neuron.get("neuron_id")
        morph = _morphometry(int(nid)) if isinstance(nid, int) else None

        metadata[key] = {"neuron": neuron, "morphometry": morph}
        resolved += 1
        new_resolved += 1
        archive = neuron.get("archive", "?")
        species = neuron.get("species", "?")
        has_morph = "yes" if morph else "no"
        print(
            f"[{i:>3}/{n}] OK    {key}  id={nid}  "
            f"archive={archive}  species={species}  morph={has_morph}"
        )
        time.sleep(args.delay)

        if new_resolved and new_resolved % 25 == 0:
            out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    out_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    unresolved_path.write_text(json.dumps(sorted(unresolved), indent=2))

    print()
    print(f"Total files     : {n}")
    print(f"Resolved        : {resolved}")
    print(f"New this run    : {new_resolved}")
    print(f"Unresolved      : {len(unresolved)}")
    print(f"Written         : {out_path}")
    print(f"Unresolved list : {unresolved_path}")

    if metadata:
        from collections import Counter

        archives: Counter[str] = Counter()
        species: Counter[str] = Counter()
        for entry in metadata.values():
            n_obj = entry.get("neuron") or {}
            archives[n_obj.get("archive", "?")] += 1
            species[n_obj.get("species", "?")] += 1
        print("\nBy archive:")
        for k, v in archives.most_common():
            print(f"  {v:>4}  {k}")
        print("\nBy species:")
        for k, v in species.most_common():
            print(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
