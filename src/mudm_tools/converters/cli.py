"""Unified CLI for muDM format conversion.

Usage:
    mudm convert --format xenium --input data/outs --output tiles/sample
    mudm convert --format obj --input data/meshes --output tiles/brain
    mudm convert --format geojson --input data/cells.geojson --output tiles/cells
    mudm list-formats
"""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        prog="mudm",
        description="muDM format converter — transform source data into tiled MVT + Parquet",
    )
    subparsers = parser.add_subparsers(dest="command")

    # convert command
    convert_parser = subparsers.add_parser(
        "convert", help="Convert source data to muDM tiled format"
    )
    convert_parser.add_argument(
        "--format",
        "-f",
        required=True,
        help="Source format (xenium, obj, geojson)",
    )
    convert_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to source data directory or file",
    )
    convert_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path for tiled output",
    )
    convert_parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to JSON config file with converter-specific settings",
    )
    convert_parser.add_argument(
        "--temp-dir",
        default=None,
        help="Temp directory for intermediate files",
    )
    convert_parser.add_argument(
        "--max-zoom",
        type=int,
        default=None,
        help="Override max zoom level",
    )

    # list-formats command
    subparsers.add_parser("list-formats", help="List available converter formats")

    args = parser.parse_args()

    if args.command == "list-formats":
        from mudm_tools.converters import list_formats

        for fmt in list_formats():
            print(f"  {fmt}")
        return

    if args.command == "convert":
        from mudm_tools.converters import convert

        # Build config
        config = {}
        if args.config:
            with open(args.config) as f:
                config = json.load(f)
        if args.temp_dir:
            config["temp_dir"] = args.temp_dir
        if args.max_zoom is not None:
            config["max_zoom"] = args.max_zoom

        result = convert(args.format, args.input, args.output, config)
        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
