#!/usr/bin/env python3

import argparse
import json
from typing import TextIO


def main(input_file: TextIO, output_file: TextIO, rename: list[str]) -> None:
    renames: dict[str, str] = {}
    for r in rename:
        if ":" not in r:
            renames[r] = r
        else:
            old, new = r.split(":")
            renames[old] = new

    data = json.load(input_file)
    new_data = [{new: d[old] for old, new in renames.items()} for d in data]

    json.dump(new_data, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="The input JSON file to rename keys in",
    )
    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="The output JSON file to write the renamed keys to",
    )
    parser.add_argument(
        "rename",
        type=str,
        nargs="+",
        help="A list of key:value pairs to rename, separated by a colon",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.rename)
