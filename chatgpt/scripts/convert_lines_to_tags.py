#!/usr/bin/env python3
"""Convert extraction in Lines format to Tags format.

The input file should be a JSON with a list of objects with the following keys:
- text (str): The input context.
- pred (str): The extracted relation in Lines format.
- answer (str): The gold relation in Lines format.
"""

import argparse
import json
from typing import TextIO


def find_part(part: str, lines: list[str]) -> str:
    for line in lines:
        if line.startswith(part):
            return line.split(":", 1)[1].strip()
    return ""


def convert_extract(extract: str) -> str:
    lines = extract.splitlines()
    cause = find_part("Cause", lines)
    effect = find_part("Effect", lines)
    relation = find_part("Relation", lines) or "cause"
    return f"[Cause] {cause} [Relation] {relation} [Effect] {effect}"


def main(input_file: TextIO, output_file: TextIO) -> None:
    data = json.load(input_file)
    converted = [
        {
            "input": d["text"],
            "output": convert_extract(d["pred"]),
            "gold": convert_extract(d["answer"]),
        }
        for d in data
    ]
    json.dump(converted, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="Input file (use - for stdin)"
    )
    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="Output file (use - for stdout)",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
