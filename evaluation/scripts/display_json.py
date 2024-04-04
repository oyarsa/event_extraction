#!/usr/bin/env python3
import argparse
import json
import textwrap
from pathlib import Path
from typing import Any


def indent(text: str, cols: int) -> str:
    return "\n".join(
        "\t" + ("\n\t» ".join(textwrap.wrap(line, width=cols)))
        for line in text.splitlines()
    )


def display(i: int, d: dict[str, Any], cols: int) -> str:
    lines = [f"# {i}"]

    for k in sorted(d.keys()):
        lines.append(f"{k.upper()}:")
        lines.extend(indent(str(d[k]), cols).splitlines())
        lines.append("")

    lines.append("-" * 80)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=Path, help="The JSON file to show. Must be an array of objects."
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=80,
        help="The number of columns to wrap the text to.",
    )
    args = parser.parse_args()

    data: list[dict[str, Any]] = json.loads(args.file.read_text())
    if not (isinstance(data, list) and all(isinstance(d, dict) for d in data)):
        raise ValueError("File must be an array of objects.")

    for i, d in enumerate(data):
        print(display(i, d, args.columns))


if __name__ == "__main__":
    main()
