#!/usr/bin/env python3

"""Convert MAVEN's straight format to tagged.

The input and output JSON files are list of objects with the following keys:
- input (str): The input context.
- output (str): The model predicted extraction.
- gold (str): The annotation extraction.
"""

import argparse
import json
from typing import TextIO, TypedDict

from beartype.door import is_bearable


class Item(TypedDict):
    input: str
    output: str
    gold: str


def main(input_file: TextIO, output_file: TextIO) -> None:
    data = json.load(input_file)

    if not is_bearable(data, list[dict[str, str]]):
        raise ValueError(
            "Invalid JSON format. Expected a list of objects. See --help for more"
            " information."
        )

    template = "[Cause] {cause} [Relation] cause [Effect]"
    new_data = [
        {
            "input": item["input"],
            "output": template.format(cause=item["output"]),
            "gold": template.format(cause=item["gold"]),
        }
        for item in data
    ]

    json.dump(new_data, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="Input JSON file"
    )
    parser.add_argument(
        "output_file", type=argparse.FileType("w"), help="Output JSON file"
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file)
