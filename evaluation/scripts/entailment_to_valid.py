#!/usr/bin/env python3

"""Convert the entailment label to valid.

Takes the 'reward_label' key and convert from 'ENTAILMENT' to 'VALID', and the rest to
'INVALID'.

Both input and output are JSON files with a list of objects. The only required key is
'reward_label'; the rest of the keys are kept as is.
"""

import argparse
import json
from typing import TextIO


def main(input_file: TextIO, output_file: TextIO) -> None:
    data = json.load(input_file)
    new_data = [
        d
        | {"reward_label": "VALID" if d["reward_label"] == "ENTAILMENT" else "INVALID"}
        for d in data
    ]
    json.dump(new_data, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="File with entailment labels"
    )
    parser.add_argument(
        "output_file", type=argparse.FileType("w"), help="File with valid labels"
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
