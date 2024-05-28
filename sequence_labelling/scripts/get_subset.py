#!/usr/bin/env python3
"""Get subset of N blocks from a sequence labeling dataset.

For the JSON datasets, it's trivial to do this using jq. For this one, we need a special
operation.

The input and output formats are blocks of text with one pair of token and label per
line. Each block represents a dataset item and are separated by a blank line (two
newlines).
"""

import argparse
from typing import TextIO


def main(input: TextIO, n: int, output: TextIO) -> None:
    content = input.read()
    blocks = content.split("\n\n")

    print(f"Before {len(blocks)=}")
    chosen_blocks = blocks[:n]
    print(f"After {len(chosen_blocks)=}")

    output.write("\n\n".join(chosen_blocks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=argparse.FileType("r"))
    parser.add_argument("n", type=int)
    parser.add_argument("output", type=argparse.FileType("w"))
    args = parser.parse_args()
    main(args.input, args.n, args.output)
