"""Remove entries from the "left" file that are in the "right" file."""

import argparse
import hashlib
import json
from typing import TextIO


def hash_data(data: dict[str, str]) -> str:
    """Hash data (list of strings) using sha256 with the first 8 characters."""
    keys = ["text", "reference", "model"]
    return hashlib.sha256("".join(data[k] for k in keys).encode()).hexdigest()[:8]


def main(left: TextIO, right: TextIO, output: TextIO) -> None:
    """Remove annotated files from the dataset."""
    left_data = json.load(left)
    right_data = json.load(right)

    right_keys = {hash_data(entry) for entry in right_data}
    left_keys = {hash_data(entry) for entry in left_data}

    if len(left_keys) != len(left_data):
        print(
            f"Warning: Duplicate keys found in the left file. Unique: {len(left_keys)}"
        )

    left_remain = [entry for entry in left_data if hash_data(entry) not in right_keys]

    print(
        "Left:", len(left_data), "Right:", len(right_data), "Remain:", len(left_remain)
    )
    json.dump(left_remain, output, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "left", type=argparse.FileType("r"), help="File to be removed from."
    )
    parser.add_argument(
        "right", type=argparse.FileType("r"), help="File to remove from the left file."
    )
    parser.add_argument("output", type=argparse.FileType("w"), help="Output file.")
    args = parser.parse_args()
    main(args.left, args.right, args.output)
