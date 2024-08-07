"""Cleans data by removing unnecessary keys and renaming them to be more descriptive."""

import argparse
import json
from pathlib import Path

from evaluation.annotation.split_training import load_and_reshape_data


def main(input: Path, output: Path) -> None:
    data = load_and_reshape_data(input)
    output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input JSON file")
    parser.add_argument("output", type=Path, help="Output JSON file")
    args = parser.parse_args()
    main(args.input, args.output)
