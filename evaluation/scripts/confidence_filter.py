import argparse
import json
from typing import Any, TextIO


def average_filter(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    average = sum(d["confidence"] for d in data) / len(data)
    return [d for d in data if d["confidence"] >= average]


FILTERS = {
    "none": lambda x: x,
    "average": average_filter,
}


def main(input_file: TextIO, output_file: TextIO, filter: str) -> None:
    data = json.load(input_file)
    filtered_data = FILTERS[filter](data)
    json.dump(filtered_data, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="Classifier inference output to filter",
    )
    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="Output file to write filtered results to",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=FILTERS,
        default="average",
        help="Method to filter outputs by confidence level. Default: %(default)s.",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.filter)
