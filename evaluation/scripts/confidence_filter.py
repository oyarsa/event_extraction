import argparse
import copy
import json
import random
from functools import partial
from typing import Any, TextIO


def shuffled[T](data: list[T]) -> list[T]:
    new_data = copy.deepcopy(data)
    random.shuffle(new_data)
    return new_data


def top_p_confidence(data: list[dict[str, Any]], p: float) -> list[dict[str, Any]]:
    n = int(len(data) * p)
    return sorted(data, key=lambda d: d["confidence"], reverse=True)[:n]


def average_filter(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the data points with confidence above the average."""
    average = sum(d["confidence"] for d in data) / len(data)
    return shuffled([d for d in data if d["confidence"] >= average])


def top_x_filter(data: list[dict[str, Any]], x: float) -> list[dict[str, Any]]:
    """Keep only the top x% of data points by confidence level."""
    return shuffled(top_p_confidence(data, x))


def alt_top_x_filter(data: list[dict[str, Any]], x: float) -> list[dict[str, Any]]:
    """Keep the top x% of valid data points and the top x% of invalid data points."""
    valid = [d for d in data if d["valid"]]
    invalid = [d for d in data if not d["valid"]]

    return shuffled(top_p_confidence(valid, x) + top_p_confidence(invalid, x))


FILTERS = {
    "none": lambda x: x,
    "average": average_filter,
    "top_0.5": partial(top_x_filter, x=0.5),
    "top_0.75": partial(top_x_filter, x=0.75),
    "alt_top_0.5": partial(alt_top_x_filter, x=0.5),
    "alt_top_0.75": partial(alt_top_x_filter, x=0.75),
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
