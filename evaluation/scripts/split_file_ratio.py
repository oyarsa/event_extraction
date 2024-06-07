#!/usr/bin/env python3

"""Split a data file by a ratio, keeping the frequency of tags the same in both parts.

The input file should be a JSON with a list of objects with the following keys:
- input (str): the input context
- output (str): the model output (tagged)
- gold (str): the gold output (tagged)
- tag (str): the tag of the example, used to sample and kept approximately the same
- valid (bool): True if the example is valid else False

The output files have the same format as the input file.
"""

import argparse
import json
import random
from collections import Counter
from typing import TextIO


def get_key(d: dict[str, str]) -> str:
    keys = ["input", "output", "gold"]
    return "".join([d[k] for k in keys])


def sample_preserving_frequencies(
    data: list[dict[str, str]], k: int, key: str
) -> list[dict[str, str]]:
    "Samples K elements preserving the frequency of values for the specified key."
    freq = Counter(item[key] for item in data)

    total = sum(freq.values())
    sampled_data: list[dict[str, str]] = []

    for value, count in freq.items():
        value_to_sample = round(k * (count / total))
        value_data = [item for item in data if item[key] == value]
        sampled_value = random.sample(value_data, value_to_sample)
        sampled_data.extend(sampled_value)

    random.shuffle(sampled_data)
    return sampled_data


def main(
    input_file: TextIO, ratio: float, first_file: TextIO, second_file: TextIO, seed: int
) -> None:
    random.seed(seed)

    data: list[dict[str, str]] = json.load(input_file)

    if not (0 <= ratio <= 1):
        raise SystemExit(f"Invalid ratio: {ratio}. Must be in [0, 1].")

    k = int(len(data) * ratio)
    first = sample_preserving_frequencies(data, k, "tag")

    sample_keys = {get_key(d) for d in first}
    second = [d for d in data if get_key(d) not in sample_keys]

    print("Length of data:", len(data))
    print("Length of first part:", len(first))
    print("Length of second part:", len(second))

    json.dump(first, first_file, indent=2)
    json.dump(second, second_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", type=argparse.FileType("r"), help="Input file")
    parser.add_argument("ratio", type=float, help="Ratio of to split the file")
    parser.add_argument(
        "first_file", type=argparse.FileType("w"), help="Output file for the first part"
    )
    parser.add_argument(
        "second_file",
        type=argparse.FileType("w"),
        help="Output file for the second part",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the random number generator"
    )
    args = parser.parse_args()
    main(args.input_file, args.ratio, args.first_file, args.second_file, args.seed)
