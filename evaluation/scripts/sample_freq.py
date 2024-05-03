#!/usr/bin/env python3
"""Sample data from dataset preserving frequencies of values for a given key."""
# pyright: basic
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def sample_preserving_frequencies(
    data: list[dict[str, Any]], k: int, key: str
) -> list[dict[str, Any]]:
    "Samples K elements preserving the frequency of values for the specified key."
    freq = Counter(item[key] for item in data)

    total = sum(freq.values())
    sampled_data = []

    for value, count in freq.items():
        value_to_sample = round(k * (count / total))
        value_data = [item for item in data if item[key] == value]
        sampled_value = random.sample(value_data, value_to_sample)
        sampled_data.extend(sampled_value)

    random.shuffle(sampled_data)
    return sampled_data


def label_dist(data: list[dict[str, Any]], key: str) -> dict[str, float]:
    "Returns the distribution of labels in the data."
    freq = Counter(item[key] for item in data)
    total = sum(freq.values())
    return {value: count / total for value, count in freq.items()}


def show_dist(data: list[dict[str, Any]], key: str, title: str) -> None:
    "Prints the distribution of labels in the data."
    dist = label_dist(data, key)

    print(f"{title.capitalize()} distribution")
    for key, ratio in dist.items():
        print(f"  {key}: {ratio:.2%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("k", type=int, help="Number of samples")
    parser.add_argument("key", type=str, help="Key to sample based on")
    parser.add_argument(
        "--output", type=Path, help="Output file. Default: <input>.<k>.json"
    )
    parser.add_argument(
        "--seed", type=int, help="Random seed. Default: %(default)s", default=0
    )
    args = parser.parse_args()

    random.seed(args.seed)
    data = json.loads(args.input.read_text())
    sampled = sample_preserving_frequencies(data, args.k, args.key)

    show_dist(data, args.key, "original")
    show_dist(sampled, args.key, "sampled")

    output = args.output or args.input.with_name(f"{args.input.stem}.{args.k}.json")
    output.write_text(json.dumps(sampled))


if __name__ == "__main__":
    main()
