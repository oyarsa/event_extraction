#!/usr/bin/env python3

"""Split a data file by a list ratios, keeping the frequency of tags the same.

The ratios are used in order, must be in [0, 1] and sum to <= 1. If the sum is < 1, the
remaining ratio will be appended. There will be one file per ratio, each with the
interval in the name.

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
from collections import defaultdict
from pathlib import Path


def get_key(d: dict[str, str]) -> str:
    keys = ["input", "output", "gold"]
    return "".join([d[k] for k in keys])


def sample_groups_by_ratios[T](
    tag_to_groups: dict[str, list[T]], ratios: list[float]
) -> list[tuple[float, float, list[T]]]:
    """Sample each group by the given ratios, keeping the frequency of tags the same.

    Args:
        tag_to_groups: mapping from tags to lists of items.
        ratios: list of ratios to split the groups, e.g. [0.8, 0.2] to split 80/20,
            [0.5, 0.3, 0.2] to split 50/30/20, etc. Must sum to 1.

    Returns:
        The combination of the the sampled groups, one for each ratio, with the
        corresponding start and end ratio.

    Note: some results might be unintuitive, e.g. if `ratios` is [0.5, 0.5], and there
    are groups with uneven elements, the result will not be a proper 50/50 split.
    """

    result: list[list[T]] = [[] for _ in ratios]
    ratio_ranges: list[tuple[float, float]] = []

    for group in tag_to_groups.values():
        random.shuffle(group)

    start = 0
    for ratio in ratios:
        end = start + ratio
        ratio_ranges.append((start, end))
        start = end

    for group in tag_to_groups.values():
        total_groups = len(group)
        start_index = 0

        for i, (_, end_ratio) in enumerate(ratio_ranges):
            end_index = int(total_groups * end_ratio)
            result[i].extend(group[start_index:end_index])
            start_index = end_index

    return [
        (start_ratio, end_ratio, parts)
        for (start_ratio, end_ratio), parts in zip(ratio_ranges, result)
    ]


def main(
    input_file: Path, output_dir: Path, ratios: list[float], output_name: str, seed: int
) -> None:
    random.seed(seed)
    data: list[dict[str, str]] = json.loads(input_file.read_text())

    if not all(0 <= ratio <= 1 for ratio in ratios):
        raise SystemExit(f"Invalid ratios: {ratios}. All ratios must be in [0, 1].")
    if sum(ratios) > 1:
        raise SystemExit(
            "Invalid ratios. The sum of ratios must be less than or equal to 1."
        )
    if sum(ratios) < 1:
        ratios.append(1 - sum(ratios))

    output_dir.mkdir(parents=True, exist_ok=True)

    tag_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in data:
        tag_groups[item["tag"]].append(item)

    sampled_splits = sample_groups_by_ratios(tag_groups, ratios)

    keys = [get_key(item) for _, _, split in sampled_splits for item in split]
    assert len(keys) == len(data), "Some items are missing"
    assert len(set(keys)) == len(data), "Some keys are duplicated"

    for start_ratio, end_ratio, split in sampled_splits:
        name = f"{output_name}_{start_ratio:.1f}-{end_ratio:.1f}.json"
        (output_dir / name).write_text(json.dumps(split, indent=2))

    print("Length of data:", len(data))
    for start_ratio, end_ratio, split in sampled_splits:
        print(f"- {start_ratio:.1f}-{end_ratio:.1f}: {len(split)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", type=Path, help="Path to the file")
    parser.add_argument("output_dir", type=Path, help="Path to the output dir")
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        help="Ratios to split the file. Must be in [0, 1] and sum to <= 1. If the sum"
        " is < 1, it will be topped up.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="split",
        help="Name of the output files, to be appended with the ratios. (default:"
        " %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the random number generator (default: %(default)s)",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_dir, args.ratios, args.output_name, args.seed)
