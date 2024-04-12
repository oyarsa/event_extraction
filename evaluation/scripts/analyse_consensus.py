#!/usr/bin/env python3
"""Calculate the consensus of lists of integers in a JSON file.

Given a JSON file with files containing lists of integers, calculate the consensus of
these lists. Supports both ordinal and binary consensus and can plot the cumulative
distribution of the consensus values.
"""
import argparse
import json
import math
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt


def unique_values(iterables: Iterable[Iterable[Any]]) -> list[Any]:
    """Build a list of unique values from iterables maintaining the order."""
    seen = set()
    return [
        seen.add(item) or item
        for iterable in iterables
        for item in iterable
        if item not in seen
    ]


def records_to_table(items: list[dict[str, Any]]) -> tuple[list[str], list[list[Any]]]:
    """Given a list of dictionaries, convert to table where each row is a record.

    The keys of the dictionaries are used as the header of the table.

    Returns (header, rows).
    """
    header = unique_values(items)
    rows = [[item[key] for key in header] for item in items]
    return header, rows


def transpose_table(
    header: list[str], rows: list[list[Any]]
) -> tuple[list[str], list[list[Any]]]:
    """Transpose a table where each row is a record.

    The header is the first column of the new table.

    Returns (header, rows).
    """
    table = [[header[i]] + [row[i] for row in rows] for i in range(len(header))]
    return table[0], table[1:]


def render_table(header: list[str], values: list[list[Any]]) -> str:
    # Calculate the maximum length for each column, considering both headers and the
    # data in values
    max_lengths = [
        max(len(str(row[i])) if i < len(row) else 0 for row in [header, *values])
        for i in range(len(header))
    ]

    fmt_parts = [f"{{{i}:<{len}}}" for i, len in enumerate(max_lengths)]
    fmt_string = " | ".join(fmt_parts)

    def format_row(row: list[Any]) -> str:
        return fmt_string.format(*(map(str, row)))

    header_line = format_row(header)
    separator_line = " | ".join("-" * length for length in max_lengths)
    rows = [format_row(row) for row in values]

    return "\n".join([header_line, separator_line, *rows])


def ord_consensus(data: list[int], *, labels: list[int]) -> float:
    """From "Consensus and dissention: A measure of ordinal dispersion (2007)"
    By William J. Tastle, Mark J. Wierman.
    """
    if len(data) < 2:
        return 1

    # Make sure the labels start at 0 for bincount
    data = [x - min(labels) for x in data]
    p = np.bincount(data, minlength=len(labels)) / len(data)

    d_x = max(labels) - min(labels)
    u_x = (p * labels).sum()

    # Page 8
    return 1 + sum(
        p[i] * math.log2(1 - (abs(labels[i] - u_x) / d_x))
        for i in range(len(labels))
        if p[i] > 0
    )


class BinConsensus(str, Enum):
    ENTROPY = "entropy"
    GINI = "gini"
    MAJORITY = "majority"


def bin_consensus(data: list[int], kind: BinConsensus) -> float:
    match kind:
        case BinConsensus.ENTROPY:
            return entropy(data)
        case BinConsensus.GINI:
            return gini_coefficient(data)
        case BinConsensus.MAJORITY:
            return majority_pct(data)


def get_majority(data: list[int]) -> int:
    return max(set(data), key=data.count)


def majority_pct(data: list[int]) -> float:
    return data.count(get_majority(data)) / len(data)


def calc_probabilities(binary_vector: list[int]) -> npt.NDArray[np.float64]:
    _, counts = np.unique(binary_vector, return_counts=True)
    return counts / len(binary_vector)


def entropy(binary_vector: list[int]) -> float:
    """Calculate the entropy of a binary vector containing 0s and 1s.

    Entropy is calculated using the formula: -sum(p(x) * log2(p(x))) where p(x) is the
    proportion of each unique value in the vector.
    """
    probabilities = calc_probabilities(binary_vector)
    entropy_value = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy_value)


def gini_coefficient(binary_vector: list[int]) -> float:
    """Calculate the Gini coefficient of a binary vector containing 0s and 1s.

    Gini coefficient is calculated using the formula: 1 - sum(p(x)^2) where p(x) is the
    proportion of each unique value in the vector.
    """
    probabilities = calc_probabilities(binary_vector)
    gini_value = 1 - np.sum(probabilities**2)
    return float(gini_value)


def describe(numbers: list[float]) -> dict[str, float]:
    """Descriptive statistics for a list of numbers."""
    nums = np.array(numbers)
    r = {
        "mean": np.mean(nums),
        "std_dev": np.std(nums),
        "variance": np.var(nums),
        "range": np.max(nums) - np.min(nums),
        "iqr": np.percentile(nums, 75) - np.percentile(nums, 25),
        "min": np.min(nums),
        "q1": np.percentile(nums, 25),
        "median": np.median(nums),
        "q3": np.percentile(nums, 75),
        "max": np.max(nums),
    }
    return {k: 0.0 if v == -0.0 else float(v) for k, v in r.items()}


def report(description: dict[str, float], title: str) -> str:
    padding = max(len(key) for key in description) + 1
    out = [
        f">>> {title}",
        *(f"  {key:<{padding}}: {value:.3f}" for key, value in description.items()),
    ]
    return "\n".join(out)


def list_range(data: list[list[int]]) -> list[int]:
    minval = min(x for lst in data for x in lst)
    maxval = max(x for lst in data for x in lst)
    return list(range(minval, maxval + 1))


def gnuplot(data: list[float], title: str, xlabel: str) -> str:
    """Given a list of numbers, plot the cumulative distribution using gnuplot."""
    data.sort()
    n = len(data)
    cumulative_data = [(i + 1) / n * 100 for i in range(n)]

    with tempfile.NamedTemporaryFile("w") as temp_file:
        for x, y in zip(cumulative_data, data):
            temp_file.write(f"{x} {y}\n")
        temp_file.flush()

        script = f"""
        set terminal dumb
        set title "{title}"
        set xlabel "{xlabel}"
        set nokey
        set style data lines
        plot "{temp_file.name}" using 1:2 with lines
        """

        try:
            return subprocess.check_output(
                ["gnuplot"],
                input=script,
                universal_newlines=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print("Error occurred while running gnuplot:")
            print(e.output)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        type=argparse.FileType(),
        help="The JSON file to read. Should be a list of objects with 'chain_lengths'"
        " and 'chain_results' keys.",
    )
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Plot the cumulative distribution of consensus values (requires gnuplot)",
    )
    parser.add_argument(
        "--save",
        type=argparse.FileType("w"),
        help="File to save the consensus results.",
    )
    args = parser.parse_args()

    @dataclass
    class Instance:
        lengths: list[int]
        results: list[int]

    data = [
        Instance(x["chain_lengths"], x["chain_results"]) for x in json.load(args.file)
    ]
    lengths = [entry.lengths for entry in data]
    results = [entry.results for entry in data]

    records: list[dict[str, Any]] = [
        {
            "name": "lengths (ordinal)",
            "consensus": [
                ord_consensus(x, labels=list_range(lengths)) for x in lengths
            ],
            "k": get_majority([len(x) for x in lengths]),
        },
        *(
            {
                "name": f"results ({kind.value})",
                "consensus": [bin_consensus(x, kind) for x in results],
                "k": get_majority([len(x) for x in results]),
            }
            for kind in BinConsensus
        ),
    ]

    described_records = [
        {"name": x["name"], **describe(x["consensus"]), "k": x["k"]} for x in records
    ]
    header, rows = transpose_table(*records_to_table(described_records))
    fmt_rows = [
        [f"{cell:.3f}" if isinstance(cell, float) else cell for cell in row]
        for row in rows
    ]
    print(render_table(header, fmt_rows))

    if args.plot:
        for x in records:
            print(gnuplot(x["consensus"], x["name"], "Percentile"))

    if args.save:
        json.dump(records, args.save, indent=2)


if __name__ == "__main__":
    main()
