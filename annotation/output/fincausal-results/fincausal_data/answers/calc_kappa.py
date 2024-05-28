#!/usr/bin/env python3
"""Uses the results of `clean_data.py` to calculate pairwise Cohen's Kappa score."""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from typing import TextIO

from sklearn.metrics import cohen_kappa_score  # type: ignore


def main(infile: TextIO, outfile: TextIO | None) -> None:
    data = json.load(infile)

    names = {a["name"] for d in data for a in d["answers"]}
    kappas: list[float] = []
    name_to_kappas: dict[str, list[float]] = defaultdict(list)

    for name1, name2 in combinations(names, 2):
        answers1: list[bool] = []
        answers2: list[bool] = []

        for item in data:
            # Only consider items where both annotators have an answer
            if {name1, name2} <= {a["name"] for a in item["answers"]}:
                answers1.append(
                    next(a["answer"] for a in item["answers"] if a["name"] == name1)
                )
                answers2.append(
                    next(a["answer"] for a in item["answers"] if a["name"] == name2)
                )

        kappa = cohen_kappa_score(answers1, answers2)
        kappas.append(kappa)
        name_to_kappas[name1].append(kappa)
        name_to_kappas[name2].append(kappa)

        print(f"{name1:<10} {name2:<10}: {kappa:.4f} n={len(answers1)}")

    avg_kappa = sum(kappas) / len(kappas)
    print(f"\nAverage Cohen's Kappa: {avg_kappa:.4f}")

    named_avg_kappas = sorted(
        {
            name: sum(kappas) / len(kappas) for name, kappas in name_to_kappas.items()
        }.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    for name, avg_kappa in named_avg_kappas:
        print(f"{name:<10}: {avg_kappa:.4f}")

    if outfile:
        ordered_names = [name for name, _ in named_avg_kappas]
        json.dump(ordered_names, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "infile", type=argparse.FileType("r"), help="Path to the input file."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=argparse.FileType("w"),
        default=None,
        help="Path to output file with the username ordered by highest average kappa.",
    )
    args = parser.parse_args()
    main(args.infile, args.output)
