#!/usr/bin/env python3
"""Calculate automated agreement metrics for evaluation files.

The input file should be a JSON file with a list of objects with the following format:
- gold (int): The human evaluation whether the example is valid.
- pred (int): The model's prediction the example.

See metrics.py for the available metrics.
"""

import argparse
import json
import os.path
from dataclasses import dataclass

import metrics


@dataclass
class DataEntry:
    gold: int
    pred: int


def calculate_metrics(metric: str, data: list[DataEntry]) -> float:
    return metrics.calculate_metric(
        metric, [r.gold for r in data], [r.pred for r in data]
    )


def load_json(file_path: str) -> list[DataEntry]:
    with open(file_path) as f:
        data = json.load(f)
        return [DataEntry(gold=d["gold"], pred=d["pred"]) for d in data]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("data_paths", type=str, nargs="+", help="path to model data")
    parser.add_argument(
        "--metrics",
        "-m",
        nargs="+",
        help="metrics to calculate",
        default=metrics.AVAILABLE_METRICS,
        choices=metrics.AVAILABLE_METRICS,
    )
    args = parser.parse_args()

    metrics_chosen = sorted(args.metrics)

    header = [
        "Model file".ljust(30),
        *[metric.upper().ljust(15) for metric in ["Valid", *metrics_chosen]],
    ]
    print(" ".join(header))

    for data_path in args.data_paths:
        data = load_json(data_path)
        metric_values: list[float] = [
            sum(r.pred for r in data) / len(data),
            *(calculate_metrics(metric, data) for metric in metrics_chosen),
        ]
        row = [
            os.path.basename(data_path).ljust(30),
            *[f"{metric:.4f}".ljust(15) for metric in metric_values],
        ]
        print(" ".join(row))


if __name__ == "__main__":
    main()
