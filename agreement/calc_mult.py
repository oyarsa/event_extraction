#!/usr/bin/env python3

import argparse
import json
import os.path
from dataclasses import dataclass

import metrics


@dataclass
class DataEntry:
    gold: int
    pred: int


def calculate_metrics(metric: str, data: list[DataEntry]) -> tuple[float, float]:
    model_valid = sum(r.pred for r in data) / len(data)
    metric_val = metrics.calculate_metric(
        metric, [r.gold for r in data], [r.pred for r in data]
    )

    return model_valid, metric_val


def load_json(file_path: str) -> list[DataEntry]:
    with open(file_path) as f:
        data = json.load(f)
        return [DataEntry(gold=d["gold"], pred=d["pred"]) for d in data]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metric", help="metric to calculate", choices=metrics.AVAILABLE_METRICS
    )
    parser.add_argument("data_paths", nargs="+", help="path to model data")
    args = parser.parse_args()

    # Table Header
    print(f"{'Model File':<30} {'Valid':<10} {args.metric.capitalize():<15}")

    for data_path in args.data_paths:
        data = load_json(data_path)
        valid, metric = calculate_metrics(args.metric, data)

        print(f"{os.path.basename(data_path):<30} {valid:<10.4f} {metric:.4f}")


if __name__ == "__main__":
    main()
