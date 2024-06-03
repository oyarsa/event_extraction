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

from . import metrics


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


def calculate_all_metrics(
    data_paths: list[str], metrics_chosen: list[str]
) -> dict[str, dict[str, float]]:
    path_metric_value: dict[str, dict[str, float]] = {}

    for data_path in data_paths:
        data = load_json(data_path)
        path_metric_value[data_path] = {
            **{metric: calculate_metrics(metric, data) for metric in metrics_chosen},
        }

    return path_metric_value


def define_output_style(
    data_paths: list[str], metric_names: list[str]
) -> tuple[int, list[int], list[str]]:
    padding_model = (
        max(len(os.path.basename(data_path)) for data_path in data_paths) + 2
    )
    padding_metrics = [len(metric) + 3 for metric in metric_names]
    header = [
        "Model file".ljust(padding_model),
        *[
            metric.capitalize().ljust(padding)
            for padding, metric in zip(padding_metrics, metric_names)
        ],
    ]
    return padding_model, padding_metrics, header


def print_output(
    data_paths: list[str],
    metric_names: list[str],
    path_metric_value: dict[str, dict[str, float]],
    padding_model: int,
    padding_metrics: list[int],
    header: list[str],
):
    output = [" ".join(header)]
    for data_path in data_paths:
        row = [
            os.path.basename(data_path).ljust(padding_model),
            *[
                f"{path_metric_value[data_path][metric]:.4f}".ljust(padding)
                for padding, metric in zip(padding_metrics, metric_names)
            ],
        ]
        output.append(" ".join(row))
    print("\n".join(output))


def main() -> None:
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

    path_metric_value = calculate_all_metrics(args.data_paths, metrics_chosen)
    model_padding, padding, header = define_output_style(
        args.data_paths, metrics_chosen
    )
    print_output(
        args.data_paths,
        metrics_chosen,
        path_metric_value,
        model_padding,
        padding,
        header,
    )


if __name__ == "__main__":
    main()
