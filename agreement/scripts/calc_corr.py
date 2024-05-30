#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=Path, nargs="+", help="path to model data")
    parser.add_argument(
        "--add-agreement",
        "-A",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to add agreement to the output. Will break on non-int/bool data.",
    )
    args = parser.parse_args()

    # TODO: Remove this duplication in the format
    header = "{:<30} {:<6} {:<12} {:<10} {:<10}".format(
        "Model", "Valid", "Krippendorff", "Pearson", "Spearman"
    )
    row_fmt = "{:<30} {:<6.4f} {:<12.4f} {:<10.4f} {:<10.4f}"
    metrics_ = ["krippendorff", "pearson", "spearman"]

    if args.add_agreement:
        header += " {:<10}".format("Agreement")
        row_fmt += " {:<10.4f}"
        metrics_.append("agreement")

    print(header)
    for data_path in args.data_paths:
        data = json.loads(data_path.read_text())
        gold = [r["gold"] for r in data]
        pred = [r["pred"] for r in data]

        valid = sum(pred) / len(data)
        metric_results = [
            metrics.calculate_metric(metric, gold, pred) for metric in metrics_
        ]

        print(row_fmt.format(data_path.stem, valid, *metric_results))


if __name__ == "__main__":
    main()
