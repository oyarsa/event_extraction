#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="path to model data", type=Path)
    args = parser.parse_args()

    data = json.loads(args.data_path.read_text())
    gold = [r["gold"] for r in data]
    pred = [r["pred"] for r in data]

    for metric_name in metrics.AVAILABLE_METRICS:
        metric = metrics.calculate_metric(metric_name, gold, pred)
        print(f"{metric_name.capitalize():<15}: {metric:.4f}")


if __name__ == "__main__":
    main()
