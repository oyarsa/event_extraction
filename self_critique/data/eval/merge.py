#!/usr/bin/env python3

import csv
import json
import sys
from pathlib import Path

import typer


def main(files: list[Path]) -> None:
    metrics: list[dict[str, float]] = []

    for file in files:
        data = json.loads(file.read_text())
        data = {k: round(float(v), 4) for k, v in data.items()}
        metrics.append({"file": file.name} | data)

    # Print CSV version so I can copy-paste to Google Sheets
    fieldnames = metrics[0].keys()
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(metrics)


if __name__ == "__main__":
    typer.run(main)
