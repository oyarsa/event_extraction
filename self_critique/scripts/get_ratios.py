#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Annotated

import typer


def get_ratio(data: list[dict[str, str]]) -> float:
    return sum(x["entailment_label"] == "ENTAILMENT" for x in data) / len(data)


def main(path: Annotated[Path, typer.Argument()] = Path(".")) -> None:
    pairs: list[tuple[float, float]] = []

    for filename in path.iterdir():
        matches = re.findall(r"mini_eval_result_(\d+)\.(\d+).json", filename.name)
        if not matches:
            continue
        data = json.loads(filename.read_text())

        epoch, batch = map(int, matches[0])
        pairs.append((epoch, batch, get_ratio(data)))

    max_batch = max(batch for _, batch, _ in pairs)
    for epoch, batch, ratio in sorted(pairs):
        idx = epoch * max_batch + batch
        print(f"{idx},{ratio:.5f}")


if __name__ == "__main__":
    typer.run(main)
