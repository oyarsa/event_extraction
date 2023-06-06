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

        epoch, batch = matches[0]
        pairs.append((int(epoch + batch), get_ratio(data)))

    for batch, ratio in sorted(pairs):
        print(f"{batch},{ratio:.5f}")


if __name__ == "__main__":
    typer.run(main)
