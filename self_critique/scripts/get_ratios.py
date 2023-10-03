#!/usr/bin/env python3
# pyright: basic
import json
import re
from pathlib import Path
from typing import Annotated

import typer


def main(
    path: Annotated[Path, typer.Argument()] = Path("."),
    key: str = "reward_label",
    class_: Annotated[str, typer.Option("--class", "-c")] = "entailment",
) -> None:
    pairs: list[tuple[float, float, float]] = []

    for filename in path.iterdir():
        matches = re.findall(r"mini_eval_result_(\d+)\.(\d+).json", filename.name)
        if not matches:
            continue
        data = json.loads(filename.read_text())

        epoch, batch = map(int, matches[0])
        ratio = sum(x[key].casefold() == class_.casefold() for x in data) / len(data)
        pairs.append((epoch, batch, ratio))

    max_batch = max(batch for _, batch, _ in pairs)
    for epoch, batch, ratio in sorted(pairs):
        idx = epoch * max_batch + batch
        print(f"{idx},{ratio:.5f}")


if __name__ == "__main__":
    typer.run(main)
