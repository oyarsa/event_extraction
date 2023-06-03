import json
import re
from pathlib import Path
from typing import Annotated

import typer


def get_ratio(data: list[dict[str, str]]) -> float:
    return sum(1 for x in data if x["entailment_label"] == "ENTAILMENT") / len(data)


def main(path: Annotated[Path, typer.Argument()] = Path(".")) -> None:
    pairs: list[tuple[float, float]] = []

    for filename in path.glob("eval_result_?.??.json"):
        num = re.search(r"eval_result_(.*).json", filename.name).group(1)
        data = json.loads(filename.read_text())
        pairs.append((float(num), get_ratio(data)))

    for num, ratio in sorted(pairs):
        print(f"{num:.2f},{ratio:.5f}")


if __name__ == "__main__":
    typer.run(main)
