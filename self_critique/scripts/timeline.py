#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from tqdm import tqdm


def to_num(x: str) -> int | float:
    try:
        return int(x)
    except ValueError:
        return float(x)


def extract_num(s: str) -> int | float:
    match = re.search(r"-?\d+(\.\d+)?", s)
    assert match is not None
    return to_num(match.group())


def load_data(dir: Path) -> list[tuple[int | float, list[dict[str, Any]]]]:
    files = list(dir.glob("eval_result_*.json"))
    nums = (extract_num(f.name) for f in files)
    nums = sorted(x for x in nums if x < 0 or not isinstance(x, int))

    sorted_files = [dir / f"eval_result_{x}.json" for x in nums]
    og_data = [json.loads(p.read_text()) for p in sorted_files]
    return list(zip(nums, og_data))


def keep_non_entailment(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        inst
        for inst in instances
        if inst["extracts"] is not None
        and any(x["label"] != "entailment" for x in inst["extracts"])
    ]


def main(path: Annotated[Path, typer.Argument()] = Path(".")) -> None:
    data = load_data(path)

    instances = [
        {
            "id": d["id"],
            "original": d["original"],
            "gold": d["context"],
        }
        for d in data[0][1]
    ]
    for batch, dd in tqdm(data):
        for d in dd:
            for inst in instances:
                if d["id"] != inst["id"]:
                    continue

                if "extracts" not in inst:
                    inst["extracts"] = []
                inst["extracts"].append(
                    {
                        "batch": batch,
                        "pred": d["rl_extract_txt"],
                        "label": d["entailment_label"],
                    }
                )
    instances = keep_non_entailment(instances)

    sys.stdout.write(json.dumps(instances, indent=2))


if __name__ == "__main__" and not hasattr(sys, "ps1"):  # not in interactive mode
    typer.run(main)


def rcf() -> None:
    "Rich comments"
    type(Path().name)
