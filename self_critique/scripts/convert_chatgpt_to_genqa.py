#!/usr/bin/env python3
# pyright: basic
"""Convert ChatGPT's line format to GenQA's tagged format.

Also convert field names: answer -> gold, pred -> output, text -> input.
"""
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Instance:
    cause: str
    effect: str
    relation: str


def parse_instance_lines(extract: str) -> Instance:
    matches = re.findall(r"Cause:(.*)Effect:(.*)Relation:(.*)", extract, re.DOTALL)
    if not matches:
        return Instance(cause="", effect="", relation="cause")

    cause, effect, relation = matches[0]
    return Instance(
        cause=cause.strip(),
        effect=effect.strip(),
        relation=relation.strip(),
    )


def convert_extract(extract: str) -> str:
    inst = parse_instance_lines(extract)
    return f"[Cause] {inst.cause} [Relation] {inst.relation} [Effect] {inst.effect}"


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python convert.py <data.json>")
        sys.exit(1)

    data = json.loads(Path(sys.argv[1]).read_text())
    converted = [
        {
            "input": d["text"],
            "output": convert_extract(d["pred"]),
            "gold": convert_extract(d["answer"]),
        }
        for d in data
    ]
    json.dump(converted, sys.stdout, indent=2)


if __name__ == "__main__" and not hasattr(sys, "ps1"):
    main()
