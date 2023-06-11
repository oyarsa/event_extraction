#!/usr/bin/env python3
"""Convert ChatGPT's line format to GenQA's tagged format.

Also convert field names: answer -> gold, pred -> output, text -> input.
"""
import json
import sys
from pathlib import Path


def find_part(part: str, lines: list[str]) -> str:
    for line in lines:
        if line.startswith(part):
            return line.split(":", 1)[1].strip()
    return ""


def convert_extract(extract: str) -> str:
    lines = extract.splitlines()
    cause = find_part("Cause", lines)
    effect = find_part("Effect", lines)
    relation = find_part("Relation", lines)
    return f"[Cause] {cause} [Relation] {relation} [Effect] {effect}"


def rcf() -> None:
    convert_extract(
        "Cause: The price of oil has risen by 50% in the last year.\n"
        "Effect: The price of nuclear uranium decreased.\n"
        "Relation: cause"
    )
    convert_extract(
        "Effect: The price of nuclear uranium decreased.\n"
        "Cause: War in Iraq.\n"
        "Cause: The price of oil has risen by 50% in the last year.\n"
        "Relation: cause"
    )


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python convert.py <data.json>")
        sys.exit(1)

    data_file = Path(sys.argv[1])
    data = json.loads(data_file.read_text())
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
