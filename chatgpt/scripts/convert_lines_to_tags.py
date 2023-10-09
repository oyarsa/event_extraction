#!/usr/bin/env python3
"Convert extraction in Lines format to Tags format."
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


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python convert_lines_to_tags.py <data.json>")
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


if __name__ == "__main__":
    main()
