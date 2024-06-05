#!/usr/bin/env python3
import json
import re
from pathlib import Path
from typing import Any

import typer


def parse_instance(answer: str) -> tuple[list[str], list[str], str]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return [], [], "cause"

    causes, relation, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())
    relation = relation.strip()

    return causes, effects, relation


def transform_str(tags: str) -> str:
    causes, effects, relation = parse_instance(tags)
    if not causes or not effects:
        return ""

    causes = " AND ".join(causes)
    effects = " AND ".join(effects)
    return "\n".join(
        [
            f"Cause: {causes}",
            f"Relation: {relation}",
            f"Effect: {effects}",
        ]
    )


def transform(input_file: Path, output_file: Path) -> None:
    data = json.loads(input_file.read_text())

    new_data: list[dict[str, Any]] = []
    for entry in data:
        new_entry = entry.copy()
        new_entry["output"] = transform_str(entry["output"])
        new_entry["gold"] = transform_str(entry["gold"])
        new_data.append(new_entry)

    output_file.write_text(json.dumps(new_data, indent=4))


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False
)


@app.command()
def main(files: list[Path]) -> None:
    for file in files:
        new_dir = file.parent / "transformed"
        new_dir.mkdir(exist_ok=True, parents=True)
        transform(file, new_dir / file.name)


if __name__ == "__main__":
    app()
