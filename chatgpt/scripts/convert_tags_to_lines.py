#!/usr/bin/env python3
"""Converts tags format to lines format for the extraction task.

This:
```
[Cause] one or more of Ecolab's customers were to experience a disastrous outcome [Relation] cause [Effect] the firm's reputation could suffer and it could lose multiple customers
```

Becomes:
```
Cause: one or more of Ecolab's customers were to experience a disastrous outcome
Effect: the firm's reputation could suffer and it could lose multiple customers
Relation: cause
```

Usage:
```
python scripts/convert_tags_to_lines.py < input.json > output.json
```

If the data is inside a key, use the key as the first argument:
```
python scripts/convert_tags_to_lines.py data < input.json > output.json
```
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from metrics import parse_instance_tags  # noqa: E402


def convert_answer(item: str) -> str:
    entities, relation = parse_instance_tags(item)
    rows = [
        f'Cause: {"| ".join(entities["Cause"])}',
        f'Effect: {"| ".join(entities["Effect"])}',
        f"Relation: {relation}",
    ]
    return "\n".join(rows)


def convert_item(item: dict[str, str]) -> dict[str, str]:
    return {**item, "answers": convert_answer(item["answers"])}


def main() -> None:
    data = json.load(sys.stdin)
    new_data: dict[str, list[dict[str, str]]] | list[dict[str, str]]
    if len(sys.argv) >= 2:
        key = sys.argv[1]
        new_data = {
            **data,
            "data": [convert_item(item) for item in data[key]],
        }
    else:
        new_data = [convert_item(item) for item in data]

    print(json.dumps(new_data, indent=2))


if __name__ == "__main__":
    main()
