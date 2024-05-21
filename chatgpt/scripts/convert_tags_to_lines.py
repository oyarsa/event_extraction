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

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

sys.path.append(str(Path(__file__).parents[1]))
from metrics import parse_instance_tags


def convert_answer(item: str, add_relation: bool) -> str:
    entities, relation = parse_instance_tags(item)
    rows = [
        f'Cause: {"| ".join(entities["Cause"])}',
        f'Effect: {"| ".join(entities["Effect"])}',
    ]
    if add_relation:
        rows.append(f"Relation: {relation}")
    return "\n".join(rows)


def convert_item(item: dict[str, str], add_relation: bool) -> dict[str, str]:
    return {**item, "answers": convert_answer(item["answers"], add_relation)}


def main(input_file: TextIO, key: str | None, add_relation: bool) -> None:
    data = json.load(input_file)
    new_data: dict[str, list[dict[str, str]]] | list[dict[str, str]]

    if key is not None:
        new_data = {
            **data,
            "data": [convert_item(item, add_relation) for item in data[key]],
        }
    else:
        new_data = [convert_item(item, add_relation) for item in data]

    print(json.dumps(new_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        default=sys.stdin,
        nargs="?",
        help="Input file",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Key to extract the data from, if the file isn't a list",
    )
    parser.add_argument(
        "--add-relation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add the relation to the output. Should be True for FCR and False for"
        " FinCausal",
    )
    args = parser.parse_args()
    main(args.input_file, args.key, args.add_relation)
