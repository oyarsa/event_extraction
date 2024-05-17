# pyright: basic
"""Convert FinCausal data to the GenQA format."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def process(template: str, data: Any) -> dict[str, str]:
    context = data.text.strip()
    out = {
        "context": context,
        "question": "What are the events?",
        "question_type": "cause",
    }
    try:
        cause = data.cause.strip()
        effect = data.effect.strip()
        return out | {
            "answers": template.format(cause=cause, effect=effect),
            "id": str(hash(context + cause + effect)),
        }
    except AttributeError:
        return out | {"answers": None, "id": str(hash(context))}


def main(input_file: Path, output_file: Path) -> None:
    input_data = pd.read_csv(input_file, delimiter=";")
    input_data.columns = [s.lower().strip() for s in input_data.columns]

    answer_template = """[Cause] {cause} [Relation] cause [Effect] {effect}"""
    examples = [process(answer_template, d) for d in input_data.itertuples()]

    print(f"There are {len(examples)} data in the input file.")

    data_output = {"version": "v1.0", "data": examples}
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(data_output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file", type=Path, help="Input CSV (semicolon separated) file path."
    )
    parser.add_argument("output_file", type=Path, help="Output JSON file path.")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
