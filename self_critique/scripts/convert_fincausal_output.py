#!/usr/bin/env python3
"""Convert tagged (GenQA) model output to FinCausal's CSV format for evaluation.

The goal is to use the original FinCausal evaluation script with our output.
"""

import argparse
import json
import re
from pathlib import Path


def parse_instance(text: str) -> tuple[str, str]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\].*?\[Effect\](.*?)$", text)
    if not matches:
        return "", ""

    causes, effects = matches[0]
    return causes.strip(), effects.strip()


def main(input_file: Path, ref: Path, pred: Path) -> None:
    data = json.loads(input_file.read_text())

    header = "Index; Text; Cause; Effect"
    ref_output: list[str] = []
    pred_output: list[str] = []

    for i, d in enumerate(data):
        text = d["input"]

        ref_cause, ref_effect = parse_instance(d["gold"])
        ref_output.append(f"{i}; {text}; {ref_cause}; {ref_effect}")

        pred_cause, pred_effect = parse_instance(d["output"])
        pred_output.append(f"{i}; {text}; {pred_cause}; {pred_effect}")

    ref.write_text("\n".join([header, *ref_output]))
    pred.write_text("\n".join([header, *pred_output]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the JSON input file. The output will be in the same folder.",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        help="Path to the output reference CSV file",
        default="ref.csv",
    )
    parser.add_argument(
        "--pred",
        type=Path,
        help="Path to the output prediction CSV file",
        default="pred.csv",
    )
    args = parser.parse_args()
    main(args.input_file, args.ref, args.pred)
