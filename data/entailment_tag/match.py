"""Match input file with synthetic entailment data.

This script takes two JSON files as input, the first one containing the original data
and the second the synthetic entailment data.

The format for the original data is an object containing a 'data' key. This is a list
of objects, each with the following keys:
- context (str): the input context
- answers (str): the gold answer (tagged)
- id (str): the unique identifier for the example

The entailment data is a list of objects, each with the following keys:
- sentence1 (str): the input context
- sentence2 (str): the generated entailment extraction (tagged)
- label (str): the entailment label: ENTAILMENT, NEUTRAL, CONTRADICTION

The output is a JSON list with the evaluation (classifier.py) format:
- input (str): the input context
- output (str): the generated entailment extraction (tagged)
- gold (str): the gold answer (tagged)
- label (str): the entailment label
- id (str): the unique identifier for the example
"""

import argparse
import json
from typing import TextIO


def main(original: TextIO, entailment: TextIO, output: TextIO) -> None:
    og: list[dict[str, str]] = json.load(original)["data"]
    ent: list[dict[str, str]] = json.load(entailment)

    matched: list[dict[str, str]] = []

    for o in og:
        for e in ent:
            if e["sentence1"] == o["context"]:
                matched.append(
                    {
                        "input": o["context"],
                        "output": e["sentence2"],
                        "gold": o["answers"],
                        "label": e["label"],
                        "id": o["id"],
                    }
                )
                break

    json.dump(matched, output, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "original",
        type=argparse.FileType("r"),
        help="The original data file from data/genqa_joint",
    )
    parser.add_argument(
        "entailment",
        type=argparse.FileType("r"),
        help="The entailment data file from data/entailment_tag",
    )
    parser.add_argument(
        "output",
        type=argparse.FileType("w"),
        help="The output file to write the matched data (evaluation format)",
    )
    args = parser.parse_args()
    main(args.original, args.entailment, args.output)
