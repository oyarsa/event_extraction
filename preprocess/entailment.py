"""Convert the FGCR dataset json to the format that the GenQA model accepts. This is for
the reconstruction task.

See convert_instance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import hash_instance, extract_relation_span, deduplicate


def convert_entailment(instance: dict[str, Any]) -> list[dict[str, str]]:
    """Convert a FGCR-format instance into a reconstruction-format instance.

    This first generates the structured information from the input, then extracts the
    truncated sentence version from the input and creates a dataset that maps structed
    -> truncated.

    This (raw input):
    ```json
    {
        "tid": 2771,
        "info": "If one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers as a result.",  # noqa
        "extraInfo": null,
        "labelData": [
        {
            "type": "cause",
            "reason": [
            [
                3,
                76
            ]
            ],
            "result": [
            [
                78,
                149
            ]
            ]
        }
        ]
    },
    ```
    Becomes:
    ```json
    {
        "sentence1": "If one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers as a result.",
        "sentence2": "one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers",
        "label": "entailment",
        "id": "354f73c1"
    },
    ```
    """
    text = instance["info"]

    instances: list[dict[str, str]] = []

    for label_data in instance["labelData"]:
        events: dict[str, list[str]] = {"reason": [], "result": []}

        for ev_type in ["reason", "result"]:
            for ev_start, ev_end in label_data[ev_type]:
                event = text[ev_start:ev_end]
                events[ev_type].append(event)

        span = extract_relation_span(events["reason"], events["result"], text)

        inst = {
            "sentence1": text,
            "sentence2": span,
            "label": "entailment",
        }
        # There are duplicate IDs in the dataset, so we hash instead.
        inst["id"] = hash_instance(inst)
        instances.append(inst)

    return instances


def convert_file_classification(infile: Path, outfile: Path) -> None:
    with open(infile) as f:
        dataset = json.load(f)

    nested_instances = [convert_entailment(instance) for instance in dataset]
    transformed = deduplicate(item for sublist in nested_instances for item in sublist)

    outfile.parent.mkdir(exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(transformed, f)


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--src",
        default="data/raw",
        help="Path to the folder containing the raw data",
    )
    argparser.add_argument(
        "--dst", default="data/entailment", help="Path to the output folder"
    )
    args = argparser.parse_args()

    raw_folder = Path(args.src)
    new_folder = Path(args.dst)

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = raw_folder / f"event_dataset_{split}.json"
        new_path = new_folder / f"{split}.json"
        convert_file_classification(raw_path, new_path)


if __name__ == "__main__":
    main()
