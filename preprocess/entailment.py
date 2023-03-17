"""Convert the FGCR dataset json to the format that the GenQA model accepts. This is for
the reconstruction task.

See convert_instance.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from common import hash_instance, extract_relation_span, deduplicate


CLASSES = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]


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
        "label": "ENTAILMENT",
        "id": "354f73c1"
    },
    ```
    """
    # Placeholder while we don't have a way to generate the contradiction class.
    placeholder_classes = ["ENTAILMENT", "CONTRADICTION"]

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
            "label": random.choice(placeholder_classes),
        }
        # There are duplicate IDs in the dataset, so we hash instead.
        inst["id"] = hash_instance(inst)
        instances.append(inst)

    return instances


def randint_except(n: int, exception: int) -> int:
    """Get a random index in [0, n], excluding `exception`.

    The integer is sampled from `random.randint(0, n)`, but if the index is the same as
    the exception, it is retried.

    This is useful when you want to generate a random index that is not the same as the
    current index. See `make_neutral_instances`.
    """
    while True:
        idx = random.randint(0, n)
        if idx != exception:
            return idx


def generate_neutral_instances(instances: list[dict[str, str]]) -> list[dict[str, str]]:
    """Randomly pair instances to create a dataset of neutral examples."""
    new_instances: list[dict[str, str]] = []

    for i, inst1 in enumerate(instances):
        j = randint_except(len(instances) - 1, i)
        inst2 = instances[j]
        assert inst1["id"] != inst2["id"]

        new_inst = {
            "sentence1": inst1["sentence1"],
            "sentence2": inst2["sentence2"],
            "label": "NEUTRAL",
        }
        new_inst["id"] = hash_instance(new_inst)
        new_instances.append(new_inst)

    return new_instances


def convert_file_classification(infile: Path, outfile: Path) -> None:
    """Convert a file from the FGCR format to the text classification (MNLI) format.

    The MNLI format is a list of instances, where each instance is a dictionary with
    keys "sentence1", "sentence2", "label", and "id".

    Example:
    ```json
    [
        {
            "sentence1": "...",
            "sentence2": "...",
            "label": "ENTAILMENT/NEUTRAL/CONTRADICTION",
            "id": "..."
        },
        {
            "sentence1": "...",
            "sentence2": "...",
            "label": "ENTAILMENT/NEUTRAL/CONTRADICTION",
            "id": "..."
        },
        ...
    ]

    Args:
        infile (Path): Path to input JSON file.
        outfile (Path): Path to output JSON file. Folders are created if they don't
            exist.
    """
    with open(infile) as f:
        dataset = json.load(f)

    entailment_instances = [convert_entailment(instance) for instance in dataset]
    unique_entailment = deduplicate(
        item for sublist in entailment_instances for item in sublist
    )
    neutral_instances = generate_neutral_instances(unique_entailment)
    final_instances = unique_entailment + neutral_instances

    outfile.parent.mkdir(exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(final_instances, f)


def main() -> None:
    random.seed(1)

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
