"""Convert the FGCR dataset json to the format that the GenQA model accepts. This is for
the reconstruction task.

See convert_instance.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from genqa_common import generate_answer_combined_tags, hash_instance, convert_file


def process_relation(causes: list[str], effects: list[str], text: str) -> str:
    start_index = int(float(1e9))
    end_index = -1

    for clause in causes + effects:
        start = text.index(clause)
        end = start + len(clause)

        start_index = min(start_index, start)
        end_index = max(end_index, end)

    assert start_index != int(float(1e9)) and end_index != -1
    return text[start_index:end_index].strip()


def convert_reconstruct(instance: dict[str, Any]) -> list[dict[str, str]]:
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
      "context": "[Cause] one or more of Ecolab's customers were to experience a disastrous outcome [Relation] cause [Effect] the firm's reputation could suffer and it could lose multiple customers",
      "question": "What is the reconstructed sentence?",
      "question_type": "cause",
      "answers": "one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers",
      "id": "573a2c31"
    },
    ```
    """
    text = instance["info"]
    label_map = {"reason": "Cause", "result": "Effect"}

    instances: list[dict[str, str]] = []

    for i, label_data in enumerate(instance["labelData"]):
        relation = label_data["type"]

        events: dict[str, list[str]] = {"reason": [], "result": []}
        for ev_type in ["reason", "result"]:
            for ev_start, ev_end in label_data[ev_type]:
                event = text[ev_start:ev_end]
                events[ev_type].append(event)

        structured = generate_answer_combined_tags(events, label_map, relation)
        answer = process_relation(events["reason"], events["result"], text)

        question = (
            "What is the reconstructed sentence from the cause, relation and effect?"
        )
        if len(instance["labelData"]) > 1:
            question = f"{i} {question}"

        inst = {
            "context": structured,
            "question": question,
            "question_type": relation,
            "answers": answer,
        }
        # There are duplicate IDs in the dataset, so we hash instead.
        inst["id"] = hash_instance(inst)
        instances.append(inst)

    return instances


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--src",
        default="data/raw",
        help="Path to the folder containing the raw data",
    )
    argparser.add_argument(
        "--dst", default="data/reconstruct", help="Path to the output folder"
    )
    args = argparser.parse_args()

    raw_folder = Path(args.src)
    new_folder = Path(args.dst)

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = raw_folder / f"event_dataset_{split}.json"
        new_path = new_folder / f"{split}.json"
        convert_file(raw_path, new_path, convert_instance=convert_reconstruct)


if __name__ == "__main__":
    main()
