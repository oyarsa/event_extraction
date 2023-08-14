"""Convert the FGCR dataset json to the format that the GenQA model accepts. This is for
the reconstruction task.

See convert_instance.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from common import deduplicate, hash_instance


def add_instance_id(instance: dict[str, Any]) -> dict[str, Any]:
    """Add an `id` field to the instance."""
    instance["id"] = hash_instance(instance)
    return instance


def convert_entailment(instance: dict[str, Any]) -> dict[str, str]:
    """Convert a FGCR-format instance into a reconstruction-format instance.

    This first generates the structured information from the input, then extracts the
    truncated sentence version from the input and creates a dataset that maps structed
    -> truncated.

    This (raw input):
    ```json
    {
      "context": "If one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers as a result.",
      "question": "What are the events?",
      "question_type": "cause",
      "answers": "[Cause] one or more of Ecolab's customers were to experience a disastrous outcome [Relation] cause [Effect] the firm's reputation could suffer and it could lose multiple customers",
      "id": "86951ffe"
    },
    ```
    Becomes:
    ```json
    {
        "sentence1": "If one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers as a result.",
        "sentence2": "[Cause] one or more of Ecolab's customers were to experience a disastrous outcome [Relation] cause [Effect] the firm's reputation could suffer and it could lose multiple customers",
        "label": "ENTAILMENT",
        "id": "354f73c1"
    },
    ```
    """
    inst = {
        "sentence1": instance["context"],
        "sentence2": instance["answers"],
        "label": "ENTAILMENT",
    }
    return add_instance_id(inst)


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
        new_instances.append(add_instance_id(new_inst))

    return new_instances


def parse_instance(answer: str) -> tuple[dict[str, list[str]], str] | None:
    """Parse string answer to separate into class and spans
    Simple case:
    [Cause] This is a cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Effect] This effect 1 | This effect 2
    """
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return None
    causes, relation, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())
    relation = relation.strip()

    return {
        "Cause": causes,
        "Effect": effects,
    }, relation


def swap_cause_effect(instance: dict[str, str]) -> dict[str, str] | None:
    """Swap the cause and effect in an instance."""
    inst = parse_instance(instance["sentence2"])
    if inst is None:
        return None
    entities, relation = inst

    new_instance = {
        "sentence1": instance["sentence1"],
        "sentence2": (
            "[Cause] "
            + " | ".join(entities["Effect"])
            + f" [Relation] {relation} [Effect] "
            + " | ".join(entities["Cause"])
        ),
        "label": "CONTRADICTION",
    }
    return add_instance_id(new_instance)


def generate_contradictory_instances(
    instances: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Generate contradictions by swapping cause and effect."""
    return [
        new_inst for instance in instances if (new_inst := swap_cause_effect(instance))
    ]


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
    dataset = json.loads(infile.read_text())["data"]

    entailment_instances = deduplicate(
        convert_entailment(instance) for instance in dataset
    )
    neutral_instances = generate_neutral_instances(entailment_instances)
    contradictory_instances = generate_contradictory_instances(entailment_instances)
    final_instances = entailment_instances + neutral_instances + contradictory_instances
    random.shuffle(final_instances)

    outfile.parent.mkdir(exist_ok=True, parents=True)
    with outfile.open("w") as f:
        json.dump(final_instances, f)


def main() -> None:
    random.seed(1)

    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--src",
        type=Path,
        default="data/genqa_joint",
        help="Path to the folder containing the raw data",
    )
    argparser.add_argument(
        "--dst",
        type=Path,
        default="data/entailment_tag",
        help="Path to the output folder",
    )
    args = argparser.parse_args()

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = args.src / f"{split}.json"
        new_path = args.dst / f"{split}.json"
        convert_file_classification(raw_path, new_path)


if __name__ == "__main__":
    main()
