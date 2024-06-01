"""Convert the MAVEN dataset to the tagged extraction format."""

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from typing import Any, TextIO

from beartype.door import is_bearable

HASH_KEYS = ("context", "answers")


def hash_instance(
    instance: dict[str, Any], keys: Sequence[str] = HASH_KEYS, length: int = 8
) -> str:
    key = "".join(str(instance[k]) for k in keys)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def process_data(data: list[dict[str, Any]], straight: bool) -> list[dict[str, str]]:
    question = "What are the events?"
    question_type = "cause"

    if straight:
        answer_template = "{cause}"
    else:
        answer_template = "[Cause] {cause} [Relation] cause [Effect]"

    examples: list[dict[str, str]] = []
    for item in data:
        content = item["content"]

        reference_sentences: list[str] = [
            content[mention["sent_id"]]["sentence"]
            for event in item["events"]
            for mention in event["mention"]
            if event["type"] == "Causation"
        ]
        if not reference_sentences:
            continue

        sentences: list[str] = [sent["sentence"] for sent in content]

        indices = [
            *sorted({sentences.index(s) for s in reference_sentences}),
            len(sentences),
        ]
        current_sentence, next_sentence = indices[:2]

        start = random.randint(0, current_sentence)
        end = random.randint(current_sentence + 1, next_sentence)

        context = ". ".join(sentences[start:end])
        assert context, "Clipped context is empty."

        answer = answer_template.format(cause=sentences[current_sentence])

        instance = {
            "context": context,
            "question": question,
            "question_type": question_type,
            "answers": answer,
        }
        examples.append(instance | {"id": hash_instance(instance)})

    return examples


def main(input_file: TextIO, output_file: TextIO, seed: int, straight: bool) -> None:
    random.seed(seed)

    data = [json.loads(line) for line in input_file]
    if not is_bearable(data, list[dict[str, Any]]):
        raise ValueError("Invalid input data format.")

    processed = process_data(data, straight)

    output = {"version": "v1.0", "data": processed}
    json.dump(output, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="Path to the MAVEN JSONLines file.",
    )
    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="Path to the output JSON tagged file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--straight",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use straight causes instead of faux-tagged.",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.seed, args.straight)
