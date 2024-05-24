#!/usr/bin/env python3
"""Make unlabelled data by splitting the context into two parts.

The context is split into two parts, the first part is used as the context and the
second part is used as the answer.

Input should be a JSON file with a list of dictionaries. Each dictionary should be:

- version: str (doesn't matter)
- data: list of dictionaries

Each dictionary in data should be:
- context: str - The input context for the entry
- question: str - The prompt to go with the context
- question_type: str - The type of relation in the question (e.g. `cause`)
- answers: str - The gold extraction of cause and effect from the context
- id: str - The unique identifier for the entry

The output has the same shape as the input.
"""

import argparse
import hashlib
import json
import random
from typing import TextIO


def hash_instance(d: dict[str, str]) -> str:
    return hashlib.sha1(str(d).encode("utf-8")).hexdigest()[:8]


def sample_length(min_length: int, max_length: int) -> int:
    return random.choice(range(min_length, max_length + 1))


def make_unlabelled(entry: dict[str, str]) -> dict[str, str]:
    context_toks = entry["context"].split()

    min_length = int(0.1 * len(context_toks))
    max_length = int(0.9 * len(context_toks))
    length = sample_length(min_length, max_length)

    context, target = context_toks[:length], context_toks[length:]

    new_entry = {
        "context": " ".join(context),
        "question": entry["question"],
        "question_type": entry["question_type"],
        "answers": " ".join(target),
    }
    return new_entry | {"id": hash_instance(new_entry)}


def main(input: TextIO, output: TextIO, seed: int) -> None:
    random.seed(seed)

    data = json.load(input)["data"]
    new_data = [make_unlabelled(entry) for entry in data]

    json.dump({"data": new_data}, output, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=argparse.FileType("r"),
        help="Input file",
    )
    parser.add_argument(
        "output",
        type=argparse.FileType("w"),
        help="Output file",
    )
    parser.add_argument("--seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()
    main(args.input, args.output, args.seed)
