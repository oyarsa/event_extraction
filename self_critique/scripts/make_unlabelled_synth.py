#!/usr/bin/env python3
"""Make unlabelled data by sampling new causes and effects from the context.

Answers are replaced with generated ones by sampling new causes and effects from the
context. The new answers are in the same tagged format as the original answers.

The clauses are sampled by selecting two random spans of text from the context. They
have a minimum of 5 tokens, and a maximum of half the length of the context.

Input should be a JSON file with a list of dictionaries. Each dictionary should be:

- version: str (doesn't matter)
- data: list of dictionaries

Each dictionary in data should be:
- context: str - The input context for the entry
- question: str - The prompt to go with the context
- question_type: str - The type of relation in the question (e.g. `cause`)
- answers: str - The gold extraction of cause and effect from the context
- id: str - The unique identifier for the entry

The output data has the same shape as the input.
"""

import argparse
import hashlib
import json
import random
from typing import TextIO


def hash_instance(d: dict[str, str]) -> str:
    return hashlib.sha1(str(d).encode("utf-8")).hexdigest()[:8]


def _generate_non_overlapping_spans(
    text_length: int, min_length: int, max_length: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate two non-overlapping spans of text from the context.

    Raises:
        ValueError: We end trying to generate spans of zero length.
    """
    if max_length * 2 > text_length:
        max_length = text_length // 2

    i = random.randint(0, text_length - min_length)
    j = random.randint(i + min_length - 1, min(i + max_length - 1, text_length - 1))

    remaining_length = text_length - (j + 1)
    if remaining_length >= min_length:
        n = random.randint(j + 1, text_length - min_length)
        m = random.randint(n + min_length - 1, min(n + max_length - 1, text_length - 1))
    else:
        n = random.randint(0, i - min_length)
        m = random.randint(n + min_length - 1, i - 1)

    return (i, j), (n, m)


# Span generation can fail, so we retry. 100 times is way too high, so if this happens,
# it's likely a bug.
_MAX_SPAN_ATTEMPTS = 100


def generate_non_overlapping_spans(
    text_length: int, min_length: int, max_length: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Generate two non-overlapping spans of text from the context.

    Retry span generation until we get two valid non-overlapping spans.

    Raises:
        ValueError: If we can't generate non-overlapping spans after 100 attempts.
    """
    count = 0
    while True:
        try:
            return _generate_non_overlapping_spans(text_length, min_length, max_length)
        except ValueError as e:
            count += 1
            if count > _MAX_SPAN_ATTEMPTS:
                raise ValueError(
                    f"Couldn't generate non-overlapping spans after {count} attempts."
                ) from e


def make_unlabelled(entry: dict[str, str]) -> dict[str, str]:
    context_toks = entry["context"].split()

    min_length = 5
    max_length = int(0.5 * len(context_toks))

    (cause_start, cause_end), (effect_start, effect_end) = (
        generate_non_overlapping_spans(len(context_toks), min_length, max_length)
    )

    cause = " ".join(context_toks[cause_start:cause_end])
    effect = " ".join(context_toks[effect_start:effect_end])
    relation = random.choice(["cause", "enable", "prevent"])
    new_answer = f"[Cause] {cause} [Relation] {relation} [Effect] {effect}"

    new_entry = {
        "context": entry["context"],
        "question": entry["question"],
        "question_type": entry["question_type"],
        "answers": new_answer,
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
        epilog="\n".join(__doc__.splitlines()[1:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
