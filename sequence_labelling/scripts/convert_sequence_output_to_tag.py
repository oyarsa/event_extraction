#!/usr/bin/env python3
"""Convert the output of sequence labelling models to tagged.

The input file should contain sequence labelled data. It is separated by blocks with an
empty line between them. Each blocks represents an instance of the dataset.

Each line of the block is a pair of token and tag separated by a space.

The output file is a JSON file with tagged data. Each instance is a dictionary with the
following keys:
- id: Unique identifier for the instance.
- input: The input context of the instance.
- gold: The tagged gold answer.
- output: The tagged model output.

The answer has the format "[Cause] <cause> [Relation] cause [Effect] <effect>".

The goal is that this output can be directly use for evaluation. E.g. self_critique's
run_reward and eval_std scripts.
"""

import argparse
import hashlib
import json
from typing import TextIO

from nltk.tokenize.treebank import TreebankWordDetokenizer


def hash_instance(data: dict[str, str], keys: list[str], length: int = 8) -> str:
    subdata = {k: data[k] for k in keys}
    return hashlib.sha256(json.dumps(subdata, sort_keys=True).encode()).hexdigest()[
        :length
    ]


def seq_clause_to_tag(
    detokeniser: TreebankWordDetokenizer, tokens: list[str], tags: list[str]
) -> str:
    """Convert a sequence of tokens and their corresponding BIO tags into tagged format.

    The tagged representation is obtained by walking the sequence of tokens and tags,
    adding tokens to the appropriate cause/effect list depending on the tag. This means
    skipping "O" tags and managing start/end of segments using the BI tags.

    The tokens for each clause are detokenized to form sentences and fit into the tagged
    format "[Cause] <cause> [Relation] cause [Effect] <effect>". The [Relation] is fixed
    to "cause" since we have no information from the sequence labels.

    NB: When an I tag (I-Cause or I-Effect) is encountered and the current tag is None,
    the function starts a new current_tag of the corrsponding type. This handles cases
    where the sequence starts with an I tag, which is technically invalid but may occur
    in practice.

    NB: Detokenisation uses the TreebankWordDetokenizer from NLTK. This means it
    assumes that the sentences were tokenised by NLTK in the first place.

    Raises:
        ValueError: If an invalid tag is encountered. The valid tags are "B-Cause",
        "I-Cause", "B-Effect", "I-Effect", and "O".
    """
    cause_tokens: list[str] = []
    effect_tokens: list[str] = []
    current_tag: str | None = None

    for token, tag in zip(tokens, tags):
        match tag:
            case "B-Cause":
                if current_tag == "Cause":
                    cause_tokens.append("|")
                current_tag = "Cause"
                cause_tokens.append(token)
            case "I-Cause":
                if current_tag is None:
                    current_tag = "Cause"
                if current_tag == "Cause":
                    cause_tokens.append(f"{token}")
            case "B-Effect":
                if current_tag == "Effect":
                    effect_tokens.append("|")
                current_tag = "Effect"
                effect_tokens.append(token)
            case "I-Effect":
                if current_tag is None:
                    current_tag = "Effect"
                if current_tag == "Effect":
                    effect_tokens.append(f"{token}")
            case "O":
                current_tag = None
            case _:
                raise ValueError(f"Invalid tag: {tag}")

    cause = detokeniser.detokenize(cause_tokens)
    effect = detokeniser.detokenize(effect_tokens)
    return f"[Cause] {cause} [Relation] cause [Effect] {effect}"


def seq_item_to_tag(
    detokeniser: TreebankWordDetokenizer, item: list[tuple[str, str, str]]
) -> dict[str, str]:
    """Convert a sequence labelled item to tagged data.

    Args:
        detokeniser: TreebankWordDetokenizer instance to convert tokens to sentences.
        item: List of tuples with token, gold tag, and predicted tag.

    Returns:
        Dictionary with the tagged data for the item with the keys "context", "question",
        "question_type", "answers", and "output". The "question" and "question_type" are
        fixed to "What are the events?" and "cause" respectively.
    """
    context = [token for token, _, _ in item]
    gold_tags = [gold for _, gold, _ in item]
    pred_tags = [pred for _, _, pred in item]

    out = {
        "input": detokeniser.detokenize(context),
        "gold": seq_clause_to_tag(detokeniser, context, gold_tags),
        "output": seq_clause_to_tag(detokeniser, context, pred_tags),
    }
    return out | {"id": hash_instance(out, ["input", "gold"])}


def main(input_file: TextIO, output_file: TextIO) -> None:
    items: list[list[tuple[str, str, str]]] = []

    blocks = input_file.read().split("\n\n")
    for block_ in blocks:
        block = block_.strip()
        if not block:
            continue

        current: list[tuple[str, str, str]] = []
        for line in block.splitlines():
            if line.startswith("#"):
                continue

            token, gold_tag, pred_tag = line.strip().split()
            current.append((token, gold_tag, pred_tag))

        items.append(current)

    detokeniser = TreebankWordDetokenizer()
    tag_data = [seq_item_to_tag(detokeniser, item) for item in items]
    json.dump(tag_data, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="Input text file with sequence labelled data",
    )
    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="Output JSON file with tagged data",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
