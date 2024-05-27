import argparse
import hashlib
import json
from typing import TextIO

from nltk.tokenize.treebank import TreebankWordDetokenizer


def hash_instance(data: dict[str, str], length: int = 8) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[
        :length
    ]


def seqlabel_to_tag(data: list[tuple[str, str]]) -> dict[str, str]:
    cause_tokens: list[str] = []
    effect_tokens: list[str] = []
    current_tag: str | None = None

    for token, tag in data:
        match tag:
            case "B-Cause":
                if current_tag == "Cause":
                    cause_tokens.append("|")
                current_tag = "Cause"
                cause_tokens.append(token)
            case "B-Effect":
                if current_tag == "Effect":
                    effect_tokens.append("|")
                current_tag = "Effect"
                effect_tokens.append(token)
            case "I-Cause":
                if current_tag == "Cause":
                    cause_tokens.append(f"{token}")
            case "I-Effect":
                if current_tag == "Effect":
                    effect_tokens.append(f"{token}")
            case _:
                current_tag = None

    detokenizer = TreebankWordDetokenizer()
    context = detokenizer.detokenize([token for token, _ in data])
    cause = detokenizer.detokenize(cause_tokens)
    effect = detokenizer.detokenize(effect_tokens)

    answers = f"[Cause] {cause} [Relation] cause [Effect] {effect}"

    result = {
        "context": context,
        "question": "What are the events?",
        "question_type": "cause",
        "answers": answers,
    }
    return result | {"id": hash_instance(result)}


def main(input_file: TextIO, output_file: TextIO) -> None:
    items: list[list[tuple[str, str]]] = []

    blocks = input_file.read().split("\n\n")
    for block_ in blocks:
        block = block_.strip()
        if not block:
            continue

        current: list[tuple[str, str]] = []
        for line in block.splitlines():
            token, tag = line.strip().split()
            current.append((token, tag))

        items.append(current)

    tag_data = [seqlabel_to_tag(item) for item in items]
    json.dump(tag_data, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sequence labelled data to tagged data",
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
