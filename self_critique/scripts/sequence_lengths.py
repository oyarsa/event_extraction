import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from typing import TextIO

from transformers import AutoTokenizer


@dataclass
class Entry:
    sentence1: str
    sentence2: str | None


def load_data(file: TextIO) -> list[Entry]:
    return [
        Entry(sentence1=x["sentence1"], sentence2=x.get("sentence2"))
        for x in json.load(file)
    ]


def calc_max_length(data: list[Entry], tokenizer: AutoTokenizer) -> int:
    """Calculate the maximum sequence length from tokenized data."""
    return max(
        len(input["input_ids"])
        for item in data
        if (input := tokenizer(item.sentence1, item.sentence2, add_special_tokens=True))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate the maximum token sequence length from a JSON dataset."
    )
    parser.add_argument(
        "model_name", type=str, help="The name of the tokenizer model to use."
    )
    parser.add_argument(
        "data", type=argparse.FileType(), help="The path to the JSON data file."
    )
    args = parser.parse_args()

    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="The sentencepiece tokenizer that you are converting to a fast tokenizer uses the byte fallback option which is not implemented in the fast tokenizers.",
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    data = load_data(args.data)
    max_length = calc_max_length(data, tokenizer)

    print(max_length)


if __name__ == "__main__":
    main()
