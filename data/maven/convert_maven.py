"""Convert the MAVEN dataset to the tagged extraction format."""

import argparse
import hashlib
import json
import os
import random
import warnings
from collections.abc import Sequence
from typing import Any, TextIO

from beartype.door import is_bearable

# Disable "None of PyTorch, TensorFlow >= 2.0, or Flax have been found." warning.
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import AutoTokenizer  # noqa: E402

HASH_KEYS = ("context", "answers")


def hash_instance(
    instance: dict[str, Any], keys: Sequence[str] = HASH_KEYS, length: int = 8
) -> str:
    key = "".join(str(instance[k]) for k in keys)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def process_data(
    data: list[dict[str, Any]],
    straight: bool,
    tokeniser_model: str,
    max_tokens: int,
    min_sentences: int,
    debug: bool,
) -> list[dict[str, Any]]:
    tokeniser = AutoTokenizer.from_pretrained(tokeniser_model)
    question = "What are the events?"
    question_type = "cause"

    if straight:
        answer_template = "{cause}"
    else:
        answer_template = "[Cause] {cause} [Relation] cause [Effect]"

    examples: list[dict[str, Any]] = []
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
            -1,
            *sorted({sentences.index(s) for s in reference_sentences}),
            len(sentences),
        ]
        n_pairs = len(indices) - 1

        for i in range(1, n_pairs):
            previous_sentence, current_sentence, next_sentence = indices[i - 1 : i + 2]

            start = previous_sentence + 1
            end = random.randint(current_sentence + 1, next_sentence)

            chosen_sentences = sentences[start:end]
            if len(chosen_sentences) < min_sentences:
                continue

            context = " ".join(chosen_sentences)
            assert context, "Clipped context is empty."

            # From seq2seq.preprocess_data, remove entries that are too long.
            source_text = f"{question}\n{context.lstrip()}"
            if len(tokeniser.tokenize(source_text)) > max_tokens:
                continue

            answer = answer_template.format(cause=sentences[current_sentence])

            instance = {
                "context": context,
                "question": question,
                "question_type": question_type,
                "answers": answer,
            }
            if debug:
                instance |= {
                    "original_context": sentences,
                    "sentences": chosen_sentences,
                    "sentence": sentences[current_sentence],
                }
            examples.append(instance | {"id": hash_instance(instance)})

    # Each example must have a unique context, otherwise we'd be expecting the model
    # to output different answers for the same context.
    assert len({x["context"] for x in examples}) == len(examples), "Duplicate contexts."
    return examples


def main(
    input_file: TextIO,
    output_file: TextIO,
    seed: int,
    straight: bool,
    tokeniser_model: str,
    max_tokens: int,
    min_sentences: int,
    debug: bool,
) -> None:
    random.seed(seed)
    # HuggingFace's warning about resume_download usage from its own library.
    warnings.filterwarnings("ignore", category=FutureWarning)

    data = [json.loads(line) for line in input_file]
    if not is_bearable(data, list[dict[str, Any]]):
        raise ValueError("Invalid input data format.")

    processed = process_data(
        data, straight, tokeniser_model, max_tokens, min_sentences, debug
    )

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
    parser.add_argument(
        "--tokeniser_model",
        type=str,
        default="google/flan-t5-large",
        help="Hugging Face tokeniser model.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens in the input.",
    )
    parser.add_argument(
        "--min_sentences",
        type=int,
        default=3,
        help="Minimum number of sentences in the context.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include debug information in the output.",
    )
    args = parser.parse_args()
    main(
        args.input_file,
        args.output_file,
        args.seed,
        args.straight,
        args.tokeniser_model,
        args.max_tokens,
        args.min_sentences,
        args.debug,
    )
