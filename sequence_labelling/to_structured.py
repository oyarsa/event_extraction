import json
from pathlib import Path
from typing import Any, Optional

import typer
from nltk.tokenize.treebank import TreebankWordDetokenizer


def tag_type(tag: str) -> str:
    return tag[2:] if tag.startswith("B-") or tag.startswith("I-") else tag


def find_tag_intervals(
    tokens: list[str], tags: list[str], reconstructed_string: str
) -> list[tuple[int, int, str]]:
    intervals = []
    start_index = 0
    current_tag = "<s>"

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        prev_tag_same_type = tag.startswith("I-") and (
            i == 0 or tag_type(tags[i - 1]) != tag_type(tag)
        )
        if tag.startswith("B-") or (prev_tag_same_type):
            if start_index != 0:
                end_index = reconstructed_string.find(token, start_index)
                intervals.append((start_index, end_index - 1, current_tag))
            start_index = reconstructed_string.find(token, start_index)
            current_tag = tag[2:]

        elif tag == "O" and start_index != 0:
            end_index = reconstructed_string.find(token, start_index)
            intervals.append((start_index, end_index - 1, current_tag))
            start_index = 0

    if start_index != 0:
        intervals.append((start_index, len(reconstructed_string), current_tag))

    return intervals


def intervals_to_genqa(text: str, intervals: list[tuple[int, int, str]]) -> str:
    causes: list[str] = []
    effects: list[str] = []
    for start, end, tag in intervals:
        if tag == "Cause":
            causes.append(text[start : end + 1])
        elif tag == "Effect":
            effects.append(text[start : end + 1])

    cause = " | ".join(sorted(causes))
    effect = " | ".join(sorted(effects))
    return f"[Cause] {cause} [Relation] cause [Effect] {effect}"


def main(
    sequence_path: Path,
    gen_path: Path,
    output_path: Path,
    n: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Convert sequence data to GenQA format.

    For the sequence labelling format, see preprocess/sequence_labelling.py
    For the GenQA format, see preprocess/genqa_joint.py

    This needs the original GenQA data to get the original gold answer. Unfortunately,
    the sequence labelling dataset does not have all the contents of the original
    dataset. For example, for dev set there are 2482 items, but the sequence labelling
    one only has 2147. And out of these, only 2037 match due to some issues with the
    (de)tokenisation.

    Anyway, these numbers aren't too big to warrant redoing the sequence labelling
    dataset and retraining everything, so we'll just have to live with it.
    """
    sequence_data = sequence_path.read_text().strip().split("\n\n")[:n]
    gen_data = json.loads(gen_path.read_text())["data"][:n]

    print(f"Match {len(sequence_data)} blocks with {len(gen_data)} generated data")

    detokenizer = TreebankWordDetokenizer()
    result: list[dict[str, Any]] = []

    for block in sequence_data:
        lines = block.split("\n")
        tokens = [line.split(" ")[0] for line in lines]
        gold_tags = [line.split(" ")[1] for line in lines]

        reconstructed_string = detokenizer.detokenize(tokens)

        gen_gold: str | None = None
        for gen in gen_data:
            if gen["context"] == reconstructed_string:
                gen_gold = gen["answers"]
                break
        else:
            continue
        assert gen_gold

        tag_intervals = find_tag_intervals(tokens, gold_tags, reconstructed_string)
        struct_output = intervals_to_genqa(reconstructed_string, tag_intervals)

        pred_tags = [line.split(" ")[2] for line in lines]
        pred_tag_intervals = find_tag_intervals(tokens, pred_tags, reconstructed_string)
        pred_struct_output = intervals_to_genqa(
            reconstructed_string, pred_tag_intervals
        )

        if verbose:
            print(f"Original:\n\n{block}\n\n")
            print(f"Reconstructed: {reconstructed_string}")
            print(f"Tag Intervals: {tag_intervals}\n")
            print(f"GenQA: {gen_gold}\n")
            print(f"Seq as GenQA: {struct_output}\n")
            print(f"Pred Tag Intervals: {pred_tag_intervals}\n")
            print(f"Pred GenQA: {pred_struct_output}\n")

        result.append(
            {
                "gold": gen_gold,
                "input": reconstructed_string,
                "output": pred_struct_output,
            }
        )

    print(f"Matched {len(result)} out of {len(sequence_data)}")
    output_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    typer.run(main)
