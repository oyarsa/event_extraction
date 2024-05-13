"""Filter data file to only include instances that need manual annotation.

Cases that can be automatically annotated using rules are removed from the dataset.
"""

import argparse
import json
import random
import string
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

from evaluation.gpt_common import parse_instance as parse_instance_


def clean_str(s: str) -> str:
    """Remove punctuation and convert to lowercase."""
    punct_to_space = str.maketrans(string.punctuation, " " * len(string.punctuation))
    return s.translate(punct_to_space).casefold()


def contains_subsequence(subseq: list[str], lst: list[str]) -> bool:
    """Check if subseq is a contiguous subsequence of lst."""
    if not subseq or not lst:
        return False
    return any(
        subseq == lst[i : i + len(subseq)] for i in range(len(lst) - len(subseq) + 1)
    )


def find_any_common_subsequence(
    lst1: list[str], lst2: list[str], min_length: int
) -> bool:
    """Check if any contiguous subsequence of any length exists between lst1 and lst2."""
    len1 = len(lst1)
    len2 = len(lst2)

    return any(
        contains_subsequence(lst1[start : start + length], lst2)
        for length in range(min_length, min(len1, len2) + 1)
        for start in range(len1 - length + 1)
    )


def do_sequences_intersect(str1: str, str2: str, min_length: int) -> bool:
    """Check if there is a common sequence of words between two strings in order.

    Subsequences of length less than min_length are ignored.
    """
    words1 = clean_str(str1).split()
    words2 = clean_str(str2).split()

    return find_any_common_subsequence(
        words1, words2, min_length
    ) or find_any_common_subsequence(words2, words1, min_length)


class Tag(Enum):
    EMPTY = "empty"
    EXACT_MATCH = "exact_match"
    NEEDS_ANNOTATION = "needs_annotation"
    NO_INTERSECTION = "no_intersection"


def get_tag(reference: str, model: str, min_subseq_length: int) -> Tag:
    """Check if the output needs manual annotation or can be done by rules alone."""
    cause_gold, effect_gold = parse_instance(reference)
    cause_pred, effect_pred = parse_instance(model)

    # All clauses are empty means it's invalid.
    if not cause_pred or not effect_pred or not cause_gold or not effect_gold:
        return Tag.EMPTY

    # Exact match means it's valid.
    if cause_pred == cause_gold and effect_pred == effect_gold:
        return Tag.EXACT_MATCH

    # No subsequence intersection between gold and predicted means it's invalid.
    if not do_sequences_intersect(
        cause_gold, cause_pred, min_subseq_length
    ) or not do_sequences_intersect(effect_gold, effect_pred, min_subseq_length):
        return Tag.NO_INTERSECTION

    return Tag.NEEDS_ANNOTATION


def parse_instance(text: str) -> tuple[str | None, str | None]:
    inst, _ = parse_instance_(text)
    cause, effect = inst["Cause"], inst["Effect"]
    if cause and effect:
        return cause[0], effect[0]
    return None, None


def tag(data: list[dict[str, Any]], min_subseq_length: int) -> list[dict[str, Any]]:
    return [
        item
        | {"tag": get_tag(item["reference"], item["model"], min_subseq_length).value}
        for item in data
    ]


def load_and_reshape_data(input: Path) -> list[dict[str, Any]]:
    """Load list of objects from a JSON file and reshape the dictionary keys."""
    return [
        {
            "text": item["input"],
            "reference": item["gold"],
            "model": item["output"],
        }
        for item in json.loads(input.read_text())
    ]


def calc_tag_ratios(
    data: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    tags = Counter(item["tag"] for item in data)
    total = sum(tags.values())
    return {
        tag: {"count": count, "ratio": count / total} for tag, count in tags.items()
    }


def show_tag_results(ratios: dict[str, dict[str, Any]]) -> str:
    return "Tag results:\n" + "\n".join(
        f"  {tag}: {d['count']} ({d['ratio']:.1%})" for tag, d in ratios.items()
    )


def remove_auto_tagged(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in data:
        if item["tag"] == Tag.NEEDS_ANNOTATION.value:
            item_ = item.copy()
            del item_["tag"]
            out.append(item_)
    return out


def main(
    input_file: Path,
    output_dir: Path,
    seed: int,
    min_subseq_length: int,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_reshape_data(input_file)
    tagged_data = tag(data, min_subseq_length)

    tag_ratios = calc_tag_ratios(tagged_data)
    print(show_tag_results(tag_ratios), end="\n\n")

    to_annotate = remove_auto_tagged(tagged_data)

    (output_dir / "tag_ratios.json").write_text(json.dumps(tag_ratios))
    (output_dir / "tagged.json").write_text(json.dumps(tagged_data))
    (output_dir / "to_annotate.json").write_text(json.dumps(to_annotate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_file", type=Path, help="Input JSON file")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory to save the split datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--min-subseq-length",
        type=int,
        default=1,
        help="Minimum subsequence length for intersection check",
    )
    args = parser.parse_args()

    main(
        args.input_file,
        args.output_dir,
        args.seed,
        args.min_subseq_length,
    )
