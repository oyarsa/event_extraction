#!/usr/bin/env python3
"""Convert tagged data to sequence labelled data.

Input format: JSON file with array of objects, each the following fields:
- context: str
- question: str
- question_type: str
- answers: str
- id: str

`answers` represents an extraction from the text, with the following format:
"[Cause] <cause text> [Relation] <cause|enable|prevent> [Effect] <effect text>"

Output format: a text file where each line is a space-separated pair of token and tag.
Example:

If O
one B-Cause
or I-Cause

Each item is in a block, with a blank line separating items.
"""

import argparse
import json
import re
from typing import TextIO

import nltk


def find_longest_subsequence_indices(
    haystack: list[str], needle: list[str]
) -> tuple[int, int] | None:  # sourcery skip: use-itertools-product, use-next
    """Find the (start, end) indices of the longest subsequence.

    The indices are for haystack. If the subsequence is not found, return None.
    The indices are exclusive of the end index, [start, end)
    """
    dp = [[0] * (len(needle) + 1) for _ in range(len(haystack) + 1)]
    for h_idx in range(1, len(haystack) + 1):
        for n_idx in range(1, len(needle) + 1):
            if haystack[h_idx - 1] == needle[n_idx - 1]:
                dp[h_idx][n_idx] = dp[h_idx - 1][n_idx - 1] + 1
            else:
                dp[h_idx][n_idx] = max(dp[h_idx - 1][n_idx], dp[h_idx][n_idx - 1])

    # Backtrack to find the LCS
    h_idx = len(haystack)
    n_idx = len(needle)
    wip_lcs: list[str] = []

    while h_idx > 0 and n_idx > 0:
        if haystack[h_idx - 1] == needle[n_idx - 1]:
            wip_lcs.append(haystack[h_idx - 1])
            h_idx -= 1
            n_idx -= 1
        elif dp[h_idx - 1][n_idx] > dp[h_idx][n_idx - 1]:
            h_idx -= 1
        else:
            n_idx -= 1

    lcs = list(reversed(wip_lcs))
    lcs_len = len(lcs)

    # Find the start and end indices of the LCS in haystack
    start_index = None
    for idx in range(len(haystack) - lcs_len + 1):
        if haystack[idx : idx + lcs_len] == lcs:
            start_index = idx
            break

    if start_index is None:
        return None

    return start_index, start_index + lcs_len


def parse_instance(answer: str) -> tuple[str, str]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\].*?\[Effect\](.*?)$", answer)
    if not matches:
        return "", ""

    causes, effects = matches[0]
    causes = sorted(causes.split("|"))
    effects = sorted(effects.split("|"))

    if not causes or not effects:
        return "", ""
    return causes[0].strip(), effects[0].strip()


def tag_to_seqlabel(item: dict[str, str]) -> list[tuple[str, str]]:
    # sourcery skip: extract-duplicate-method
    """Transform a tagged item to sequence labelled data.

    Each token in the context is tagged with respect to whether it is part of the cause
    or effect.

    Returns:
        A list of (token, tag) tuples.
    """
    cause, effect = parse_instance(item["answers"])

    context_tokens = nltk.word_tokenize(item["context"])
    cause_tokens = nltk.word_tokenize(cause)
    effect_tokens = nltk.word_tokenize(effect)

    tags = ["O"] * len(context_tokens)

    if cause_span := find_longest_subsequence_indices(context_tokens, cause_tokens):
        cause_start, cause_end = cause_span
        tags[cause_start] = "B-Cause"
        for i in range(cause_start + 1, cause_end):
            tags[i] = "I-Cause"

    if effect_span := find_longest_subsequence_indices(context_tokens, effect_tokens):
        effect_start, effect_end = effect_span
        tags[effect_start] = "B-Effect"
        for i in range(effect_start + 1, effect_end):
            tags[i] = "I-Effect"

    return list(zip(context_tokens, tags))


def main(input_file: TextIO, output_file: TextIO) -> None:
    tag_data = json.load(input_file)
    if "data" in tag_data:
        tag_data = tag_data["data"]
    seqlabel_data = [tag_to_seqlabel(obj) for obj in tag_data]

    out_items: list[str] = []
    for item in seqlabel_data:
        lines = [f"{token} {tag}" for token, tag in item]
        out_items.append("\n".join(lines))

    output = "\n\n".join(out_items)
    output_file.write(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=argparse.FileType("r"),
        help="Input JSON file with tagged data",
    )
    parser.add_argument(
        "output_file",
        type=argparse.FileType("w"),
        help="Output text file with sequence labelled data",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
