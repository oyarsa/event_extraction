#!/usr/bin/env python3

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def clean(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r'""+"', '"', s)


def parse_instance(answer: str) -> tuple[str, str]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\].*?\[Effect\](.*?)$", answer)
    if not matches:
        return "", ""

    causes, effects = matches[0]
    causes = sorted(s for c in causes.split("|") if (s := clean(c)))
    effects = sorted(s for e in effects.split("|") if (s := clean(e)))

    if not causes or not effects:
        return "", ""
    return causes[0], effects[0]


def precision_recall_f1(
    pred: list[str], input: list[str]
) -> tuple[float, float, float]:
    """Calculate precision between the prediction and the input context.

    This is done by counting the number of common elements between the prediction
    and dividing by the number of tokens in the prediction.

    The goal is to measure how much of the prediction is actually in the input. The
    intention is that this should always be 100%. If it isn't, it means that the model
    is hallucinating tokens.
    """
    if not pred:
        return 0, 0, 0

    pred_count = Counter(pred)
    input_count = Counter(input)
    common_count = pred_count & input_count
    common = sum((common_count).values())

    precision = common / len(pred)
    recall = common / len(input)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def find_longest_subsequence(a: list[str], b: list[str]) -> list[str]:
    """Find the longest common contiguous subsequence between two lists."""
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    max_length = 0
    end_index = 0

    # sourcery skip: use-itertools-product
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i

    start_index = end_index - max_length
    return a[start_index:end_index]


def longest_subsequence_ratio(pred: list[str], input: list[str]) -> float:
    """Find the ratio between the longest subsequence and the length of the prediction.

    We want to find the longest sequence of words between the two sentences. In a
    perfect world, the longest subsequence should be the entirety of the shortest
    sentence. If it isn't, the model is hallucinating tokens.

    If the prediction is empty, the ratio is 0.
    """
    if not pred:
        return 0

    longest = find_longest_subsequence(pred, input)
    return len(longest) / len(pred)


def calc_metrics(data: list[dict[str, str]]) -> dict[str, float]:
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    total_subseq = 0

    for item in data:
        causes, effects = parse_instance(item["output"])
        causes_toks = causes.split()
        effects_toks = effects.split()
        input_toks = item["input"].split()

        cause_p, cause_r, cause_f1 = precision_recall_f1(causes_toks, input_toks)
        effect_p, effect_r, effect_f1 = precision_recall_f1(effects_toks, input_toks)
        total_recall += (cause_r + effect_r) / 2
        total_precision += (cause_p + effect_p) / 2
        total_f1 += (cause_f1 + effect_f1) / 2

        cause_subseq = longest_subsequence_ratio(causes_toks, input_toks)
        effect_subseq = longest_subsequence_ratio(effects_toks, input_toks)
        total_subseq += (cause_subseq + effect_subseq) / 2

    return {
        "precision": total_precision / len(data),
        "recall": total_recall / len(data),
        "f1": total_f1 / len(data),
        "longest_subsequence": total_subseq / len(data),
    }


def main(infile: Path, outfile: Path | None) -> None:
    data: list[dict[str, str]] = json.loads(infile.read_text())

    out = calc_metrics(data)
    for k, v in out.items():
        print(f"{k}: {v:.4f}")

    if outfile is None:
        outfile = infile.with_stem(f"{infile.stem}_overlap")
    with open(outfile, "w") as f:
        json.dump({"model": infile.stem} | out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=Path, help="Input file")
    parser.add_argument(
        "outfile",
        type=Path,
        nargs="?",
        help="Output file. Defaults to the input file with '_overlap' appended.",
    )
    args = parser.parse_args()
    main(args.infile, args.outfile)
