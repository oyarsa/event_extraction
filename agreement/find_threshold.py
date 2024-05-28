"""Find the best threshold for a binary classifier.

Given a JSON file with gold labels and prediction scores, this script finds the
threshold that maximizes agreement with the gold labels. The JSON file should
be a list of dictionaries, each with the specified `gold` and `pred` keys.
"""

import argparse
import json
from typing import TextIO


def calculate_agreement(golds: list[int], preds: list[int]) -> float:
    assert len(golds) == len(preds)
    return sum(x == y for x, y in zip(golds, preds)) / len(golds)


def find_best_threshold(
    golds: list[int], pred_scores: list[float]
) -> tuple[float, float]:
    assert len(golds) == len(pred_scores)

    best_threshold = 0
    best_agreement = 0

    for threshold in pred_scores:
        preds = [int(score >= threshold) for score in pred_scores]
        agreement = calculate_agreement(golds, preds)
        if agreement > best_agreement:
            best_agreement = agreement
            best_threshold = threshold

    return best_threshold, best_agreement


def main(input: TextIO, gold_key: str, pred_key: str) -> None:
    data = json.load(input)

    golds = [d[gold_key] for d in data]
    pred_scores = [d[pred_key] for d in data]

    threshold, agreement = find_best_threshold(golds, pred_scores)

    print(f"Best threshold: {threshold}")
    print(f"Agreement with it: {agreement}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=argparse.FileType("r"), help="Input JSON file")
    parser.add_argument(
        "--gold",
        type=str,
        default="gold",
        help="Key in the JSON for the gold label (1 or 0)",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default="pred",
        help="Key in the JSON for the prediction score (float)",
    )
    args = parser.parse_args()

    main(args.input, args.gold, args.pred)
