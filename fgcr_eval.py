from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sklearn import metrics


@dataclass
class Metric:
    precision: float
    recall: float
    f1: float
    em: float

    def __str__(self) -> str:
        return (
            f"Precision: {self.precision:.2%}\n"
            f"Recall:    {self.recall:.2%}\n"
            f"F1:        {self.f1:.2%}\n"
            f"EM:        {self.em:.2%}"
        )


def evaluate(gold: list[list[str]], predicted: list[list[str]]) -> Metric:
    """Evaluate labels for F1 and Exact Match

    Args:
        gold (list[list[str]]): gold instances, each a lists of gold labels
        predicted (list[list[str]]):
           predicted instances, each a list of predicted labels

    Returns:
        Metric: set of metrics we track
    """
    exact_match = 0
    y_gold: list[str] = []
    y_predicted: list[str] = []

    for g, p in zip(gold, predicted):
        exact_match += all(x == y for x, y in zip(g, p))
        y_gold.extend(g)
        y_predicted.extend(p)

    p, r, f1, _ = metrics.precision_recall_fscore_support(
        y_gold, y_predicted, average="macro"
    )
    em = exact_match / len(gold)

    return Metric(
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        em=em,
    )


@dataclass
class Entry:
    token: str
    gold: str
    pred: str


def evaluate_conll(path: Path) -> Metric:
    sentences: list[list[Entry]] = []
    sentence: list[Entry] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                sentences.append(sentence)
                sentence = []
            else:
                token, gold, pred, *_ = line.split()
                sentence.append(Entry(token, gold, pred))

    golds = [[e.gold for e in sent] for sent in sentences]
    preds = [[e.pred for e in sent] for sent in sentences]

    result = evaluate(golds, preds)
    return result


evals = {
    "conll": evaluate_conll,
}


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("format", help=f"Format to evaluate. One of {list(evals)}.")
    argparser.add_argument("path", help="Path to predictions file or folder.")

    args = argparser.parse_args()
    format = args.format.lower().strip()
    path = Path(args.path)

    if format not in evals:
        raise ValueError("Invalid model: " + args.format)

    print(evals[format](path))


if __name__ == "__main__":
    main()
