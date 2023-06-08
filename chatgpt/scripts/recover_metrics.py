import json
import sys
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support


def get_relation(s: str) -> str:  # sourcery skip: use-next
    for line in s.splitlines():
        if line.lower().strip().startswith("relation:"):
            return line.split(":")[1].strip()
    return ""


def clean_relation(s: str) -> str:
    return s if s in {"cause", "enable", "prevent"} else "other"


def compute_classification_metrics(
    pred: list[str], gold: list[str]
) -> dict[str, float]:
    accuracy = sum(int(p == g) for p, g in zip(pred, gold)) / len(pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold,
        pred,
        average="macro",
        zero_division=0,  # type: ignore
    )

    return {
        "cls_accuracy": accuracy,
        "cls_precision": float(precision),
        "cls_recall": float(recall),
        "cls_f1": float(f1),
    }


def get_relations(data: list[dict[str, str]], field: str) -> list[str]:
    return [clean_relation(get_relation(d[field])) for d in data]


def main() -> None:
    data = json.loads(Path(sys.argv[1]).read_text())
    gold = get_relations(data, "answer")
    pred = get_relations(data, "pred")
    metrics = compute_classification_metrics(pred, gold)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
