import json
from pathlib import Path

import typer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def main(original_path: Path, result_path: Path) -> None:
    original = json.loads(original_path.read_text())
    result = json.loads(result_path.read_text())

    gold: list[bool] = []
    pred: list[bool] = []

    for og in original:
        for res in result:
            if og["passage"] == res["input"] and og["annotation"] == res["gold"]:
                gold.append(bool(og["gold"]))
                pred.append(res["valid"])
                break

    assert len(gold) == len(pred), "Length of gold and pred must be same."
    assert len(gold) == len(original), "Length of gold and original must be same."

    acc = accuracy_score(gold, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0  # type: ignore
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":
    typer.run(main)
