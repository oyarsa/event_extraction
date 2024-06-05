#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path

import numpy as np
import typer
from sklearn.metrics import confusion_matrix


def class_acc(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    true_counts = Counter(y_true)
    class_accuracies: dict[str, float] = {}

    for label, true_count in true_counts.items():
        correct_count = sum(
            true_label == label and pred_label == label
            for true_label, pred_label in zip(y_true, y_pred)
        )
        class_accuracies[label] = correct_count / true_count

    return class_accuracies


def print_confusion(cm: np.ndarray, labels: list[str]) -> None:
    label_width = max(len(label) for label in labels)

    # Print header row with class labels
    header = "  ".join(label.ljust(label_width) for label in labels)
    top_left_corner = "T\\P".center(label_width)
    print(f"{top_left_corner}  {header}")

    # Print each row of the confusion matrix
    for i, row in enumerate(cm):
        row_render = "  ".join(str(val).ljust(label_width) for val in row)
        print(f"{labels[i].ljust(label_width)}  {row_render}")


def print_acc(accuracies: dict[str, float]) -> None:
    print("\nCLASS ACCURACY")
    label_width = max(len(label) for label in accuracies) + 1
    for label, rate in accuracies.items():
        print(f"{label.ljust(label_width)}: {rate:.2%}")


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False
)


@app.command()
def main(file: Path) -> None:
    """Print confusion matrix and class accuracies from a JSON file.

    Usage: python scripts/confusion_matrix.py FILE

    FILE should be a JSON file containing a list of objects with "gold" and "prediction"
    keys. The values should be strings representing class labels.
    """
    data = json.loads(file.read_text())

    y_true = [str(d["gold"]) for d in data]
    y_pred = [str(d["prediction"] if "prediction" in d else d["pred"]) for d in data]

    accuracy = sum(true == pred for true, pred in zip(y_true, y_pred)) / len(y_true)
    print(f"Overall accuracy: {accuracy:.2%}\n")

    labels = list(set(y_true + y_pred))
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    print_confusion(confusion, labels)

    acc = class_acc(y_true, y_pred)
    print_acc(acc)


if __name__ == "__main__":
    app()
