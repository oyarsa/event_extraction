import dataclasses
import logging
import warnings

import krippendorff
from scipy.stats import spearmanr, ConstantInputWarning
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)


@dataclasses.dataclass
class EvaluationResult:
    golds: list[int]
    preds: list[int]
    passages: list[str]
    outputs: list[str]
    annotations: list[str]
    loss: float
    tags: list[str] | None = None


def spearman(x: list[int], y: list[int]) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConstantInputWarning)
        return spearmanr(x, y)[0]


def calc_metrics(results: EvaluationResult) -> dict[str, float]:
    x, y = results.golds, results.preds

    acc = accuracy_score(x, y)
    prec, rec, f1, _ = precision_recall_fscore_support(
        x, y, average="binary", zero_division=0  # type: ignore
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "agreement": sum(a == b for a, b in zip(x, y)) / len(x),
        "krippendorff": krippendorff.alpha([x, y], level_of_measurement="nominal"),
        "spearman": spearman(x, y),
        "cohen": cohen_kappa_score(x, y),
        "eval_loss": results.loss,
    }


def report_metrics(
    logger: logging.Logger, metrics: dict[str, float], desc: str
) -> None:
    logger.info(
        f"{desc} results\n"
        f"    Accuracy      : {metrics['accuracy']:.4f}\n"
        f"    Precision     : {metrics['precision']:.4f}\n"
        f"    Recall        : {metrics['recall']:.4f}\n"
        f"    F1            : {metrics['f1']:.4f}\n"
        f"    Agreement     : {metrics['agreement']:.4f}\n"
        f"    Krippendorff  : {metrics['krippendorff']:.4f}\n"
        f"    Spearman      : {metrics['spearman']:.4f}\n"
        f"    Cohen         : {metrics['cohen']:.4f}\n"
        f"    Eval Loss     : {metrics['eval_loss']:.4f}\n"
    )
