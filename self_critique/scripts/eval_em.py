#!/usr/bin/env python
# pyright: basic
# ruff: noqa: E501
import json
import re
import string
from pathlib import Path
from typing import Any, TypedDict

import typer


class Instance(TypedDict):
    id: str
    kind: str
    cause_predictions: list[str]
    cause_golds: list[str]
    effect_predictions: list[str]
    effect_golds: list[str]
    pred_relation: str | None
    gold_relation: str | None


class MetricPrediction(TypedDict):
    id: str
    prediction_text: str


class MetricReference(TypedDict):
    id: str
    answers: str
    context: str


def get_id(d: dict[str, str]) -> str:
    return str(hash(d["input"] + d["gold"] + d["output"]))


def rewrite_clause(parts: list[str]) -> str:
    return ", and".join(parts)


def calculate_metrics(
    predictions: list[MetricPrediction], references: list[MetricReference]
) -> dict[str, float]:
    instances: list[Instance] = []

    for pred, refer in zip(predictions, references):
        assert pred["id"] == refer["id"], "Prediction and reference IDs do not match"

        pred_entities, pred_relation = parse_instance(pred["prediction_text"])
        ref_entities, ref_relation = parse_instance(refer["answers"])

        instance: Instance = {
            "id": refer["id"],
            "kind": "cause",
            "cause_predictions": pred_entities["cause"],
            "cause_golds": ref_entities["cause"],
            "effect_predictions": pred_entities["effect"],
            "effect_golds": ref_entities["effect"],
            "pred_relation": pred_relation,
            "gold_relation": ref_relation,
        }
        instances.append(instance)

    return compute_metrics(instances)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.casefold()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def compute_metrics(instances: list[Instance]) -> dict[str, float]:
    cause_equal = 0
    effect_equal = 0
    both_equal = 0

    for instance in instances:
        cause_pred_toks = get_tokens(" ".join(instance["cause_predictions"]))
        cause_gold_toks = get_tokens(" ".join(instance["cause_golds"]))
        effect_pred_toks = get_tokens(" ".join(instance["effect_predictions"]))
        effect_gold_toks = get_tokens(" ".join(instance["effect_golds"]))

        cause_equal += int(cause_gold_toks == cause_pred_toks)
        effect_equal += int(effect_gold_toks == effect_pred_toks)
        both_equal += int(
            cause_gold_toks == cause_pred_toks and effect_gold_toks == effect_pred_toks
        )

    return {
        "both": both_equal / len(instances),
        "cause": cause_equal / len(instances),
        "effect": effect_equal / len(instances),
    }


def clean(s: str) -> str:
    s = s.strip()
    return re.sub(r'""+"', '"', s)


def parse_instance(answer: str) -> tuple[dict[str, list[str]], str]:
    """Parse string answer to separate into class and spans
    Simple case:
    [Cause] This is a cause [Relation] cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Relation] enable [Effect] This effect 1 | This effect 2
    """
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return {
            "cause": [],
            "effect": [],
        }, "cause"

    causes, relation, effects = matches[0]
    causes = sorted(s for c in causes.split("|") if (s := clean(c)))
    effects = sorted(s for e in effects.split("|") if (s := clean(e)))
    relation = relation.strip().lower()

    return {
        "cause": causes,
        "effect": effects,
    }, relation


def main(infiles: list[Path]) -> None:
    """
    Expected input data format for each file is a JSON with the following structure:

    \b
    [
        {
            "input": "This is a text passage.",
            "gold": "[Cause] ... [Relation] cause [Effect] ...",
            "output": "[Cause] ... [Relation] cause [Effect] ...",
        },
        ...
    ]

    Where `gold` is the annotation and `output` is the model output.

    Prints metrics to stdout and saves to `outfile`. By default, `outfile` is the same
    name as the input file but with the extension `.metrics.json`.
    """
    for infile in infiles:
        run_file_metrics(infile)


def run_file_metrics(infile: Path) -> None:
    "Run metrics on a single file."
    print(">>>", infile)

    data = json.loads(infile.read_text())
    metrics = get_data_metrics(data)

    for key, val in metrics.items():
        print(f"{key}: {val:.2%}")

    print()


def get_data_metrics(data: list[dict[str, Any]]) -> dict[str, float]:
    predictions: list[MetricPrediction] = [
        {
            "id": get_id(d),
            "prediction_text": d["output"],
        }
        for d in data
    ]
    references: list[MetricReference] = [
        {
            "id": get_id(d),
            "answers": d["gold"],
            "context": d["input"],
        }
        for d in data
    ]

    return calculate_metrics(predictions, references)


if __name__ == "__main__":
    typer.run(main)
