#!/usr/bin/env python
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

import typer
from sklearn.metrics import precision_recall_fscore_support


class Instance(TypedDict):
    id: str
    kind: str
    predictions: list[str]
    golds: list[str]
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


def calculate_metrics(
    predictions: list[MetricPrediction],
    references: list[MetricReference],
) -> dict[str, float]:
    instances: list[Instance] = []
    for pred, refer in zip(predictions, references):
        assert pred["id"] == refer["id"]

        pred_entities, pred_relation = parse_instance(pred["prediction_text"])
        ref_entities, ref_relation = parse_instance(refer["answers"])

        for itype in ref_entities:
            instance: Instance = {
                "id": refer["id"],
                "kind": itype,
                "predictions": pred_entities[itype],
                "golds": ref_entities[itype],
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
    extraction = compute_extraction_metrics(instances)
    classification = compute_classification_metrics(instances)
    return extraction | classification


def compute_classification_metrics(instances: list[Instance]) -> dict[str, float]:
    pred = [instance["pred_relation"] for instance in instances]
    gold = [instance["gold_relation"] for instance in instances]

    accuracy = sum(int(p == g) for p, g in zip(pred, gold)) / len(instances)
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


def compute_extraction_metrics(instances: list[Instance]) -> dict[str, float]:
    gold_lens = {"cause": 0, "effect": 0}
    pred_lens = {"cause": 0, "effect": 0}
    commons = {"cause": 0, "effect": 0}
    equal = {"cause": 0, "effect": 0}
    num_instances = {"cause": 0, "effect": 0}

    for instance in instances:
        kind = instance["kind"]
        pred_toks = get_tokens(" ".join(instance["predictions"]))
        gold_toks = get_tokens(" ".join(instance["golds"]))

        gold_lens[kind] += len(gold_toks)
        pred_lens[kind] += len(pred_toks)

        common = Counter(gold_toks) & Counter(pred_toks)
        commons[kind] += sum(common.values())

        equal[kind] += int(gold_toks == pred_toks)
        num_instances[kind] += 1

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for kind in gold_lens:
        if pred_lens[kind] != 0:
            precision = commons[kind] / pred_lens[kind]
        else:
            precision = 0

        recall = commons[kind] / gold_lens[kind]

        if precision + recall != 0:
            f1 = (2 * precision * recall) / (precision + recall)
        else:
            f1 = 0

        result[kind]["precision"] = precision
        result[kind]["recall"] = recall
        result[kind]["f1"] = f1
        result[kind]["em"] = equal[kind] / num_instances[kind]

    def macro_avg(metric: str) -> float:
        return sum(result[kind][metric] for kind in result) / len(result)

    return {metric: macro_avg(metric) for metric in ["precision", "recall", "f1", "em"]}


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
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())
    relation = relation.strip()

    return {
        "cause": causes,
        "effect": effects,
    }, relation


def main(infile: Path) -> None:
    """Expected format:

    - list of objects
    - each object has keys: input, gold, output
        - input: string. Text passage.
        - gold: string. Annotated answer (tag form).
        - output: string. Model prediction (tag form).
    """
    data = json.loads(infile.read_text())

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

    metrics = calculate_metrics(predictions, references)
    for key, val in metrics.items():
        print(f"{key}: {val:.2%}")


if __name__ == "__main__":
    typer.run(main)
