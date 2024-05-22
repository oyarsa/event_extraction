#!/usr/bin/env python
# pyright: basic
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional, TypedDict

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


def rewrite_clause(parts: list[str]) -> str:
    return ", and".join(parts)


def calculate_metrics(
    predictions: list[MetricPrediction], references: list[MetricReference]
) -> dict[str, float]:
    clause_types = ["cause", "effect"]

    instances: list[Instance] = []
    golds: dict[str, list[str]] = defaultdict(list)
    preds: dict[str, list[str]] = defaultdict(list)
    tagged: dict[str, list[str]] = defaultdict(list)

    for pred, refer in zip(predictions, references):
        assert pred["id"] == refer["id"], "Prediction and reference IDs do not match"

        tagged["pred"].append(pred["prediction_text"])
        tagged["gold"].append(refer["answers"])

        pred_entities, pred_relation = parse_instance(pred["prediction_text"])
        if not pred_relation:
            continue

        ref_entities, ref_relation = parse_instance(refer["answers"])

        for itype in clause_types:
            instance: Instance = {
                "id": refer["id"],
                "kind": itype,
                "predictions": pred_entities[itype],
                "golds": ref_entities[itype],
                "pred_relation": pred_relation,
                "gold_relation": ref_relation,
            }
            instances.append(instance)

            golds[itype].append(rewrite_clause(ref_entities[itype]))
            preds[itype].append(rewrite_clause(pred_entities[itype]))

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

    accuracy = sum(p == g for p, g in zip(pred, gold)) / len(instances)
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


def clean(s: str) -> str:
    s = s.strip()
    return re.sub(r'""+"', '"', s)


def parse_instance(answer: str) -> tuple[dict[str, list[str]], str | None]:
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
        }, None

    causes, relation, effects = matches[0]
    causes = sorted(s for c in causes.split("|") if (s := clean(c)))
    effects = sorted(s for e in effects.split("|") if (s := clean(e)))

    relation = relation.strip().lower()
    if relation not in {"cause", "enable", "prevent"}:
        relation = None

    return {
        "cause": causes,
        "effect": effects,
    }, relation


def main(infiles: list[Path], tag: Optional[Path] = None, save: bool = True) -> None:
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
        if tag is None:
            outfile = infile.with_suffix(".metrics.json")
        else:
            outfile = infile.with_suffix(f".{tag}.metrics.json")

        run_file_metrics(
            infile,
            outfile,
            save_results=save,
        )


def run_file_metrics(
    infile: Path,
    outfile: Path,
    save_results: bool,
) -> None:
    "Run metrics on a single file."
    print(">>>", infile)

    metrics = get_file_metrics(infile)

    if save_results:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_text(json.dumps(metrics, indent=2))

    for key, val in metrics.items():
        print(f"{key}: {val:.2%}")

    print()


def get_file_metrics(infile: Path) -> dict[str, float]:
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

    return calculate_metrics(predictions, references)


if __name__ == "__main__":
    typer.run(main)
