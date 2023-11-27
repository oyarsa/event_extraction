#!/usr/bin/env python
# pyright: basic
import contextlib
import io
import json
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

import evaluate
import typer
from sklearn.metrics import precision_recall_fscore_support

from self_critique.minimal.util import suppress_transformers_warnings


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
    predictions: list[MetricPrediction],
    references: list[MetricReference],
    use_bertscore: bool,
    use_bleurt: bool,
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

    standard = compute_metrics(instances)
    rougel_separate = calculate_rouge_separate(golds, preds, clause_types)
    rougel_tagged = calculate_rouge_tagged(tagged)
    bleu = calculate_bleu(golds, preds, clause_types)

    metrics = standard | rougel_separate | rougel_tagged | bleu

    if use_bertscore:
        bertscore = calculate_bertscore(golds, preds, clause_types)
        metrics = metrics | bertscore

    if use_bleurt:
        bleurt = calculate_bleurt(golds, preds, clause_types)
        metrics = metrics | bleurt

    return metrics


def calculate_rouge_tagged(tagged: dict[str, list[str]]) -> dict[str, float]:
    rouge = evaluate.load("rouge")
    return {
        "rougel_tag": rouge.compute(
            predictions=tagged["pred"], references=tagged["gold"]
        )["rougeL"]
    }


def calculate_rouge_separate(
    golds: dict[str, list[str]], preds: dict[str, list[str]], clause_types: list[str]
) -> dict[str, float]:
    rouge = evaluate.load("rouge")

    results: dict[str, float] = {
        itype: rouge.compute(predictions=preds[itype], references=golds[itype])[
            "rougeL"
        ]
        for itype in clause_types
    }
    return {"rougel_sep": statistics.mean(results.values())}


def calculate_bleu(
    golds: dict[str, list[str]], preds: dict[str, list[str]], clause_types: list[str]
) -> dict[str, float]:
    bleu = evaluate.load("bleu")

    # BLEU takes a list of reference "translations" for every instance, so we wrap
    # each gold in a list.
    results: dict[str, float] = {
        itype: bleu.compute(
            predictions=preds[itype], references=[[g] for g in golds[itype]]
        )["bleu"]
        for itype in clause_types
    }
    return {"bleu": statistics.mean(results.values())}


def calculate_bleurt(
    golds: dict[str, list[str]], preds: dict[str, list[str]], clause_types: list[str]
) -> dict[str, float]:
    try:
        bleurt = evaluate.load("bleurt")
    except ImportError as e:
        raise ImportError(
            "BLEURT is not installed. Install from the repository: "
            "https://github.com/google-research/bleurt/tree/master"
        ) from e

    # BLEURT returns a list of scores for every instance, so we take the mean.
    results: dict[str, float] = {
        itype: statistics.mean(
            bleurt.compute(predictions=preds[itype], references=golds[itype])["scores"]
        )
        for itype in clause_types
    }
    return {"bleurt": statistics.mean(results.values())}


def compute_bertscore(
    golds: dict[str, list[str]], preds: dict[str, list[str]], clause_types: list[str]
) -> dict[str, dict[str, list[float]]]:
    """
    bert_score (underlying evaluate) prints a warning to stderr, so we suppress it
    the warning is when a sentence is empty, which is fine
    """
    suppress_transformers_warnings()
    bertscore = evaluate.load("bertscore")
    with contextlib.redirect_stderr(io.StringIO()):
        return {
            itype: bertscore.compute(
                predictions=preds[itype], references=golds[itype], lang="en"
            )
            for itype in clause_types
        }


def calculate_bertscore(
    golds: dict[str, list[str]], preds: dict[str, list[str]], clause_types: list[str]
) -> dict[str, float]:
    metrics = ["precision", "recall", "f1"]
    results = compute_bertscore(golds, preds, clause_types)

    results_agg_itype = {
        itype: {metric: statistics.mean(results[itype][metric]) for metric in metrics}
        for itype in clause_types
    }

    results_agg_macro_avg = {
        metric: statistics.mean(
            results_agg_itype[itype][metric] for itype in clause_types
        )
        for metric in metrics
    }

    return {
        f"bertscore_{metric}": results_agg_macro_avg[metric]
        for metric in results_agg_macro_avg
    }


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


def main(infiles: list[Path], bertscore: bool = True, bleurt: bool = True) -> None:
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

    Prints metrics to stdout and saves to a file with the same name as the input file
    but with the extension `.metrics.json`.
    """
    for file in infiles:
        print(">>>", file)
        run_file_metrics(file, bertscore, bleurt)
        print()


def run_file_metrics(infile: Path, use_bertscore: bool, use_bleurt: bool) -> None:
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

    metrics = calculate_metrics(
        predictions, references, use_bertscore=use_bertscore, use_bleurt=use_bleurt
    )
    for key, val in metrics.items():
        print(f"{key}: {val:.2%}")

    metrics_file = infile.with_suffix(".metrics.json")
    metrics_file.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    typer.run(main)
