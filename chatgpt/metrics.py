# pyright: basic
import re
import string
from collections import Counter, defaultdict
from enum import Enum
from typing import TypedDict

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
    question_type: str
    context: str


class StructureFormat(str, Enum):
    TAGS = "tags"
    LINES = "lines"


def calculate_metrics(
    predictions: list[MetricPrediction],
    references: list[MetricReference],
    mode: StructureFormat = StructureFormat.TAGS,
) -> dict[str, float]:
    instances: list[Instance] = []
    for pred, refer in zip(predictions, references):
        assert pred["id"] == refer["id"]

        pred_entities, pred_relation = parse_instance(pred["prediction_text"], mode)
        ref_entities, ref_relation = parse_instance(refer["answers"], mode)

        assert ref_relation == refer["question_type"], (
            "Extracted reference relation does not match the question type."
            f" {ref_relation} != {refer['question_type']}"
        )

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
    s = s.lower()

    # remove punctuation
    exclude = set(string.punctuation)
    s = "".join(ch for ch in s if ch not in exclude)

    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # remove extra whitespace
    s = " ".join(s.split())

    return s


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def compute_metrics(instances: list[Instance]) -> dict[str, float]:
    extraction = compute_extraction_metrics(instances)
    classification = compute_classification_metrics(instances)
    return {**extraction, **classification}


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
    # sourcery skip: assign-if-exp
    gold_lens = {"Cause": 0, "Effect": 0}
    pred_lens = {"Cause": 0, "Effect": 0}
    commons = {"Cause": 0, "Effect": 0}
    equal = {"Cause": 0, "Effect": 0}
    num_instances = {"Cause": 0, "Effect": 0}

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


def parse_instance(
    answer: str, mode: StructureFormat
) -> tuple[dict[str, list[str]], str]:
    if mode == StructureFormat.TAGS:
        return parse_instance_tags(answer)
    else:
        return parse_instance_lines(answer)


def clean(s: str) -> str:
    s = s.strip()
    return re.sub(r'""+"', '"', s)


def parse_spans(
    answer: str, pattern: str, flags: int = 0
) -> tuple[dict[str, list[str]], str]:
    match = re.search(pattern, answer, flags)
    if not match:
        return {"Cause": [], "Effect": []}, "cause"

    causes = sorted(s for c in match["cause"].split("|") if (s := clean(c)))
    effects = sorted(s for e in match["effect"].split("|") if (s := clean(e)))
    relation = match["relation"].strip().lower()

    return {"Cause": causes, "Effect": effects}, relation


def parse_instance_lines(answer: str) -> tuple[dict[str, list[str]], str]:
    pattern = r"Cause:(?P<cause>.*)Effect:(?P<effect>.*)Relation:(?P<relation>.*)"
    return parse_spans(answer, pattern, re.DOTALL)


def parse_instance_tags(answer: str) -> tuple[dict[str, list[str]], str]:
    pattern = r"\[Cause\](?P<cause>.*?)\[Relation\](?P<relation>.*?)\[Effect\](?P<effect>.*?)$"
    return parse_spans(answer, pattern)
