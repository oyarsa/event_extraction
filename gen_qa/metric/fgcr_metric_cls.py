# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hugginface Metric for FGCR"""

import re
import string
from collections import Counter, defaultdict

import datasets
import evaluate
from sklearn.metrics import precision_recall_fscore_support
from typing_extensions import TypedDict  # Python 3.7 doesn't have this in typing


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


class FGCRCls(evaluate.Metric):
    def _info(self):
        features = datasets.Features(
            {
                "predictions": {
                    "id": datasets.Value("string"),
                    "prediction_text": datasets.Value("string"),
                },
                "references": {
                    "id": datasets.Value("string"),
                    "answers": datasets.Value("string"),
                    "question_type": datasets.Value("string"),
                },
            }
        )
        return evaluate.MetricInfo(description="", citation="", features=features)

    def _compute(
        self, predictions: list[MetricPrediction], references: list[MetricReference]
    ) -> dict[str, float]:
        instances: list[Instance] = []
        for pred, refer in zip(predictions, references):
            assert pred["id"] == refer["id"]

            pred_entities, pred_relation = parse_instance(pred["prediction_text"])
            ref_entities, ref_relation = parse_instance(refer["answers"])

            assert (
                ref_relation == refer["question_type"]
            ), "Extracted reference relation does not match the question type."

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
        gold, pred, average="macro", zero_division=0
    )

    return {
        "cls_accuracy": accuracy,
        "cls_precision": float(precision),
        "cls_recall": float(recall),
        "cls_f1": float(f1),
    }


def compute_extraction_metrics(instances: list[Instance]) -> dict[str, float]:
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

    result = defaultdict(dict)
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


def parse_instance(answer: str) -> tuple[dict[str, list[str]], str | None]:
    """Parse string answer to separate into class and spans
    Simple case:
    [Cause] This is a cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Effect] This effect 1 | This effect 2
    """
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return {
            "Cause": [],
            "Effect": [],
        }, "cause"
    causes, effects, relation = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())
    relation = relation.strip()

    return {
        "Cause": causes,
        "Effect": effects,
    }, relation
