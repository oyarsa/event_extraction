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
from typing import Dict, List

import datasets
from typing_extensions import TypedDict  # Python 3.7 doesn't have this in typing


class Instance(TypedDict):
    id: str
    kind: str
    predictions: List[str]
    golds: List[str]


class MetricPrediction(TypedDict):
    id: str
    prediction_text: str


class MetricReference(TypedDict):
    id: str
    answers: str
    question_type: str


class FGCR(datasets.Metric):
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
        return datasets.MetricInfo(description="", citation="", features=features)

    def _compute(
        self, predictions: List[MetricPrediction], references: List[MetricReference]
    ) -> Dict[str, float]:
        instances: list[Instance] = []
        for pred, refer in zip(predictions, references):
            assert pred["id"] == refer["id"]

            pred_entities = parse_instance(pred["prediction_text"])
            ref_entities = parse_instance(refer["answers"])

            for itype in ref_entities.keys():
                instance: Instance = {
                    "id": refer["id"],
                    "kind": itype,
                    "predictions": pred_entities[itype],
                    "golds": ref_entities[itype],
                }
                instances.append(instance)

        result = compute_metric(instances)
        return result


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


def get_tokens(s: str) -> List[str]:
    return normalize_answer(s).split()


def compute_metric(instances: List[Instance]) -> Dict[str, float]:
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


def parse_instance(answer: str) -> Dict[str, List[str]]:
    """Parse string answer to separate into class and spans
    Simple case:
    [Cause] This is a cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Effect] This effect 1 | This effect 2
    """
    matches = re.findall(r"\[Cause\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return {
            "Cause": [],
            "Effect": [],
        }
    causes, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())

    return {
        "Cause": causes,
        "Effect": effects,
    }
