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
"""Hugginface Metric for Reconstruction"""

import re
import string
from collections import Counter
from typing import TypedDict

import datasets
import evaluate


class Instance(TypedDict):
    id: str
    prediction: str
    gold: str


class MetricPrediction(TypedDict):
    id: str
    prediction_text: str


class MetricReference(TypedDict):
    id: str
    answers: str
    question_type: str


class ReconstructMetric(evaluate.Metric):
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

            instance: Instance = {
                "id": refer["id"],
                "prediction": pred["prediction_text"],
                "gold": refer["answers"],
            }
            instances.append(instance)

        return compute_metrics(instances)


_PUNCTUATION = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()

    # remove punctuation
    s = "".join(ch for ch in s if ch not in _PUNCTUATION)

    # remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # remove extra whitespace
    s = " ".join(s.split())

    return s


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def compute_metrics(instances: list[Instance]) -> dict[str, float]:
    gold_lens = 0
    pred_lens = 0
    commons = 0
    equal = 0
    num_instances = 0

    for instance in instances:
        pred_toks = get_tokens(instance["prediction"])
        gold_toks = get_tokens(instance["gold"])

        gold_lens += len(gold_toks)
        pred_lens += len(pred_toks)

        common = Counter(gold_toks) & Counter(pred_toks)
        commons += sum(common.values())

        equal += int(gold_toks == pred_toks)
        num_instances += 1

    if pred_lens != 0:
        precision = commons / pred_lens
    else:
        precision = 0

    recall = commons / gold_lens

    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "em": equal / num_instances,
    }
