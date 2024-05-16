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
"""Calculate macro-averaged Exact Match (EM) for Cause and Effect clauses."""

import re
import string
from typing import TypedDict


class Prediction(TypedDict):
    id: str
    prediction_text: str


class Reference(TypedDict):
    id: str
    answers: str


class Instance(TypedDict):
    clause: str
    predictions: list[str]
    golds: list[str]


# sourcery skip: snake-case-variable-declarations
class ParsedInstance(TypedDict):
    Cause: list[str]
    Effect: list[str]


def calc_macro_em(predictions: list[Prediction], references: list[Reference]) -> float:
    """Calculate macro-averaged Exact Match (EM) for Cause and Effect clauses.

    Instances where the causes match but the effects don't, or vice-verse, count as
    half since this is macro-averaged between the cause/effect clauses.
    """
    instances: list[Instance] = []

    for pred, refer in zip(predictions, references):
        assert pred["id"] == refer["id"]

        pred_entities = parse_instance(pred["prediction_text"])
        ref_entities = parse_instance(refer["answers"])

        # Two instances per example: one Cause and one Effect, each with pred/gold
        instances.extend(
            {
                "clause": clause,
                "predictions": pred_entities[clause],
                "golds": ref_entities[clause],
            }
            for clause in ref_entities
        )

    return compute_macro_em(instances)


def get_tokens(s: str) -> list[str]:
    """Clean text and split into words.

    Lower text and remove punctuation, articles and extra whitespace.
    """
    s = s.casefold()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())  # remove repeated whitespace
    return s.split()


def compute_macro_em(instances: list[Instance]) -> float:
    """Compute the macro-averaged EM between predictions and golds.

    Instances are dealt with together, but they are separated into Cause and Effect
    clauses. The EM is calculated for each separate clause (twice per instance) and then
    averaged.
    """
    equal = {"Cause": 0, "Effect": 0}
    num_instances = {"Cause": 0, "Effect": 0}

    for instance in instances:
        clause = instance["clause"]
        # Join the (potentially) multiple causes/effects into a single string, then
        # split into tokens for comparison.
        pred_toks = get_tokens(" ".join(instance["predictions"]))
        gold_toks = get_tokens(" ".join(instance["golds"]))

        equal[clause] += int(gold_toks == pred_toks)
        num_instances[clause] += 1

    result = {clause: equal[clause] / num_instances[clause] for clause in equal}
    # Macro average the EM between Cause and Effect
    return sum(result.values()) / len(result)


def parse_instance(answer: str) -> ParsedInstance:
    """Parse string answer into causes and effects.

    Simple case:
    [Cause] This is a cause [Relation] cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Relation] effect [Effect] This effect 1 | This effect 2

    This version ignores the relations and only returns the causes and effects.
    """
    matches = re.findall(r"\[Cause\](.*?)\[Relation\].*?\[Effect\](.*?)$", answer)
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
