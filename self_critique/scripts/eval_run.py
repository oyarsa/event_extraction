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
    cause_predictions: list[str]
    cause_golds: list[str]
    effect_predictions: list[str]
    effect_golds: list[str]


def rewrite_clause(parts: list[str]) -> str:
    return ", and".join(parts)


def evaluate(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for inst in instances:
        pred_entities, _ = parse_instance(inst["output"])
        ref_entities, _ = parse_instance(inst["gold"])

        instance: Instance = {
            "cause_predictions": pred_entities["cause"],
            "cause_golds": ref_entities["cause"],
            "effect_predictions": pred_entities["effect"],
            "effect_golds": ref_entities["effect"],
        }
        label = "VALID" if is_valid(instance) else "INVALID"
        result.append(
            {
                "input": inst["input"],
                "gold": inst["gold"],
                "output": inst["output"],
                "reward_label": label,
            }
        )

    return result


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.casefold()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def is_valid(instance: Instance) -> bool:
    cause_pred_toks = get_tokens(" ".join(instance["cause_predictions"]))
    cause_gold_toks = get_tokens(" ".join(instance["cause_golds"]))
    effect_pred_toks = get_tokens(" ".join(instance["effect_predictions"]))
    effect_gold_toks = get_tokens(" ".join(instance["effect_golds"]))

    return cause_gold_toks == cause_pred_toks and effect_gold_toks == effect_pred_toks


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


def main(infile: Path, outfile: Path) -> None:
    """
    Add "reward_label" to instances based on exact match.

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

    The output JSON has the same structure as the input JSON, but with an additional
    "reward_label" field, with value "VALID" or "INVALID".
    """
    data = json.loads(infile.read_text())
    results = evaluate(data)
    outfile.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    typer.run(main)
