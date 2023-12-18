#!/usr/bin/env python
# pyright: basic
# ruff: noqa: E501
import json
import re
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import typer


@dataclass
class InputData:
    input: str
    gold: str
    output: str


@dataclass
class OutputData:
    input: str
    gold: str
    output: str
    reward_label: str


@dataclass
class ParsedInstance:
    cause: list[str]
    effect: list[str]
    relation: str


def evaluate(instances: list[InputData]) -> list[OutputData]:
    result: list[OutputData] = []

    for inst in instances:
        gold = parse_instance(inst.gold)
        pred = parse_instance(inst.output)

        label = "VALID" if is_valid(pred, gold) else "INVALID"
        result.append(
            OutputData(
                input=inst.input,
                gold=inst.gold,
                output=inst.output,
                reward_label=label,
            )
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


def is_valid(pred: ParsedInstance, gold: ParsedInstance) -> bool:
    cause_pred_toks = get_tokens(" ".join(pred.cause))
    cause_gold_toks = get_tokens(" ".join(gold.cause))
    effect_pred_toks = get_tokens(" ".join(pred.effect))
    effect_gold_toks = get_tokens(" ".join(gold.effect))

    return (
        cause_gold_toks == cause_pred_toks
        and effect_gold_toks == effect_pred_toks
        and pred.relation == gold.relation
    )


def clean(s: str) -> str:
    s = s.strip()
    return re.sub(r'""+"', '"', s)


def parse_instance(answer: str) -> ParsedInstance:
    """Parse string answer to separate into class and spans
    Simple case:
    [Cause] This is a cause [Relation] cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Relation] enable [Effect] This effect 1 | This effect 2
    """
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return ParsedInstance(cause=[], effect=[], relation="cause")

    causes, relation, effects = matches[0]
    causes = sorted(s for c in causes.split("|") if (s := clean(c)))
    effects = sorted(s for e in effects.split("|") if (s := clean(e)))
    relation = relation.strip().lower()

    return ParsedInstance(cause=causes, effect=effects, relation=relation)


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
    input_data = [
        InputData(
            input=item["input"],
            gold=item["gold"],
            output=item["output"],
        )
        for item in json.loads(infile.read_text())
    ]
    results = evaluate(input_data)
    outfile.write_text(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    typer.run(main)
