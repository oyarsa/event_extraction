"Annotate train and test output with valid label where possible."
import json
import re
from pathlib import Path
from typing import Any

import typer


def parse_instance(answer: str) -> tuple[str | None, str | None]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return None, None
    causes, _, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())

    if not (causes and effects):
        return None, None
    return causes[0], effects[0]


def clean_str(s: str) -> str:
    s = s.lower().strip()
    return re.sub(r"\s", "", s)


def symm_substr(a: str, b: str) -> bool:
    a = clean_str(a)
    b = clean_str(b)
    return a in b or b in a


def annotate_entry(gold: str, pred: str) -> bool | None:
    cause_pred, effect_pred = parse_instance(pred)
    cause_gold, effect_gold = parse_instance(gold)

    # All clauses are empty means it's invalid.
    if not (cause_pred and effect_pred and cause_gold and effect_gold):
        return False

    # Exact match means it's valid.
    if cause_pred == cause_gold and effect_pred == effect_gold:
        return True

    # For both cause and effect, if one is a substring of the other, then it _could_ be
    # valid. Needs a manual check.
    if symm_substr(cause_pred, cause_gold) and symm_substr(effect_pred, effect_gold):
        return None
    # If not substrings, it must be invalid.
    return False


def annotate(data: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {**entry, "valid": annotate_entry(entry["gold"], entry["output"])}
        for entry in data
    ]


def main(data_path: Path, output_path: Path, add_classify: bool = False) -> None:
    data = json.loads(data_path.read_text())

    annotated = annotate(data)
    for value in [True, False, None]:
        print(f"{value}: {sum(entry['valid'] is value for entry in annotated)}")

    rule_labelled = [entry for entry in annotated if entry["valid"] is not None]
    to_classify = [entry for entry in annotated if entry["valid"] is None]

    output_path.mkdir(exist_ok=True, parents=True)
    (output_path / "rule_labelled.json").write_text(json.dumps(rule_labelled, indent=2))
    if add_classify:
        (output_path / "to_classify.json").write_text(json.dumps(to_classify, indent=2))


if __name__ == "__main__":
    typer.run(main)
