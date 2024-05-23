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
"""Evaluate element-wise F1 score.

The F1 score is average of the F1 scores for Cause and Effect clauses.

The input data is a JSON file with a list of objects with the following shape:
- input: str = The input text passage
- output: str = The extracted relation in tag form
- gold: str = The gold relation in tag form
- valid: bool = True if the output is valid, False otherwise
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


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


def calc_f1_sentence(gold: list[str], pred: list[str]) -> float:
    gold_toks = get_tokens(" ".join(gold))
    pred_toks = get_tokens(" ".join(pred))

    if not gold_toks or not pred_toks:
        return 0

    common = Counter(gold_toks) & Counter(pred_toks)
    precision = sum(common.values()) / len(pred_toks)
    recall = sum(common.values()) / len(gold_toks)

    if precision + recall != 0:
        return (2 * precision * recall) / (precision + recall)
    else:
        return 0


def calc_f1_instance(gold: str, pred: str) -> float:
    gold_entities, _ = parse_instance(gold)
    pred_entities, _ = parse_instance(pred)

    f1_cause = calc_f1_sentence(gold_entities["Cause"], pred_entities["Cause"])
    f1_effect = calc_f1_sentence(gold_entities["Effect"], pred_entities["Effect"])

    return (f1_cause + f1_effect) / 2


def parse_instance(answer: str) -> tuple[dict[str, list[str]], str | None]:
    """Parse string answer to separate into class and spans
    Simple case:
    [Cause] This is a cause [Effect] This is an effect

    Complex case:
    [Cause] This cause 1 | This cause 2 [Effect] This effect 1 | This effect 2
    """
    # TODO (italo): Document the relation
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return {
            "Cause": [],
            "Effect": [],
        }, "cause"
    causes, relation, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())
    relation = relation.strip()

    return {
        "Cause": causes,
        "Effect": effects,
    }, relation


def main(input_file: Path, output_file: Path) -> None:
    data = json.loads(input_file.read_text())
    out = [
        {
            "gold": d["valid"],
            "pred": calc_f1_instance(d["gold"], d["output"]),
            "annotation": d["gold"],
            "output": d["output"],
            "input": d["input"],
        }
        for d in data
    ]
    output_file.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="\n".join(__doc__.splitlines()[1:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_file", type=Path, help="Path to the input file")
    parser.add_argument("output_file", type=Path, help="Path to the output file")
    args = parser.parse_args()
    main(args.input_file, args.output_file)
