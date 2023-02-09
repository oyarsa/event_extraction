"""Convert the FGCR dataset json to the format that the GenQA model accepts.

Note: ACE doesn't support classification, so we're only generating the cause and
effect labels here.

Example input:
```json
{
  "tid": 2771,
  "info": "If one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers as a result.",  # noqa
  "extraInfo": null,
  "labelData": [
    {
      "type": "cause",
      "reason": [
        [
          3,
          76
        ]
      ],
      "result": [
        [
          78,
          149
        ]
      ]
    }
  ]
}
```

Example output:
```
{
  "version": "v1.0",
  "data": [
    {
      "id": "1234",
      "context": "If one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers as a result.",  # noqa
      "question": "What are the events?",
      "question_type": "Events",
      "answers": "[Cause] one or more of Ecolab's customers were to experience a disastrous outcome [Effect] the firm's reputation could suffer and it could lose multiple customers",  # noqa
    }
  ]
}
```
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def generate_answer(
    instances: list[tuple[list[str], str, list[str]]],
    combined_sep: str = " | ",
) -> str:
    answers: list[str] = []

    for causes, relation, effects in instances:
        causes = combined_sep.join(sorted(causes))
        effects = combined_sep.join(sorted(effects))
        answers.append(f"{causes} {relation} {effects}")

    return "\n".join(sorted(answers))


def convert_instance(
    instance: dict[str, Any], combined_sep: str = " | ", natural_like: bool = False
) -> dict[str, str]:
    """Convert a FGCR-format instance into an EEQA-format instance.

    This ignores the relationship and only annotates the causes and effects by
    building a list of (start, end, label) triplets.

    `mode` decides how we should handle multi-span cases (e.g. multiple reason spans).
    It can be 'separate', where every event is annotated separately (e.g. '[Cause] C1
    [Cause] C2') or 'combined', where all events of the same type are annotated together
    and separated by `COMBINED_SEP` (e.g. '[Cause] C1 | C2').

    `combined_sep` is the separator used when `mode` is 'combined'. It's discarded
    when `mode` is 'separate'.

    The spans are at the token level, so we tokenise the text using the
    NLTKWordTokenizer.

    The output is a dictionary with the following keys:
    - sentence: the tokenised text
    - s_start: always zero because we're considering a few sentences only, not a
               document
    - ner: unsupported, so always an empty list
    - relation: unsupported, so always an empty list
    - event: a list of lists of events, where each event is a triplet of
      [start, end, label]. I'm not sure why this is a 3-level list instead of
      just 2 levels (i.e. list of events).
    """
    if natural_like:
        combined_sep = " AND "

    text = instance["info"]
    relation_map = {"enable": "enables", "cause": "causes", "prevent": "prevents"}

    # Instance: list of causes, relation, list of effects
    instances: list[tuple[list[str], str, list[str]]] = []

    for label_data in instance["labelData"]:
        relation = relation_map[label_data["type"]]
        if natural_like:
            relation = relation.upper()
        else:
            relation = f"[{relation}]"

        events: dict[str, list[str]] = {"reason": [], "result": []}
        for ev_type in ["reason", "result"]:
            for ev_start, ev_end in label_data[ev_type]:
                event = text[ev_start:ev_end].strip()
                events[ev_type].append(event)

        instances.append((events["reason"], relation, events["result"]))

    inst = {
        "context": text,
        "question": "What are the events and relations?",
        "question_type": "Events",
        "answers": generate_answer(instances, combined_sep),
    }
    inst["id"] = hashlib.sha1(str(inst).encode("utf-8")).hexdigest()[:8]

    return inst


def convert_file(
    infile: Path, outfile: Path, combined_sep: str = " | ", natural_like: bool = False
) -> None:
    with open(infile) as f:
        dataset = json.load(f)

    instances = [
        convert_instance(instance, combined_sep, natural_like) for instance in dataset
    ]
    transformed = {"version": "v1.0", "data": instances}

    outfile.parent.mkdir(exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(transformed, f)


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--src",
        default="data/raw",
        help="Path to the folder containing the raw data",
    )
    argparser.add_argument(
        "--dst", default="data/genqa_joint", help="Path to the output folder"
    )
    argparser.add_argument(
        "--combined-sep", default=" | ", help="Separator for combined mode"
    )
    argparser.add_argument("--natural-like", action="store_true")
    args = argparser.parse_args()

    raw_folder = Path(args.src)
    new_folder = Path(args.dst)
    if args.natural_like:
        new_folder = new_folder / "natural"
    else:
        new_folder = new_folder / "original"

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = raw_folder / f"event_dataset_{split}.json"
        new_path = new_folder / f"{split}.json"
        convert_file(raw_path, new_path, args.combined_sep, args.natural_like)


if __name__ == "__main__":
    main()
