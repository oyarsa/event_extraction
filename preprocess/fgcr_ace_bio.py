"""Convert the FGCR dataset json to the format that the ACE model accepts.

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
If O
one B-Cause
or I-Cause
more I-Cause
of I-Cause
Ecolab I-Cause
's I-Cause
customers I-Cause
were I-Cause
to I-Cause
experience I-Cause
a I-Cause
disastrous I-Cause
outcome I-Cause
, O
the B-Effect
firm I-Effect
's I-Effect
reputation I-Effect
could I-Effect
suffer I-Effect
and I-Effect
it I-Effect
could I-Effect
lose I-Effect
multiple I-Effect
customers I-Effect
as O
a O
result O
. O

```
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nltk.tokenize import NLTKWordTokenizer


def convert_instance(instance: dict[str, Any]) -> list[tuple[str, str]]:
    """Convert a FGCR-format instance into an ACE-format instance.

    This ignores the relationship and only annotates the causes and effects using
    B-Cause, I-Cause, B-Effect, I-Effect and O.

    This happens at the token level, so we tokenise the text using the
    NLTKWordTokenizer. The output is a list of (token, label) pairs.
    """
    text = instance["info"]
    spans = list(NLTKWordTokenizer().span_tokenize(text))

    label_map = {"reason": "Cause", "result": "Effect"}

    labels = {}
    for label_data in instance["labelData"]:
        for ev_type in ["reason", "result"]:
            for ev_start, ev_end in label_data[ev_type]:
                is_first = True
                for t_start, t_end in spans:
                    if ev_start <= t_start and t_end <= ev_end:
                        tag = "B" if is_first else "I"
                        is_first = False
                        labels[(t_start, t_end)] = f"{tag}-{label_map[ev_type]}"
    out = []
    for start, end in spans:
        token = text[start:end]
        label = labels.get((start, end), "O")
        out.append((token, label))

    return out


def format_instance(instances: list[tuple[str, str]]) -> str:
    return "\n".join(f"{token} {label}" for token, label in instances)


def convert_file(infile: Path, outfile: Path) -> None:
    with open(infile) as f:
        dataset = json.load(f)

    instances = [convert_instance(instance) for instance in dataset]
    converted = "\n\n".join(format_instance(i) for i in instances)

    outfile.parent.mkdir(exist_ok=True)
    with open(outfile, "w") as f:
        print(converted, file=f)


def main() -> None:
    raw_folder = Path("../data_fgcr/raw")
    new_folder = Path("./ace_bio")

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = raw_folder / f"event_dataset_{split}.json"
        new_path = new_folder / f"{split}.txt"
        convert_file(raw_path, new_path)


if __name__ == "__main__":
    main()
