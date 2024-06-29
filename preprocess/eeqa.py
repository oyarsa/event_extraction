"""Convert the FGCR dataset json to the format that the EEQA model accepts.

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
  ],
  "id": <same as 'tid'>
}
```

Example output:
```
{
  "sentence": <tokenised 'info'>,
  "s_start": 0,
  "ner": [],  # not used by the model
  "relation": [],  # not used by the model
  "event": [  # I'm not sure why there are two levels of lists
    [
      [
        span_start,
        span_end,
        type
      ]
    ]
  ]
}
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nltk.tokenize import NLTKWordTokenizer


def convert_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Convert a FGCR-format instance into an EEQA-format instance.

    This ignores the relationship and only annotates the causes and effects by
    building a list of (start, end, label) triplets.

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
    tokeniser = NLTKWordTokenizer()
    text = instance["info"]
    tokens = list(tokeniser.tokenize(text))
    spans = list(tokeniser.span_tokenize(text))

    label_map = {"reason": "Cause", "result": "Effect"}

    out = {
        "sentence": tokens,
        "s_start": 0,
        "ner": [],
        "relation": [],
        "id": instance["tid"],
    }

    events = []
    for label_data in instance["labelData"]:
        for ev_type in ["reason", "result"]:
            for ev_start, ev_end in label_data[ev_type]:
                start, end = -1, -1
                for sindex, (t_start, t_end) in enumerate(spans):
                    if ev_start <= t_start and t_end <= ev_end:
                        if start == -1:
                            start = sindex
                        end = sindex
                if start != -1 and end != -1:
                    events.append([start, end, label_map[ev_type]])

    out["event"] = [events]
    return out


def convert_file(infile: Path, outfile: Path) -> None:
    with infile.open() as f:
        dataset = json.load(f)

    instances = [convert_instance(instance) for instance in dataset]

    outfile.parent.mkdir(exist_ok=True, parents=True)
    with outfile.open("w") as f:
        for i in instances:
            print(json.dumps(i), file=f)


def main() -> None:
    raw_folder = Path("data/raw")
    new_folder = Path("data/eeqa")

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = raw_folder / f"event_dataset_{split}.json"
        new_path = new_folder / f"{split}.json"
        convert_file(raw_path, new_path)


if __name__ == "__main__":
    main()
