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
      "cause": ["one or more of Ecolab's customers were to experience a disastrous outcome"],  # noqa
      "effect": ["the firm's reputation could suffer and it could lose multiple customers outcome", "the firm's reputation could suffer and it could lose multiple customers"],  # noqa
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


def convert_instance(instance: dict[str, Any]) -> dict[str, Any]:
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
    text = instance["info"]

    events_text: dict[str, list[str]] = {"reason": [], "result": []}
    events_start: dict[str, list[int]] = {"reason": [], "result": []}

    for label_data in instance["labelData"]:
        for ev_type in ["reason", "result"]:
            for ev_start, ev_end in label_data[ev_type]:
                event = text[ev_start:ev_end]
                events_text[ev_type].append(event)
                events_start[ev_type].append(ev_start)

    # There are duplicate IDs in the dataset, so we hash instead.
    # I'm hashing the entire instance instead of just the text in an abundance
    # of caution.
    instance_id = hashlib.sha1(str(instance).encode("utf-8")).hexdigest()[:8]

    return {
        "id": instance_id,
        "context": text,
        "question": "What are the events?",
        "question_type": "Events",
        # Limiting to one answer each because the model doesn't support multi-span yet.
        "cause": {
            "text": events_text["reason"][:1],
            "answer_start": events_start["reason"][:1],
        },
        "effect": {
            "text": events_text["result"][:1],
            "answer_start": events_start["result"][:1],
        },
    }


def convert_file(infile: Path, outfile: Path) -> None:
    with infile.open() as f:
        dataset = json.load(f)

    instances = [convert_instance(instance) for instance in dataset]
    transformed = {"version": "v1.0", "data": instances}

    outfile.parent.mkdir(exist_ok=True, parents=True)
    with outfile.open("w") as f:
        json.dump(transformed, f)


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--src",
        type=Path,
        help="Path to the folder containing the raw data",
    )
    argparser.add_argument(
        "--dst", type=Path, default="data", help="Path to the output folder"
    )
    args = argparser.parse_args()

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = args.raw_folder / f"event_dataset_{split}.json"
        new_path = args.new_folder / f"{split}.json"
        convert_file(raw_path, new_path)


if __name__ == "__main__":
    main()
