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
from pathlib import Path
from typing import Any

from common import convert_file_qa, generate_answer_combined_tags, hash_instance


def convert_genqa_joint(instance: dict[str, Any]) -> list[dict[str, str]]:
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

    This (raw input):
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
    },
    ```
    Becomes:
    ```json
    {
      "context": "If one or more of Ecolab's customers were to experience a disastrous outcome, the firm's reputation could suffer and it could lose multiple customers as a result.",  # noqa
      "question": "What are the reconstructe",
      "question_type": "cause",
      "answers": "[Cause] one or more of Ecolab's customers were to experience a disastrous outcome [Relation] cause [Effect] the firm's reputation could suffer and it could lose multiple customers",  # noqa
      "id": "86951ffe"
    }
    ```
    """
    text = instance["info"]
    label_map = {"reason": "Cause", "result": "Effect"}

    instances: list[dict[str, str]] = []

    for i, label_data in enumerate(instance["labelData"]):
        relation = label_data["type"]

        events: dict[str, list[str]] = {"reason": [], "result": []}
        for ev_type in ["reason", "result"]:
            for ev_start, ev_end in label_data[ev_type]:
                event = text[ev_start:ev_end]
                events[ev_type].append(event)

        answer = generate_answer_combined_tags(events, label_map, relation)

        question = "What are the events?"
        if len(instance["labelData"]) > 1:
            question = f"{i} {question}"

        inst = {
            "context": text,
            "question": question,
            "question_type": relation,
            "answers": answer,
        }
        # There are duplicate IDs in the dataset, so we hash instead.
        inst["id"] = hash_instance(inst)
        instances.append(inst)

    return instances


def main() -> None:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--src",
        type=Path,
        default="data/raw",
        help="Path to the folder containing the raw data",
    )
    argparser.add_argument(
        "--dst", type=Path, default="data/genqa_joint", help="Path to the output folder"
    )
    args = argparser.parse_args()

    splits = ["dev", "test", "train"]
    for split in splits:
        raw_path = args.raw_folder / f"event_dataset_{split}.json"
        new_path = args.new_folder / f"{split}.json"
        convert_file_qa(raw_path, new_path, convert_instance=convert_genqa_joint)


if __name__ == "__main__":
    main()
