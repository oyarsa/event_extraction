#!/usr/bin/env python3
"""Convert the Maven ERE dataset to the tagged format.

Input file is the MAVEN-ERE JSONLines file from https://github.com/THU-KEG/MAVEN-ERE.
Download script: https://github.com/THU-KEG/MAVEN-ERE/blob/ac81a9711a69f43f55bfbc50b3bb573fd11c64b0/data/download_maven.sh
Direct link: https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1

Only train.jsonl and valid.jsonl can be used since test.jsonl doesn't have the events,
just the input sentences.

Since MAVEN-ERE is somewhat big (the processed train.jsonl is almost 100M), this script
allows for compression of the output using gzip. It works pretty well; the compressed
train.jsonl with default settings is 2.9M. However, this might require some small
changes to training scripts to handle gzipped files.
"""

import argparse
import gzip
import hashlib
import json
from typing import TextIO

REL2ID = {
    "CAUSE": 0,
    "PRECONDITION": 1,
    "NONE": 2,
}

# Instance keys used to compute the hash ID
HASH_KEYS = ("context", "question", "answers")


def hash_instance(
    instance: dict[str, str], keys: tuple[str, ...] = HASH_KEYS, length: int = 8
) -> str:
    """Create unique ID for the instance by hashing."""
    key = " ".join(instance[k] for k in keys)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def main(input_file: TextIO, output_file: str) -> None:
    # sourcery skip: low-code-quality
    answer_template = "[Cause] {cause} [Relation] cause [Effect] {effect}"
    question = "What are the events?"
    question_type = "cause"

    examples: list[dict[str, str]] = []

    for line in input_file:
        data = json.loads(line.strip())

        if "causal_relations" not in data:
            continue

        events: list[dict[str, str]] = []
        event_id_to_mentions: dict[str, list[str]] = {}

        # save all the events so that we can trace the casual ones from it
        for e in data["events"]:
            events.extend(e["mention"])
            event_id_to_mentions[e["id"]] = [m["id"] for m in e["mention"]]

        relations: dict[str, list[tuple[str, str]]] = data["causal_relations"]

        # save the causal-effect pair
        pair2rel = {
            (cause_event_id, effect_event_id): REL2ID[rel]
            for rel in relations
            for cause_id, effect_id in relations[rel]
            for cause_event_id in event_id_to_mentions[cause_id]
            for effect_event_id in event_id_to_mentions[effect_id]
        }

        sentences = data["sentences"]

        done = False
        for cause_event in events:
            if done:
                break

            for effect_event in events:
                if cause_event["id"] == effect_event["id"]:
                    continue

                if (cause_event["id"], effect_event["id"]) not in pair2rel:
                    continue

                if cause_event["sent_id"] == effect_event["sent_id"]:
                    continue

                context = " ".join(sentences)
                cause = sentences[cause_event["sent_id"]]
                effect = sentences[effect_event["sent_id"]]

                answer = answer_template.format(cause=cause, effect=effect)

                example = {
                    "context": context,
                    "question": question,
                    "question_type": question_type,
                    "answers": answer,
                    "sentences": sentences,
                }
                example["id"] = hash_instance(example)

                examples.append(example)
                done = True
                break

    print(f"{len(examples)} examples.")

    data_output = {"version": "v1.0", "data": examples}

    open_ = gzip.open if output_file.endswith(".gz") else open
    with open_(output_file, "wt") as f:
        json.dump(data_output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="Input MAVEN-ERE JSONLines file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to output tagged JSON file. If it ends with .gz, it will be gzipped.",
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
