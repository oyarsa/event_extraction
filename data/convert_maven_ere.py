#!/usr/bin/env python3
"""Convert the Maven ERE dataset to the tagged format."""

import argparse
import hashlib
import json
from typing import TextIO

REL2ID = {
    "CAUSE": 0,
    "PRECONDITION": 1,
    "NONE": 2,
}

# Instance keys used to compute the hash ID
HASH_KEYS = ["context", "question", "answers"]


def hash_instance(instance: dict[str, str], keys: list[str], length: int = 8) -> str:
    key = " ".join([instance[k] for k in keys])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def main(input_file: TextIO, output_file: TextIO) -> None:
    # sourcery skip: low-code-quality
    lines = input_file.readlines()

    answer_template = "[Cause] {cause} [Relation] cause [Effect] {effect}"
    question = "What are the events?"
    question_type = "cause"
    examples: list[dict[str, str]] = []

    for doc_id, line in enumerate(lines):
        data = json.loads(line.strip())
        sentences = data["sentences"]

        events: list[dict[str, str]] = []
        eid2mentions: dict[str, list[str]] = {}

        # save all the events so that we can trace the casual ones from it
        if "events" not in data:
            events = data["event_mentions"]
        else:
            for e in data["events"]:
                events += e["mention"]
                eid2mentions[e["id"]] = [m["id"] for m in e["mention"]]
        relations = data["causal_relations"]

        # save the causal-effect pair
        pair2rel = {}
        for rel in relations:
            for pair in relations[rel]:
                for cause_event in eid2mentions[pair[0]]:
                    for effect_event in eid2mentions[pair[1]]:
                        pair2rel[(cause_event, effect_event)] = REL2ID[rel]

        n_samples_per_doc = 0
        for cause_event in events:
            for effect_event in events:
                if cause_event["id"] == effect_event["id"]:
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
                }
                example["id"] = hash_instance(example, HASH_KEYS)
                examples.append(example)
                n_samples_per_doc += 1

        print(f"There are {n_samples_per_doc} for {doc_id}th doc")

    print(f"{len(examples)} examples.")

    data_output = {"version": "v1.0", "data": examples}
    json.dump(data_output, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file", type=argparse.FileType("r"), help="Input MAVEN-ERE JSON file"
    )
    parser.add_argument(
        "output_file", type=argparse.FileType("w"), help="Output tagged JSON file"
    )
    args = parser.parse_args()
    main(args.input_file, args.output_file)
