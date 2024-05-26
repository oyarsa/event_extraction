#!/usr/bin/env python3
"""Convert annotation results to a common format, clean and collect the data.

The output has all the answers for each instance, with majority voting for the valid
label.

The result can be passed to `calc_kappa.py` to calculate pairwise Cohen's Kappa score.
"""

import argparse
import hashlib
import json
from typing import Any, TextIO


def hash_data(data: dict[str, str]) -> str:
    """Hash data (list of strings) using sha256 with the first 8 characters."""
    keys = ["text", "reference", "model"]
    return hashlib.sha256("".join(data[k] for k in keys).encode()).hexdigest()[:8]


def clean_name(name: str) -> str:
    """Clean name by keeping only ASCII characters."""
    return name.encode("ascii", "ignore").decode()


def shape_data(username: str, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "username": username,
            "id": d["id"],
            "answer": d["answer"],
            "text": d["data"]["text"],
            "reference": d["data"]["reference"],
            "model": d["data"]["model"],
        }
        for d in data
    ]


def main(
    train_file: TextIO,
    dev_ann_file: TextIO,
    raw_files: list[TextIO],
    output_file: TextIO,
) -> None:
    train_data = json.load(train_file)

    # Dev also contains Italo's annotations
    dev_data = json.load(dev_ann_file)
    inputs = [shape_data("italo", dev_data)]

    # We need to convert the annotations from the others to the same format
    for file in raw_files:
        raw_data = json.load(file)
        username = clean_name(raw_data["username"])
        inputs.append(shape_data(username, raw_data["items"]))

    indexed_instances: dict[str, dict[str, Any]] = {}
    for data in inputs:
        for d in data:
            if d["id"] not in indexed_instances:
                new_d = d.copy()
                new_d["answers"] = []
                del new_d["answer"]
                indexed_instances[d["id"]] = new_d

            indexed_instances[d["id"]]["answers"].append(
                {
                    "name": d["username"],
                    "answer": d["answer"],
                }
            )
            # Create standardised ID across all datasets
            indexed_instances[d["id"]]["new_id"] = hash_data(d)

    instances = list(indexed_instances.values())

    # Find where the instances came from
    id_to_source = {hash_data(d["data"]): "dev" for d in dev_data} | {
        hash_data(d): "train" for d in train_data
    }
    for item in instances:
        item["source"] = id_to_source.get(item["new_id"])

    # Use majority voting (defaulting to False) to get the valid label
    for item in instances:
        answers = [a["answer"] == "valid" for a in item["answers"]]
        item["valid"] = answers.count(True) > answers.count(False)

    json.dump(instances, output_file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="\n".join(__doc__.splitlines()[1:]),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--train",
        type=argparse.FileType("r"),
        help="Path to the training data file",
        required=True,
    )
    parser.add_argument(
        "--dev",
        type=argparse.FileType("r"),
        help="Path to the development data file with annotations",
        required=True,
    )
    parser.add_argument(
        "--ann",
        type=argparse.FileType("r"),
        nargs="+",
        help="Paths to the raw annotation files",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        help="Path to the output file",
        default="result.json",
    )
    args = parser.parse_args()
    main(args.train, args.dev, args.ann, args.output)
