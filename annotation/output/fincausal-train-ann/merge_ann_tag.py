import argparse
import hashlib
import json
import sys
from typing import Any, TextIO


def hash_item(data: dict[str, Any], keys: list[str], length: int = 8) -> str:
    key = "".join([data[key] for key in keys])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def main(tagged_file: TextIO, ann_file: TextIO, outfile: TextIO) -> None:
    tagged = json.load(tagged_file)
    ann = json.load(ann_file)

    hash_keys = ["text", "reference", "model"]
    ann_indexed = {hash_item(item, hash_keys): item["valid"] for item in ann}

    new_data: list[dict[str, Any]] = []

    for tag_item in tagged:
        if tag_item["tag"] != "needs_annotation":
            new_data.append(tag_item | {"valid": tag_item["tag"] == "exact_match"})
            continue

        tag_hash = hash_item(tag_item, hash_keys)
        if (answer := ann_indexed.get(tag_hash)) is not None:
            new_data.append(tag_item | {"valid": answer})

    not_found = len(tagged) - len(new_data)
    print(f"Annotations not found: {not_found}", file=sys.stderr)

    output = [
        {
            "input": item["text"],
            "gold": item["reference"],
            "output": item["model"],
            "valid": item["valid"],
            "tag": item["tag"],
        }
        for item in new_data
    ]
    json.dump(output, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tagged_file",
        type=argparse.FileType("r"),
        help="The tagged file to merge with the annotation file",
    )
    parser.add_argument(
        "ann_file", type=argparse.FileType("r"), help="The annotation file"
    )
    parser.add_argument("outfile", type=argparse.FileType("w"), help="The output file")
    args = parser.parse_args()
    main(args.tagged_file, args.ann_file, args.outfile)
