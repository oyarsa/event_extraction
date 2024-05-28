import argparse
import hashlib
import json
from typing import Any, TextIO


def hash_item(data: dict[str, Any], keys: list[str], length: int = 8) -> str:
    key = "".join([data[key] for key in keys])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:length]


def main(file1: TextIO, file2: TextIO, outfile: TextIO) -> None:
    data1 = json.load(file1)
    data2 = json.load(file2)

    hash_keys = ["text", "reference", "model"]

    ann_not_found: list[dict[str, Any]] = []
    found: list[dict[str, Any]] = []

    for tag_item in data1:
        tag_hash = hash_item(tag_item, hash_keys)
        for ann_item in data2:
            if tag_hash == hash_item(ann_item, hash_keys):
                found.append(ann_item)
                break

        ann_not_found.append(tag_item)

    print(f"Annotations found: {len(found)}")
    print(f"Annotations not found: {len(ann_not_found)}")

    # json.dump(tagged, outfile, indent=2)


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
