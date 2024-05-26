#!/usr/bin/env python3
"""Combine data auto-tagged with human-annotated data."""

import argparse
import json
from pathlib import Path


def main(annotated_file: Path, tagged_file: Path) -> None:
    raw_ann = json.loads(annotated_file.read_text())
    tag = json.loads(tagged_file.read_text())
    ann = [a["data"] | {"valid": a["answer"] == "valid"} for a in raw_ann]

    # Find the tag for each annotation
    for a in ann:
        for t in tag:
            if a["reference"] == t["reference"] and a["model"] == a["model"]:
                a["tag"] = t["tag"]
                break

    annout = [
        {
            "input": a["text"],
            "gold": a["reference"],
            "output": a["model"],
            "tag": a["tag"],
            "valid": a["valid"],
        }
        for a in ann
    ]

    # Items that don't need annotation have automatic valid values
    tagout = [
        {
            "input": t["text"],
            "gold": t["reference"],
            "output": t["model"],
            "tag": t["tag"],
            "valid": t["tag"] == "exact_match",
        }
        for t in tag
        if t["tag"] != "needs_annotation"
    ]

    out = annout + tagout
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("annotated_file", type=Path, help="Human-annotated file")
    parser.add_argument("tagged_file", type=Path, help="Data file with tags")
    args = parser.parse_args()
    main(args.annotated_file, args.tagged_file)
