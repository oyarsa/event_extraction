#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("label", type=str, help="The label to analyze in the input data.")
parser.add_argument(
    "file",
    nargs="?",
    type=argparse.FileType("r"),
    default=sys.stdin,
    help="Optional JSON file to read from; reads from stdin if not provided.",
)
parser.add_argument("--compact", "-c", action="store_true", help="Compact output.")
args = parser.parse_args()

data = json.load(args.file)
preds = [d[args.label] for d in data]
freqs = Counter(preds)

if not args.compact:
    print(f"Total: {len(preds)}\n")

for k, v in sorted(freqs.items()):
    print(f"{k}: {v} ({v / len(preds):.2%})")
