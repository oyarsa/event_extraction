#!/usr/bin/env python3
# pyright: basic
import argparse
import json
import sys
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("label", type=str)
parser.add_argument("--short", action="store_true")

args = parser.parse_args()
data = json.load(sys.stdin)

preds = [d[args.label] for d in data]
freqs = Counter(preds)

if not args.short:
    print(f"Total: {len(preds)}\n")

for k, v in sorted(freqs.items()):
    print(f"{k}: {v} ({v/len(preds):.2%})")
