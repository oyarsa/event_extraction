#!/usr/bin/env python3
# pyright: basic
import json
import sys
from collections import Counter

label = sys.argv[1] if len(sys.argv) > 1 else "label"
data = json.load(sys.stdin)

preds = [d["label"] for d in data]
freqs = Counter(preds)

for k, v in freqs.items():
    print(f"{k}: {v} ({v/len(preds):.2%})")
