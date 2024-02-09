import json
import sys
from collections import Counter


def trykeys(d: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in d:
            return d[key]
    raise KeyError(f"None of the keys {keys} found in {d}")


with open(sys.argv[1]) as f:
    data = json.load(f)

data = [
    {
        "input": trykeys(d, "context", "input", "passage"),
        "gold": trykeys(d, "answers", "annotation", "gold"),
        "pred": trykeys(d, "output", "rl_extract_txt"),
    }
    for d in data
]

count = Counter([d["input"] + d["gold"] for d in data])
for k, v in count.items():
    if v > 1:
        print("input", repr(k))
        print("count", v)
        preds = [d["pred"] for d in data if d["input"] + d["gold"] == k]
        print("preds")
        for p in preds:
            print(" ", p)
        print("unique preds", len(set(preds)))
        print()
