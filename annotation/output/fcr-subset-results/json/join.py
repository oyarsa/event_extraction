import json
import sys
from collections import defaultdict


def getname(d: dict[str, str]) -> str:
    name = d["filename"]
    return name if name in ["hanqi", "siying", "yanzheng"] else "italo"


with open(sys.argv[1]) as f:
    ad = json.load(f)
with open(sys.argv[2]) as f:
    bd = json.load(f)
data = ad + bd


results = defaultdict(list)
for d in data:
    results[d["id"]].append({"valid": d["valid"], "name": getname(d)})

# remove items in results with less than 3 entries
results = {k: v for k, v in results.items() if len(v) == 4}

print(json.dumps([{"id": k, "valid": v} for k, v in results.items()], indent=2))
