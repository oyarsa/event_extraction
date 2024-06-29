#!/usr/bin/env python3
"Convert ChatGPT's extraction output to the format expected by the evaluation script."

# pyright: basic
import json
import sys
from pathlib import Path

if len(sys.argv) != 3:
    print(f"Usage: {Path(sys.argv[0]).name} <file1> <file2>")
    sys.exit(1)

data1 = json.loads(Path(sys.argv[1]).read_text())
data2 = json.loads(Path(sys.argv[2]).read_text())
output: list[dict[str, str]] = []

for d1, d2 in zip(data1, data2):
    assert d1["input"] == d2["input"], f"Inputs differ: {d1['input']} != {d2['input']}"

    output.append({
        "input": d1["input"],
        "gold": d1["gold"],
        "output1": d1["output"],
        "output2": d2["output"],
    })

json.dump(output, sys.stdout)
