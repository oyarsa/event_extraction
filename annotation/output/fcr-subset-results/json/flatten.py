import json
import sys

# Read the JSON file
with open(sys.argv[1]) as f:
    data = json.load(f)

# Flatten the valid subobject
for item in data:
    valid_dict = {v["name"]: v["valid"] for v in item["valid"]}
    item.update(valid_dict)
    del item["valid"]

json.dump(data, sys.stdout, indent=2)
