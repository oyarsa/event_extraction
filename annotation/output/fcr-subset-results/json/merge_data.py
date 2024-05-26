import sys
import json

# Read the first JSON file
with open(sys.argv[1]) as f:
    data1 = json.load(f)

# Read the second JSON file
with open(sys.argv[2]) as f:
    data2 = json.load(f)

# Create a dictionary to map IDs to objects from the second file
id_map = {item["id"]: item for item in data2}

# Merge the data from the second file into the first file based on matching IDs
for item in data1:
    id = item["id"]
    if id in id_map:
        item["input"] = id_map[id]["input"]
        item["output"] = id_map[id]["output"]
        item["gold"] = id_map[id]["gold"]

# Write the merged data to a new JSON file
print(json.dumps(data1, indent=2))
