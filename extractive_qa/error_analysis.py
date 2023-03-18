import json
import sys
from pathlib import Path

input_data_path = Path(sys.argv[1])
prediction_path = Path(sys.argv[2])
# Load input data
with input_data_path.open() as f:
    input_data_json = json.load(f)

input_data = {
    instance["id"]: {
        "context": instance["context"],
        "cause": instance["cause"]["text"],
        "effect": instance["effect"]["text"],
    }
    for instance in input_data_json["data"]
}

# Load output data
with prediction_path.open() as f:
    predictions = json.load(f)

# Map input data to output data
result = [
    {
        "id": id,
        "context": input_data[id]["context"],
        "cause": {
            "text": input_data[id]["cause"][0],
            "prediction": pred["cause"],
        },
        "effect": {
            "text": input_data[id]["effect"][0],
            "prediction": pred["effect"],
        },
    }
    for id, pred in predictions.items()
]

with Path("error_analysis.json").open("w") as f:
    json.dump(result, f, indent=2)
