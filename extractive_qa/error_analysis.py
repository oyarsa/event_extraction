import json
import sys

input_data_path = sys.argv[1]
prediction_path = sys.argv[2]
# Load input data
with open(input_data_path) as f:
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
with open(prediction_path) as f:
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

with open("error_analysis.json", "w") as f:
    json.dump(result, f, indent=2)
