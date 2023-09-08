import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

from tqdm import tqdm

model_names = ["microsoft/deberta-base", "microsoft/deberta-v3-base"]
learning_rates = [1e-5, 5e-5, 1e-4]

data_path = "./substr_labelled.json"
split_percentage = 0.8
num_epochs = 100
early_stopping_patience = 5
max_samples = 16
max_seq_length = 256
batch_size = 8
config_file = "classifier_config.json"
output_path = "output/hparam"

output_names: list[str] = []

for i, (model_name, learning_rate) in enumerate(
    tqdm(product(model_names, learning_rates))
):
    print(f"\n\n>>>> {i+1} / {len(model_names) * len(learning_rates)} <<<<")
    print(f"Model {model_name} and learning rate {learning_rate}\n")

    model_name_for_path = model_name.replace("/", ".")
    output_name = f"{model_name_for_path}_{learning_rate}"
    output_names.append(output_name)

    args = [
        sys.executable,
        "classifier.py",
        "--config",
        config_file,
        "--max_samples",
        str(max_samples),
        "--model_name",
        model_name,
        "--learning_rate",
        str(learning_rate),
        "--output_path",
        output_path,
        "--output_name",
        output_name,
        "--batch_size",
        str(batch_size),
        "--num_epochs",
        str(num_epochs),
        "--early_stopping_patience",
        str(early_stopping_patience),
        "--max_seq_length",
        str(max_seq_length),
        "--data_path",
        data_path,
        "--split_percentage",
        str(split_percentage),
    ]
    print(" ".join(args[1:]))
    subprocess.run(args, stdout=subprocess.DEVNULL, check=True)

all_metrics: list[dict[str, Any]] = []
for output_name in output_names:
    metric_path = Path(f"{output_path}/{output_name}/metrics.json")
    metrics = json.loads(metric_path.read_text())
    all_metrics.append({**metrics, "name": output_name})

print("\n>>>> METRICS <<<<")
print(json.dumps(all_metrics, indent=2))

metrics_path = Path(f"{output_path}/all_metrics.json")
print("\nSaving metrics to", metrics_path)
metrics_path.write_text(json.dumps(all_metrics, indent=2))
