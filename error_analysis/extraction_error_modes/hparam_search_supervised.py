import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

from tabulate import tabulate
from tqdm import tqdm

# Hyperparameters to search over
# model_names = ["microsoft/deberta-base"]
# learning_rates = [1e-5]
model_names = ["microsoft/deberta-base", "microsoft/deberta-v3-base"]
learning_rates = [1e-5, 5e-5, 1e-4]
use_passages = [False, True]

# Other (fixed) hyperparameters
data_path = "./prediction_substr.labelled.json"
split_percentage = 0.8
num_epochs = 100
early_stopping_patience = 10
# max_samples = 100
max_samples = None
max_seq_length = 256
batch_size = 32
output_path = "output/hparam"

print(">>> Fixed hyperparameters")
print(f"num_epochs: {num_epochs}")
print(f"early_stopping_patience: {early_stopping_patience}")
print(f"max_samples: {max_samples}")
print(f"max_seq_length: {max_seq_length}")
print(f"batch_size: {batch_size}")
print(f"output_path: {output_path}")
print(f"data_path: {data_path}")
print(f"split_percentage: {split_percentage}")
print()

all_metrics: list[dict[str, Any]] = []
total_combinations = len(model_names) * len(learning_rates) * len(use_passages)

for i, (model_name, learning_rate, use_passage) in enumerate(
    tqdm(product(model_names, learning_rates, use_passages))
):
    print(f"\n\n>>>> {i+1} / {total_combinations} <<<<")
    print(
        f"Model: {model_name}, learning rate: {learning_rate}, use passage:"
        f" {use_passage}\n"
    )

    model_name_for_path = model_name.replace("/", ".")
    output_name = f"{model_name_for_path}_{learning_rate}_{use_passage}"

    args = [
        sys.executable,
        "classifier.py",
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
        "--use_passage",
        str(use_passage),
    ]
    if max_samples is not None:
        args.extend(["--max_samples", str(max_samples)])

    print(" ".join(args[1:]))
    subprocess.run(args, stdout=subprocess.DEVNULL, check=True)

    metric_path = Path(f"{output_path}/{output_name}/metrics.json")
    metrics = json.loads(metric_path.read_text())

    detailed_metrics = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "use_passage": use_passage,
        **metrics,
    }
    all_metrics.append(detailed_metrics)

    print("\nMetrics")
    print(json.dumps(detailed_metrics, indent=2))
    print()


sorted_metrics = sorted(all_metrics, key=lambda x: x["f1"], reverse=True)
print("\n>>>> METRICS <<<<")
print(tabulate(sorted_metrics, headers="keys"))

metrics_path = Path(f"{output_path}/all_metrics.json")
print("\nSaving metrics to", metrics_path)
metrics_path.write_text(json.dumps(sorted_metrics, indent=2))
