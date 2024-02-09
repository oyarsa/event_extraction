import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

from tabulate import tabulate
from tqdm import tqdm

# Hyperparameters to search over
model_names = ["google/flan-t5-small", "google/flan-t5-base", "t5-base"]
learning_rates = [1e-5, 5e-5, 1e-4, 1e-3]
max_seq_lengths = [256]
train_data_paths = [
    "./training/weakly.json",
    "./training/resampled.json",
    "./training/transformed/weakly.json",
]

# Other (fixed) hyperparameters
# train_data_path = "./training/weakly.json"
eval_data_path = "./evaluation/all_labelled.json"
test_data_path = "./prediction/all_labelled.json"
infer_data_path = "./prediction/all_labelled.json"
num_epochs = 2
num_epochs = 100
early_stopping_patience = 50
# max_samples = 100
max_samples = None
output_path = "output/hparam_weak_t5_seq"
do_train = True
do_evaluation = True
do_test = False
do_inferece = False

print(">>> Fixed hyperparameters")
print(f"{num_epochs=}")
print(f"{early_stopping_patience=}")
print(f"{max_samples=}")
print(f"{output_path=}")
# print(f"{train_data_path=}")
print(f"{eval_data_path=}")
print(f"{test_data_path=}")
print(f"{infer_data_path=}")
print()

all_metrics: list[dict[str, Any]] = []
total_combinations = (
    len(model_names)
    * len(learning_rates)
    * len(max_seq_lengths)
    * len(train_data_paths)
)

for i, (model_name, learning_rate, max_seq_length, train_data_path) in enumerate(
    tqdm(product(model_names, learning_rates, max_seq_lengths, train_data_paths))
):
    if "small" in model_name:
        batch_size = 32
    else:
        batch_size = 16
    train_name = {
        "./training/weakly.json": "standard",
        "./training/resampled.json": "resampled",
        "./training/transformed/weakly.json": "transformed",
    }[train_data_path]

    print(f"\n\n>>>> {i+1} / {total_combinations} <<<<")
    print(
        f"{model_name=} {learning_rate=} {max_seq_length=} {batch_size=} {train_name=}"
    )

    model_name_for_path = model_name.replace("/", ".")
    output_name = f"{model_name_for_path}_{learning_rate}_{max_seq_length}_{train_name}"

    args = [
        sys.executable,
        "classifier_t5_seq.py",
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
        "--train_data_path",
        train_data_path,
        "--eval_data_path",
        eval_data_path,
        "--test_data_path",
        test_data_path,
        "--infer_data_path",
        infer_data_path,
        "--do_train",
        str(do_train),
        "--do_evaluation",
        str(do_evaluation),
        "--do_test",
        str(do_test),
        "--do_inference",
        str(do_inferece),
    ]
    if max_samples is not None:
        args.extend(["--max_samples", str(max_samples)])

    print(" ".join(args[1:]))
    start = time.time()
    subprocess.run(args, stdout=subprocess.DEVNULL, check=True)

    metric_path = Path(f"{output_path}/{output_name}/eval_metrics.json")
    metrics = json.loads(metric_path.read_text())

    detailed_metrics = {
        "model_name": model_name,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "time_elapsed_s": time.time() - start,
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
