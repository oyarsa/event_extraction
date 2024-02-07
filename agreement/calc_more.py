import argparse
import json
import os.path
import sys
from dataclasses import dataclass
from typing import Any

import krippendorff
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


@dataclass
class Result:
    context: str
    gold: str
    base_valid: bool
    model_valid: bool


def convert_model(data: dict[str, Any]) -> dict[str, str]:
    # sourcery skip: default-get
    """Convert model output to match base data format.

    The data is expected to be in the following format:
    - 'context' or 'input': the context or input.
    - 'answers' or 'gold': the gold answer.
    - 'reward_label': the model's output.

    All values are strings.
    """
    return {
        # The explicit check is necessary because 'input' and 'gold' might not exist,
        # which would break the 'get' version
        "input": data["context"] if "context" in data else data["input"],
        "gold": data["answers"] if "answers" in data else data["gold"],
        "reward_label": data["reward_label"].casefold().strip(),
    }


def calculate_metric(metric: str, x: list[Any], y: list[Any]) -> float:
    x = [int(value) for value in x]
    y = [int(value) for value in y]

    match metric:
        case "agreement":
            return sum(a == b for a, b in zip(x, y)) / len(x)
        case "krippendorff" | "kdf":
            return krippendorff.alpha([x, y], level_of_measurement="nominal")
        case "spearman":
            return spearmanr(x, y)[0]
        case "cohen":
            return cohen_kappa_score(x, y)
        case _:
            raise ValueError(f"Unknown metric: {metric}")


def calculate_metrics(
    metric: str,
    base: list[dict[str, Any]],
    model: list[dict[str, str]],
    true_class: str,
) -> tuple[float, float]:
    matches: list[Result] = []

    for b in base:
        for m in model:
            if b["input"] == m["input"] and b["gold"] == m["gold"]:
                matches.append(
                    Result(
                        context=b["input"],
                        gold=b["gold"],
                        base_valid=b["valid"],
                        model_valid=m["reward_label"] == true_class,
                    )
                )
                break

    if not matches:
        return 0.0, 0.0

    model_valid = sum(r.model_valid for r in matches) / len(matches)
    metric_val = calculate_metric(
        metric, [r.base_valid for r in matches], [r.model_valid for r in matches]
    )

    return model_valid, metric_val


def load_json(file_path: str) -> list[dict[str, Any]]:
    with open(file_path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python calc.py <metric> <base_path> <model_path,true_class> [<model_path,true_class> ...]"
        )
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("metric", help="metric to calculate")
    parser.add_argument("base_path", help="path to base data")
    parser.add_argument("models", nargs="+", help="path to model data")
    args = parser.parse_args()

    base_data = load_json(args.base_path)

    # Table Header
    print(f"{'Model File':<30} {'Valid':<10} {args.metric.capitalize():<15}")

    for arg in args.models:
        model_path, true_class = arg.split(",")
        model_data = [convert_model(d) for d in load_json(model_path)]
        valid, metric = calculate_metrics(
            args.metric, base_data, model_data, true_class.casefold().strip()
        )

        print(f"{os.path.basename(model_path):<30} {valid:<10.4f} {metric:.4f}")


if __name__ == "__main__":
    main()
