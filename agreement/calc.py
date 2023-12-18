import json
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    context: str
    gold: str
    base_valid: bool
    model_valid: bool


def convert_model(data: dict[str, Any]) -> dict[str, str]:
    "Convert model output to match base data format."
    return {
        "input": data.get("context", data["input"]),
        "gold": data.get("answers", data["gold"]),
        "reward_label": data["reward_label"].casefold().strip(),
    }


def calculate_metrics(
    base: list[dict[str, Any]], model: list[dict[str, str]], true_class: str
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
    agreement = sum(r.base_valid == r.model_valid for r in matches) / len(matches)
    return model_valid, agreement


def load_json(file_path: str) -> list[dict[str, Any]]:
    with open(file_path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python calc.py <base_path> <model_path,true_class> [<model_path,true_class> ...]"
        )
        sys.exit(1)

    base_path = sys.argv[1]
    base_data = load_json(base_path)

    # Table Header
    print(f"{'Model File':<30} {'Valid':<10} {'Agreement':<15}")

    for arg in sys.argv[2:]:
        model_path, true_class = arg.split(",")
        model_data = [convert_model(d) for d in load_json(model_path)]
        valid, agreement = calculate_metrics(
            base_data, model_data, true_class.casefold().strip()
        )

        # Print each row of the table
        print(f"{model_path:<30} {valid:<10.2%} {agreement:.2%}")


if __name__ == "__main__":
    main()
