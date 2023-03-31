"""
Stdin input from the output_from_log.nu script.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from metrics import (  # noqa: E402
    MetricPrediction,
    MetricReference,
    StructureFormat,
    calculate_metrics,
    parse_instance,
)


def get_relation(inp: str, format: StructureFormat) -> str:
    _entities, relation = parse_instance(inp, format)
    return relation or "cause"


def find_original_data(
    og_data: list[dict[str, str]], log_data: list[dict[str, str]]
) -> list[dict[str, str]]:
    output = og_data.copy()
    for log_entry in log_data:
        for out in output:
            if log_entry["input"] == out["context"]:
                out["prediction"] = log_entry["output"]
    return output


def parse_log_output(log_file: Path) -> list[dict[str, str]]:
    with log_file.open() as f:
        data = (json.loads(line) for line in f)
        return [
            {
                "input": d["params"]["messages"][-1]["content"],
                "output": d["response"]["choices"][0]["message"]["content"],
            }
            for d in data
        ]


def get_classification_results(
    outputs: list[MetricPrediction],
    inputs: list[MetricReference],
    mode: StructureFormat,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for pred, refer in zip(outputs, inputs):
        assert pred["id"] == refer["id"]
        _, pred_relation = parse_instance(pred["prediction_text"], mode)
        _, ref_relation = parse_instance(refer["answers"], mode)
        assert pred_relation and ref_relation

        assert ref_relation == refer["question_type"], (
            "Extracted reference relation does not match the question type."
            f" {ref_relation} != {refer['question_type']}"
        )
        output.append({"gold": ref_relation, "pred": pred_relation})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("log_file", type=Path)
    parser.add_argument("--pred-file", type=Path, default="predictions.json")
    args = parser.parse_args()

    log_data = parse_log_output(args.log_file)
    og_data = json.loads(args.data.read_text())["data"]

    dataset = find_original_data(og_data, log_data)

    format = (
        StructureFormat.TAGS
        if re.match(r"\[.*\]", log_data[0]["input"])
        else StructureFormat.LINES
    )
    inputs = [MetricReference(**d) for d in dataset]
    outputs = [
        MetricPrediction(id=d["id"], prediction_text=d["prediction"])
        for d in dataset
        if "prediction" in d
    ]
    metrics = calculate_metrics(
        outputs,
        inputs,
        mode=format,
    )
    print(json.dumps(metrics, indent=2))

    classfication_results = get_classification_results(outputs, inputs, mode=format)
    args.pred_file.write_text(json.dumps(classfication_results, indent=2))


if __name__ == "__main__":
    main()
