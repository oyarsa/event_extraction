import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer


def git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


CLASSIFIER_SCRIPT = Path("classifier.py")
EVALUATOR_SCRIPT = git_root() / "agreement" / "calc_more.py"


def run_classifier(config_path: Path, output_path: Path, output_name: str) -> None:
    args = [
        sys.executable,
        str(CLASSIFIER_SCRIPT),
        "--config",
        str(config_path),
        "--output_path",
        str(output_path),
        "--output_name",
        output_name,
    ]
    subprocess.run(args, check=True)


def run_evaluator(metric: str, base_file: Path, eval_file: Path) -> None:
    args = [
        sys.executable,
        str(EVALUATOR_SCRIPT),
        metric,
        str(base_file),
        f"{eval_file},valid",
    ]
    subprocess.run(args, check=True)


def evaluate(human_file: Path, output_file: Path) -> None:
    print()
    print(">>>> EVALUATING")
    for metric in ["agreement", "krippendorff", "spearman", "cohen"]:
        run_evaluator(metric, human_file, output_file)
        print()


def transform_data(data: list[dict[str, Any]]) -> list[dict[str, str]]:
    """
    Transforms the data from the classifier into the format expected by the
    agreement evaluation script.

    The classifier format is:
    - passage: str
    - pred: int
    - annotation: str

    The agreement evaluation format is:
    - input: str
    - reward_label: str
    - gold: str
    """
    return [
        {
            "input": item["passage"],
            "reward_label": "VALID" if item.get("pred") == 1 else "INVALID",
            "gold": item["annotation"],
        }
        for item in data
    ]


def transform(input_file: Path, output_file: Path) -> None:
    input_data = json.loads(input_file.read_text())
    output_data = transform_data(input_data)
    output_file.write_text(json.dumps(output_data, indent=2))


def main(
    dir_name: Path = typer.Argument(help="Path to output directory"),
    run_name: str = typer.Argument(help="Name of the run"),
    config: Path = typer.Argument(help="Path to the config file"),
    human_file: Path = typer.Argument(help="Path to the human base file"),
) -> None:
    "Train classifier and evaluate output for agreement."
    run_path = dir_name / run_name
    input_file = run_path / "test_results.json"
    output_file = run_path / "knowwhy_valid.json"

    run_classifier(config, dir_name, run_name)
    transform(input_file, output_file)
    evaluate(human_file, output_file)


if __name__ == "__main__":
    typer.run(main)
