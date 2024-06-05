#!/usr/bin/env python3

"""Run a reward model on a given dataset.


The input is a JSON file with a list of objects with the following fields:
- input (str): the input context
- gold (str): the extraction annotation (tagged)
- output (str) the model output extraction (tagged)

We try to get the model from two possible paths, in order:
- evaluation/output/classifier/{model}-deberta-best
- evaluation/output/classifier/{model}
- treat {model} as the absolute path

The output is saved in self_critique/output/reward/{name}-{model}eval.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from beartype.door import is_bearable


def git_root() -> Path:
    return Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True
        ).strip()
    )


ROOT = git_root()


def validate_file(file: Path) -> None:
    data = json.loads(file.read_text())
    data_keys = {"input", "output", "gold"}
    if not is_bearable(data, list[dict[str, Any]]):
        raise SystemExit("Invalid JSON format. Expected a list of objects.")
    if missing := data_keys - data[0].keys():
        raise SystemExit(f"Invalid JSON format. Missing keys: {missing}.")


def main(
    input_file: Path,
    name: str,
    model: str,
    is_cpu: bool,
    batch_size: int | None,
    prompt: str,
    reward_type: str,
) -> None:
    validate_file(input_file)

    model_path = ROOT / f"evaluation/output/classifier/{model}-deberta-best"
    if not model_path.is_dir():
        model_path = ROOT / f"evaluation/output/classifier/{model}"
    if not model_path.is_dir():
        model_path = Path(model)

    model_name = model_path.name
    output_dir = ROOT / f"self_critique/output/reward/{name}-{model_name}eval"

    if batch_size is None:
        batch_size = int(1e9) if is_cpu else 32

    # fmt: off
    args = [
        sys.executable, ROOT / "self_critique/self_critique/rl/run_reward.py",
        "--model_path", model_path,
        "--data_file", input_file,
        "--output_dir", output_dir,
        "--batch_size", batch_size,
        "--eval_prompt", prompt.upper(),
        "--reward_type", reward_type,
    ]
    # fmt: on
    if is_cpu:
        args.extend(["--device", "cpu"])

    subprocess.run([str(x) for x in args], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", type=Path, help="Path to the input JSON file.")
    parser.add_argument("name", type=str, help="Name of the output.")
    parser.add_argument(
        "--model",
        type=str,
        default="fcr",
        help="Name of the model. Default: %(default)s.",
    )
    parser.add_argument(
        "--cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CPU instead of GPU for evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for the evaluation model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["passage", "gold", "combined"],
        default="combined",
        help="Evaluation prompt mode",
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        choices=["valid", "entailment"],
        default="valid",
        help="Reward type for the reward model. Default: %(default)s.",
    )
    args = parser.parse_args()
    main(
        args.input_file,
        args.name,
        args.model,
        args.cpu,
        args.batch_size,
        args.prompt,
        args.reward_type,
    )
