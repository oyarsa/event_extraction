# pyright: basic
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer


def run(args: list[Any]) -> None:
    try:
        args = [str(arg) for arg in args]
        subprocess.run(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        print()
        print(f"Command failed with return code {e.returncode}")
        print("=== stdout ===")
        print(e.stdout)
        print("=== stderr ===")
        print(e.stderr)
        print()


def main(
    data_dir: Path,
    models_dir: Path,
    max_samples: Optional[int] = None,
    output_dir: Path = Path("output/reward"),
) -> None:
    parameters = [
        (
            "valid",
            [
                "--model_path",
                models_dir / "valid_classifier",
                "--reward_type",
                "valid",
            ],
        ),
        (
            "entailment",
            [
                "--model_path",
                models_dir / "entailment_deberta_tag",
                "--reward_type",
                "entailment",
            ],
        ),
        (
            "mnli",
            [
                "--model_path",
                "microsoft/deberta-large-mnli",
                "--reward_type",
                "entailment",
                "--rewrite",
            ],
        ),
    ]

    results: list[dict[str, Any]] = []

    files = list(data_dir.glob("*.json"))
    count = 0
    n = len(files) * len(parameters)

    for file in files:
        for kind, args in parameters:
            count += 1
            print(f"{count}/{n} File: {file}. Reward: {kind}")

            name = f"{file.stem}_{kind}"
            cli_args = [
                sys.executable,
                "run_reward.py",
                "--data_file",
                file,
                "--run_name",
                name,
                "--output_dir",
                output_dir,
                *args,
            ]
            if max_samples:
                cli_args.extend(["--max_samples", max_samples])
            run(cli_args)

            metrics = json.loads((output_dir / name / "metrics.json").read_text())
            results.append(
                {
                    "file": file.stem.replace("_eval", ""),
                    "reward_type": kind,
                    "reward": metrics["reward"],
                }
            )
            print(json.dumps(metrics, indent=2))

            print("-" * 80)

        print()
        print("#" * 80)
        print()

    df = pd.DataFrame(results)
    df_pivot = df.pivot(
        index="file", columns="reward_type", values="reward"
    ).reset_index()
    print(df_pivot.to_string(index=False))
    df_pivot.to_json("results.json", orient="records")


if __name__ == "__main__":
    typer.run(main)
