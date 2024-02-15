#!/usr/bin/env python3
import json

import typer


def main(
    input_path: typer.FileText = typer.Argument(
        ..., help="Path to input data (GPT eval output)."
    ),
    output_path: typer.FileTextWrite = typer.Argument(..., help="Path to output file."),
    threshold: int = typer.Option(
        4, "--threshold", "-n", help="Minimum score to consider an answer valid."
    ),
) -> None:
    input_data = json.load(input_path)

    output_data = [
        {"pred": int(d["gpt_reward"] >= threshold), "gold": int(d["valid"])}
        for d in input_data
    ]

    output_path.write(json.dumps(output_data, indent=2))


if __name__ == "__main__":
    typer.run(main)
