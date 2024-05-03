#!/usr/bin/env python3
# pyright: basic
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer


def main(
    env: str = "dev",
    mode: str = "lines",
    keys_file: str = os.path.expanduser("~/.config/event_extraction/keys.json"),
    key: str = "azure",
    run_name: Optional[str] = None,
    data: Optional[Path] = None,
    user_prompt: int = 1,
    system_prompt: int = 1,
    model: str = "gpt-3.5-turbo",
    envs: bool = False,
) -> None:
    files = {
        "test": ("extraction_test_full.json", "extraction_examples.json"),
        "full": ("extraction_dev_full.json", "extraction_examples.json"),
        "exp": ("extraction_dev_100.json", "extraction_examples.json"),
        "dev": ("extraction_dev_10.json", "extraction_examples_3.json"),
        "debug": ("extraction_dev_2.json", "extraction_examples_3.json"),
    }

    if envs:
        fmt = "{:<7} {:<27} {:<20}"
        print(fmt.format("env", "input_file", "examples_file"))
        print(fmt.format("-" * 7, "-" * 27, "-" * 20))
        for name, (input_file, examples_file) in files.items():
            print(fmt.format(name, input_file, examples_file))
        return

    run_name = run_name or datetime.now().isoformat()
    if mode not in ["lines", "tags"]:
        raise ValueError(f"Invalid mode {mode}")

    available_models = ["gpt-3.5-turbo", "gpt-4"]
    if not any(model.startswith(x) for x in available_models):
        raise ValueError(
            f"Invalid model {model}. Options: {', '.join(available_models)} and derived."
        )

    input_file, examples_file = map(Path, files[env])

    is_gpt4 = model.startswith("gpt-4")
    if is_gpt4:
        system_prompt = 1
        if mode == "lines":
            user_prompt = 1
        elif mode == "tags":
            user_prompt = 2
    elif not model.startswith("gpt-3.5"):
        raise ValueError(f"Invalid model {model}")

    input_file = data or Path("data") / "extraction" / mode / input_file

    # Go to level above script's directory
    # This should be the chatgpt project root
    # TODO: Use git root instead
    os.chdir(Path(__file__).parent.parent)

    output_dir = Path("output") / "extraction" / env / mode / model / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    args = [
        sys.executable,
        "extraction.py",
        keys_file,
        key,
        "--model",
        model,
        "--input",
        input_file,
        "--output",
        output_dir / "output.json",
        "--metrics-path",
        output_dir / "metrics.json",
        "--args-path",
        output_dir / "args.json",
        "--log-file",
        output_dir / "log.jsonl",
        "--mode",
        mode,
        "--prompt",
        user_prompt,
        "--sys-prompt",
        system_prompt,
    ]
    if not is_gpt4:
        args += ["--examples", Path("data") / "extraction" / mode / examples_file]

    cmd = [str(x) for x in args]
    subprocess.run(cmd, check=True)

    args_classes = [
        sys.executable,
        "get_classes.py",
        output_dir / "output.json",
        "--mode",
        mode,
    ]
    cmd_classes = [str(x) for x in args_classes]
    subprocess.run(cmd_classes, check=True)


if __name__ == "__main__":
    typer.run(main)
