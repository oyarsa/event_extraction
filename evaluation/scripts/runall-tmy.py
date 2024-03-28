#!/usr/bin/env python
import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Config: name, prompt, mode
configurations = [
    # 0
    (
        "Simple QA prompt (valid)",
        "simple",
        "valid",
    ),
    # 1
    (
        "Instructions QA prompt (valid)",
        "instructions_valid",
        "valid",
    ),
    # 2
    (
        "Instructions QA prompt (score)",
        "instructions",
        "score",
    ),
    # 3
    (
        "Instructions QA prompt (valid) v2",
        "instructions_valid_v2",
        "valid",
    ),
    # 4
    (
        "Instructions QA prompt (score) v2",
        "instructions",
        "score",
    ),
]


class ListConfigs(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None = None,
        option_string: str | None = None,
    ):
        print("Configurations:\n")
        for i, (name, _, _) in enumerate(configurations):
            print(f"{i}: {name}")
        parser.exit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list-configs",
        nargs=0,
        action=ListConfigs,
        help="List configurations and their indices",
    )
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--api", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--name", type=str)
    parser.add_argument("--all-data", action=argparse.BooleanOptionalAction)
    parser.add_argument("--configs", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--gpt-config",
        type=str,
        default=Path.home() / ".config/event_extraction/gpt_config.json",
    )
    args, unknown_args = parser.parse_known_args()

    print("Configuration:")
    print(f" n: {'all' if args.all_data else args.n}")
    print(f" api: {args.api}")
    print(f" model: {args.model}")
    print(f" name: {args.name}")

    chosen_configs = [configurations[i] for i in args.configs]
    for name, prompt, mode in chosen_configs:
        print(f">>> {name}")

        # fmt: off
        command = [
            sys.executable, "src/evaluation/gpt.py",
            "../../knowwhy/data/converted/dev.json",
            "--no-print-messages",
            "--system-prompt", "prompts/qa/system.txt",
            "--data-mode", "qa",
            "--openai-config-path", args.gpt_config,
            "--model", args.model,
            "--api-type", args.api,
            "--user-prompt", f"prompts/qa/{prompt}.txt",
            "--result-mode", mode,
        ]
        # fmt: on

        if args.all_data:
            command.append("--all-data")
        else:
            command.extend(["--n", str(args.n)])

        if args.name:
            command.extend(["--run-name", f"{args.name}-{prompt}"])

        command.extend(unknown_args)
        subprocess.run(command, check=True)
        print()


if __name__ == "__main__":
    main()
