#!/usr/bin/env python

import argparse
import subprocess
import sys

# Config: name, prompt, mode
configurations = [
    (
        "Simple QA prompt (valid)",
        "simple_qa",
        "valid",
    ),
    (
        "Instructions QA prompt (valid)",
        "instructions_qa_valid",
        "valid",
    ),
    (
        "Instructions QA prompt (score)",
        "instructions_qa",
        "score",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--api", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--name", type=str)
    parser.add_argument("--all-data", action=argparse.BooleanOptionalAction)
    args, unknown_args = parser.parse_known_args()

    print("Configuration:")
    print(f" n: {'all' if args.all_data else args.n}")
    print(f" api: {args.api}")
    print(f" model: {args.model}")
    print(f" name: {args.name}")

    for name, prompt, mode in configurations:
        print(f">>> {name}")

        # fmt: off
        command = [
            sys.executable, "src/evaluation/gpt.py",
            "../../knowwhy/data/converted/dev.json",
            "--no-print-messages",
            "--system-prompt", "qa",
            "--data-mode", "qa",
            "--openai-config-path", "gpt_config.json",
            "--model", args.model,
            "--api-type", args.api,
            "--user-prompt", prompt,
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
