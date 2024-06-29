import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from itertools import product
from typing import Any


def run_evaluation(
    n: int,
    api_type: str,
    openai_config_path: str,
    system_prompt: str,
    user_prompt: str,
    data_mode: str,
    result_mode: str,
    run_name: str,
    chains_path: str | None = None,
    temperature: float | None = None,
    num_samples: int | None = None,
) -> None:
    # fmt: off
    command = [
        sys.executable, "src/evaluation/gpt.py",
        "../../knowwhy/data/converted/dev.json",
        "--n", n,
        "--api-type", api_type,
        "--openai-config-path", openai_config_path,
        "--system-prompt", system_prompt,
        "--user-prompt", user_prompt,
        "--data-mode", data_mode,
        "--no-print-messages",
        "--no-debug",
        "--result-mode", result_mode,
        "--output-dir", "output/experiments",
        "--run-name", run_name,
    ]
    # fmt: on

    if chains_path:
        command.extend(["--chains-path", chains_path])
    if temperature:
        command.extend(["--temperature", temperature])
    if num_samples:
        command.extend(["--num-samples", num_samples])

    subprocess.run([str(c) for c in command], check=False)


prompt_configs: list[tuple[str, str]] = [
    ("Instructions Valid v1", "instructions_valid.txt"),
    ("Instructions Valid v2", "instructions_valid_v2.txt"),
    ("Instructions Valid v1 CoT", "instructions_valid_cot.txt"),
    ("Instructions Valid v2 CoT", "instructions_valid_cot_v2.txt"),
]


class ListPrompts(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None = None,
        option_string: str | None = None,
    ) -> None:
        print("Prompt configurations:")
        for i, (name, _) in enumerate(prompt_configs):
            print(f"{i}: {name}")

        print(
            "\nUse the indices with the --prompt-configs flag to select configurations."
        )
        print("E.g. `--prompt-configs 0 1`\n")
        parser.exit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GPT evaluations with different configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=2,
        help="Number of examples.",
    )
    parser.add_argument(
        "-k",
        "--num-samples",
        type=int,
        default=3,
        help="Number of samples.",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        type=float,
        default=0.7,
        help="Model temperature.",
    )
    parser.add_argument(
        "--openai-config-path",
        type=str,
        default="~/.config/event_extraction/gpt_config.json",
        help="Path to the OpenAI configuration file.",
    )
    parser.add_argument("--api-type", type=str, default="azure35", help="API type.")
    parser.add_argument(
        "--chain-path",
        type=str,
        required=True,
        help="Path to the chains file.",
    )
    parser.add_argument(
        "--prompt-configs",
        type=int,
        nargs="+",
        default=range(len(prompt_configs)),
        choices=range(len(prompt_configs)),
        help="Indices of prompt configurations to use. See --list-prompts.",
    )
    parser.add_argument(
        "--chains-configs",
        type=int,
        nargs="+",
        default=[0, 1],
        choices=[0, 1],
        help="Chains configurations.",
    )
    parser.add_argument(
        "--sampling-configs",
        type=int,
        nargs="+",
        default=[0, 1],
        choices=[0, 1],
        help="Sampling configurations.",
    )
    parser.add_argument(
        "--list-prompts",
        nargs=0,
        action=ListPrompts,
        help="List prompts and their indices",
    )
    args = parser.parse_args()

    n = args.num_examples
    k = args.num_samples
    t = args.temperature

    openai_config_path = os.path.expanduser(args.openai_config_path)
    api_type = args.api_type
    data_mode = "qa"
    result_mode = "valid"

    if api_type == "azure35":
        model = "GPT-3.5"
    elif api_type == "azure4":
        model = "GPT-4"
    else:
        raise ValueError(f"Invalid API type: {api_type}")

    chains_configs = [bool(i) for i in args.chains_configs]
    sampling_configs = [bool(i) for i in args.sampling_configs]

    selected_prompt_configs = [prompt_configs[i] for i in args.prompt_configs]
    configurations = list(
        product(selected_prompt_configs, chains_configs, sampling_configs)
    )
    print(f"Number of configurations: {len(configurations)}\n")

    for i, ((prompt_version, user_prompt_file), use_chains, use_sampling) in enumerate(
        configurations
    ):
        user_prompt = f"./prompts/qa/{user_prompt_file}"
        chains_path = args.chain_path if use_chains else None

        prompt_name = prompt_version.lower().replace(" ", "_")
        run_name = (
            f"{model.lower()}-{prompt_name}"
            f"-{'chains' if use_chains else 'no_chains'}"
            f"-{'sampling' if use_sampling else 'no_sampling'}"
            f"{f'-k{k}-t{t}' if use_sampling else ''}"
        )

        print(
            f">>> #{i + 1} {model} {prompt_version} - "
            f"{'With chains' if use_chains else 'No chains'} - "
            f"{f'With sampling (K: {k} + T: {t})' if use_sampling else 'No sampling'}"
            f" - Examples: {n}"
        )

        run_evaluation(
            n,
            api_type,
            openai_config_path,
            "./prompts/qa/system.txt",
            user_prompt,
            data_mode,
            result_mode,
            run_name,
            chains_path,
            t if use_sampling else None,
            k if use_sampling else None,
        )
        print()


if __name__ == "__main__":
    main()
