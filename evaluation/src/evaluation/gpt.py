#!/usr/bin/env python3
# pyright: basic
import inspect
import json
import math
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openai
import typer
from tqdm import tqdm

from evaluation import log, metrics
from evaluation.gpt_common import (
    MODEL_COSTS,
    Chain,
    DataMode,
    FilterStatus,
    GptResult,
    Message,
    ModelResult,
    ResultMode,
    calculate_metrics,
    confusion_matrix,
    count_steps,
    format_result,
    hash_file,
    init_client,
    load_chains,
    logger,
    make_chain_prompt,
    make_messages,
    most_common,
    reformat_output,
    render_messages,
    run_gpt,
    split_data,
)


def print_model_messages(
    system_prompt: str,
    user_prompt: str,
    chain_prompt: str | None,
    msg: Message,
    gpt_result: GptResult,
    print_chains: bool,
    result: int,
) -> None:
    output = [
        "-" * 80,
        system_prompt,
        "-" * 80,
        user_prompt,
    ]
    if chain_prompt and print_chains and gpt_result.filtered is FilterStatus.UNFILTERED:
        output.extend(["-" * 80, chain_prompt])
    if gpt_result.filtered is FilterStatus.UNFILTERED:
        output.extend(
            [
                "-" * 80,
                msg.gpt_msg,
                "-" * 80,
                "\nGPT: ",
                *(format_result(i, r) for i, r in enumerate(gpt_result.results)),
                f"\nParsed: {result}",
                f"NOT SENT {'~' * 50}",
                msg.answer_msg,
                "*" * 80,
                "",
            ]
        )
    else:
        output.append("Filtered output.")
    logger.info("\n".join(output))


def run_model(
    messages: list[Message],
    model: str,
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    print_messages: bool,
    result_mode: ResultMode,
    temperature: float,
    num_samples: int | None,
    chains: list[Chain],
    print_chains: bool,
    debug: bool,
) -> ModelResult:
    results: defaultdict[tuple[int, int], int] = defaultdict(int)
    total_cost = 0
    model_used: str | None = None
    filtered = 0
    output_data: list[dict[str, Any]] = []
    chain_prompt = make_chain_prompt(chains, result_mode) if chains else None

    for msg in tqdm(messages):
        gpt_result = run_gpt(
            client=client,
            model=model,
            message=msg.gpt_msg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            num_samples=num_samples,
            chain_prompt=chain_prompt,
            debug=debug,
        )
        total_cost += gpt_result.cost
        model_used = gpt_result.model_used

        if gpt_result.filtered is FilterStatus.FILTERED:
            filtered += 1
            logger.info(f"Content filtered. Occurrences: {filtered}.")

            if print_messages:
                logger.info(
                    f"  Prompt:\n{render_messages(json.loads(gpt_result.results[0]))}"
                )

        parsed_results = [result_mode.extract_result(r) for r in gpt_result.results]
        chain_lengths = [count_steps(r) for r in gpt_result.results]
        result = most_common(parsed_results)
        results[msg.gold_label, result] += 1

        output_data.append(
            msg.item
            | {
                "gpt_result": result,
                "gpt_outputs": gpt_result.results,
                "chain_lengths": chain_lengths,
                "chain_results": parsed_results,
            }
        )

        if print_messages:
            print_model_messages(
                system_prompt,
                user_prompt,
                chain_prompt,
                msg,
                gpt_result,
                print_chains,
                result,
            )

    logger.info(f"Total filtered: {filtered}")
    return ModelResult(output_data, results, total_cost, model_used or "<unknown>")


def extract_lines(text: str) -> str:
    pattern = r"(?i)(?:explanation|analysis):\s*(.*?)\s*(?:validity|valid|score):"
    if match := re.search(pattern, text, re.DOTALL):
        return match[1].strip()
    return ""


def main(
    file: Path = typer.Argument(
        ...,
        help="Path to the json file containing the data (list of objects with keys"
        " 'input', 'output', 'gold', 'valid').",
        exists=True,
    ),
    system_prompt_path: Path = typer.Option(
        ...,
        "--system-prompt",
        help="Path to the system prompt file.",
        exists=True,
    ),
    user_prompt_path: Path = typer.Option(
        ...,
        "--user-prompt",
        help="Path to the user prompt file.",
        exists=True,
    ),
    n: int = typer.Option(
        10,
        help="Number of examples to run. Should be even. If not, the number is rounded"
        " up.",
    ),
    rand: bool = typer.Option(
        True,
        help="Whether to shuffle the data before selecting n examples.",
    ),
    seed: int = typer.Option(
        0,
        help="Random seed for shuffling the data.",
    ),
    openai_config_path: Path = typer.Option(
        Path("config.json"),
        help="Path to the file containing the OpenAI API keys.",
        exists=True,
    ),
    api_type: str = typer.Option(
        "openai",
        help="API type, defaults to 'openai'.",
    ),
    model: str = typer.Option(
        "gpt-4-0125-preview",
        help=f"Which GPT model to use. Options: {tuple(MODEL_COSTS)}.",
    ),
    print_messages: bool = typer.Option(
        False,
        help="Whether to print messages including the prompt, context, gold, and"
        " prediction.",
    ),
    debug: bool = typer.Option(
        False,
        help="Whether to print debug messages.",
    ),
    output_dir: Path = typer.Option(
        Path("output") / "gpt",
        help="Directory to save the output files.",
        writable=True,
        file_okay=False,
    ),
    run_name: Optional[str] = typer.Option(
        None,
        help="Name of the run. Generated from the model and prompts if not provided.",
    ),
    tag: Optional[str] = typer.Option(
        None,
        help="Tag to add to the run name.",
    ),
    all_data: bool = typer.Option(
        False,
        help="Whether to run the entire dataset. If true, n is ignored.",
    ),
    data_mode: DataMode = typer.Option(DataMode.EXTRACTION, help="Data mode."),
    result_mode: ResultMode = typer.Option(ResultMode.SCORE, help="Result mode."),
    temperature: float = typer.Option(
        0.0, help="Temperature for the GPT model.", min=0.0, max=1.0
    ),
    num_samples: Optional[int] = typer.Option(
        None, help="Number of samples to generate."
    ),
    chains_path: Optional[Path] = typer.Option(
        None,
        help="Path to the file containing the chains.",
    ),
    print_chains: bool = typer.Option(
        False,
        help="Whether to the prompt Chain of Thought chains.",
    ),
) -> None:
    "Run a GPT model on the given data and evaluate the results."

    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {tuple(MODEL_COSTS)}")
    if num_samples is not None and temperature == 0:
        raise ValueError("Number of samples is set but temperature is 0.")
    if (num_samples is None or num_samples == 1) and temperature != 0:
        raise ValueError("Temperature is set but number of samples is not.")

    git_hash = log.get_current_commit_shorthash()
    params = log.get_func_params(inspect.currentframe(), serialise=True)
    reproduction_info = {
        "command": sys.argv,
        "data_hash": hash_file(file),
        "git_hash": git_hash,
        "params": params,
    }

    if run_name is None:
        model_name = model if api_type == "openai" else api_type
        run_name = (
            f"{model_name}-sys_{system_prompt_path.name}-user_{user_prompt_path.name}"
        )
        if all_data:
            run_name += "-all"
        else:
            run_name += f"-n{n}"

        if rand:
            run_name += f"-rand{seed}"
        if num_samples is not None:
            run_name += f"-k{num_samples}"
        if temperature != 0:
            run_name += f"-t{temperature}"
        if chains_path:
            run_name += "-chains"
    if tag:
        run_name += f"-{tag}"

    ts = datetime.now(timezone.utc).isoformat()
    output_path = output_dir / run_name / ts
    output_path.mkdir(exist_ok=True, parents=True)
    log.setup_logger(logger, output_path)

    logger.info(f"Git hash: {git_hash}")
    logger.info(f"Run name: {run_name}")

    openai_config: dict[str, dict[str, str]] = json.loads(
        openai_config_path.read_text()
    )
    client, config_model = init_client(api_type, openai_config)
    model = config_model or model
    logger.info(f"Model: {model}")

    data = json.loads(file.read_text())
    if rand:
        random.seed(seed)
        random.shuffle(data)

    if all_data:
        sampled_data = data
    else:
        # Get same number of valid and invalid examples
        valids, invalids = split_data(data, math.ceil(n / 2))
        sampled_data = valids + invalids

    messages = make_messages(sampled_data, data_mode, result_mode)
    system_prompt = system_prompt_path.read_text()
    user_prompt = user_prompt_path.read_text()
    chains = load_chains(chains_path) if chains_path else []

    model_result = run_model(
        messages,
        model,
        client,
        system_prompt,
        user_prompt,
        print_messages,
        result_mode,
        temperature,
        num_samples,
        chains,
        print_chains=print_chains,
        debug=debug,
    )
    formatted_output = reformat_output(model_result.output_data, result_mode)
    metrics_ = calculate_metrics(formatted_output, result_mode)

    (output_path / "full_output.json").write_text(
        json.dumps(model_result.output_data, indent=2)
    )
    (output_path / "results.json").write_text(json.dumps(formatted_output, indent=2))
    (output_path / "metrics.json").write_text(json.dumps(metrics_, indent=2))
    (output_path / "reproduction.json").write_text(
        json.dumps(reproduction_info, indent=2)
    )
    metrics.report_metrics(
        logger, metrics_, "GPT", mse=result_mode is ResultMode.LIKERT
    )

    with Path("cost.csv").open("a") as f:
        f.write(f"{ts},{model_result.total_cost}\n")

    logger.info(f"\n{confusion_matrix(model_result.results)}\n")
    logger.info(f"Total cost: ${model_result.total_cost}")
    logger.info(f"Model used: {model_result.model_used}")


if __name__ == "__main__":
    typer.run(main)
