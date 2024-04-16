# pyright: basic
import inspect
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import openai
import typer
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from evaluation import log
from evaluation.gpt import (
    MODEL_COSTS,
    FilterStatus,
    GptResult,
    calculate_cost,
    hash_file,
    init_client,
    make_chat_request,
)

logger = logging.getLogger("evaluation.chain_generation")


def run_gpt(
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    message: str,
    print_messages: bool,
) -> GptResult:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "user", "content": message},
    ]

    response, filtered = make_chat_request(
        client,
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=0,
    )
    result = response.choices[0].message.content or "<empty>"
    if result == "<empty>":
        logger.warning("Empty response. Prompt:")
        logger.warning(user_prompt)

    if print_messages:
        heading = "=" * 15
        out = [
            "",
            *(
                f"{msg['role'].upper()}\n{heading}\n{msg.get('content')}"
                for msg in messages
            ),
            f"GPT RESPONSE\n{heading}\n{result}",
            "*" * 80,
            "",
        ]
        logger.info("\n".join(out))

    cost = calculate_cost(model, response)

    return GptResult(
        results=[result],
        cost=cost,
        model_used=response.model,
        filtered=filtered,
    )


@dataclass
class Entry:
    input: str
    output: str
    gold: str
    valid: bool
    score: int


def clean_line(line: str) -> str:
    line = line.strip()
    return re.sub(r"Step (\d+):", r"\1.", line)


def clean_chain(chain: str) -> str:
    return "\n".join(clean_line(line) for line in chain.splitlines() if line.strip())


def is_filtered(
    client: openai.OpenAI,
    model: str,
    entry: Entry,
    system_prompt: str,
    user_prompt: str,
    print_messages: bool,
) -> GptResult:
    message = f"{entry.input}\nanswer: {entry.output}"
    return run_gpt(client, model, system_prompt, user_prompt, message, print_messages)


def run_filter(
    client: openai.OpenAI,
    model: str,
    data: list[Entry],
    system_prompt: str,
    user_template: str,
    print_results: bool,
) -> tuple[list[Entry], float]:
    results: list[Entry] = []
    total_cost = 0

    for d in tqdm(data):
        result = is_filtered(
            client, model, d, system_prompt, user_template, print_results
        )
        if result.filtered is FilterStatus.UNFILTERED:
            results.append(d)
        else:
            logger.warning("Filtered content detected.")
            logger.warning(f"Input: {d.input}")
            logger.warning(f"Output: {d.output}")
        total_cost += result.cost

    return results, total_cost


def main(
    file: Path,
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
        "gpt-3.5-turbo",
        help=f"Which GPT model to use. Options: {tuple(MODEL_COSTS)}.",
    ),
    output_dir: Path = typer.Option(
        Path("output") / "chain_generation",
        help="Directory to save the output files.",
        writable=True,
        file_okay=False,
    ),
    run_name: Optional[str] = typer.Option(
        None,
        help="Name of the run. Generated from the model and prompts if not provided.",
    ),
    print_messages: bool = typer.Option(
        False,
        "--print-messages",
        help="Print the generated chains.",
    ),
    system_prompt_path: Path = typer.Option(
        ...,
        "--system-prompt",
        help="Path to the system prompt file.",
        exists=True,
    ),
    user_template_path: Path = typer.Option(
        ...,
        "--user-template",
        help="Path to the user template file.",
        exists=True,
    ),
    n: Optional[int] = typer.Option(None, "-n", help="Number of chains to generate."),
) -> None:
    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {tuple(MODEL_COSTS)}")

    git_hash = log.get_current_commit_shorthash()
    params = log.get_func_params(inspect.currentframe(), serialise=True)
    reproduction_info = {
        "command": sys.argv,
        "data_hash": hash_file(file),
        "git_hash": git_hash,
        "params": params,
    }
    openai_config: dict[str, dict[str, str]] = json.loads(
        openai_config_path.read_text()
    )
    client, config_model = init_client(api_type, openai_config)
    model = config_model or model
    logger.info(f"Model: {model}")

    run_name = run_name or file.stem

    output_path = output_dir / run_name
    output_path.mkdir(exist_ok=True, parents=True)
    log.setup_logger(logger, output_path)
    logger.info(f"Run name: {run_name}")

    data = [
        Entry(
            input=d["input"],
            output=d["output"],
            gold=d["gold"],
            valid=d["valid"],
            score=d["score"],
        )
        for d in json.loads(file.read_text())
    ][:n]

    system_prompt = system_prompt_path.read_text()
    user_template = user_template_path.read_text()

    unfiltered, cost = run_filter(
        client, model, data, system_prompt, user_template, print_messages
    )
    logger.info(f"Total cost: {cost}")
    logger.info(f"Before: {len(data)}, after: {len(unfiltered)}")

    (output_path / "reproduction.json").write_text(
        json.dumps(reproduction_info, indent=2)
    )
    (output_path / "results.json").write_text(
        json.dumps([asdict(c) for c in unfiltered], indent=2)
    )


if __name__ == "__main__":
    typer.run(main)
