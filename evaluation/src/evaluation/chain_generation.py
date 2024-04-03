# pyright: basic
import json
import logging
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import openai
import typer
from openai.types.chat import ChatCompletionMessageParam

from evaluation import log
from evaluation.gpt import (
    MODEL_COSTS,
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
) -> GptResult:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response, filtered = make_chat_request(
        client,
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=0,
    )
    result = response.choices[0].message.content
    cost = calculate_cost(model, response)

    return GptResult(
        results=[result or "<empty>"],
        cost=cost,
        model_used=response.model,
        filtered=filtered,
    )


@dataclass
class ChainData:
    input: str
    output: str


@dataclass
class ChainResult:
    input: str
    output: str
    chain: str


def generate_chain(
    client: openai.OpenAI,
    model: str,
    data: ChainData,
    system_prompt: str,
    user_template: str,
) -> str:
    user_prompt = user_template.format(INPUT=data.input, ANSWER=data.output)
    result = run_gpt(client, model, system_prompt, user_prompt)
    return result.results[0]


def indent(text: str) -> str:
    return "\n".join(
        textwrap.indent(subline, "\t")
        for line in text.splitlines()
        for subline in textwrap.wrap(line, width=80)
    )


def print_result(data: ChainData, result: str) -> str:
    out: list[str] = [
        "",
        f"Input:\n{indent(data.input)}",
        f"Output:\n{indent(data.output)}",
        f"Chain:\n{indent(result)}",
        "\n",
    ]
    return "\n".join(out)


def generate_chains(
    client: openai.OpenAI,
    model: str,
    data: list[ChainData],
    system_prompt: str,
    user_template: str,
    print_results: bool,
) -> list[ChainResult]:
    results: list[ChainResult] = []

    for d in data:
        chain = generate_chain(client, model, d, system_prompt, user_template)
        results.append(ChainResult(input=d.input, output=d.output, chain=chain))

        if print_results:
            logger.info(print_result(d, chain))

    return results


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
    print_results: bool = typer.Option(
        False,
        "--print-results",
        help="Print the generated chains.",
    ),
) -> None:
    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {tuple(MODEL_COSTS)}")

    reproduction_info = {
        "command": sys.argv,
        "data_hash": hash_file(file),
    }

    if run_name is None:
        run_name = (
            f"{model}_{system_prompt_path.name}_{user_template_path.name}_{file.name}"
        )

    output_path = output_dir / run_name
    output_path.mkdir(exist_ok=True, parents=True)
    log.setup_logger(logger, output_path)
    logger.info(f"Run name: {run_name}")

    openai_config: dict[str, dict[str, str]] = json.loads(
        openai_config_path.read_text()
    )
    client, config_model = init_client(api_type, openai_config)
    model = config_model or model
    logger.info(f"Model: {model}")

    data = [
        ChainData(input=d["input"], output=d["output"])
        for d in json.loads(file.read_text())
    ]

    system_prompt = system_prompt_path.read_text()
    user_template = user_template_path.read_text()

    chains = generate_chains(
        client, model, data, system_prompt, user_template, print_results
    )

    (output_path / "reproduction.json").write_text(
        json.dumps(reproduction_info, indent=2)
    )
    (output_path / "results.json").write_text(
        json.dumps([asdict(c) for c in chains], indent=2)
    )


if __name__ == "__main__":
    typer.run(main)
