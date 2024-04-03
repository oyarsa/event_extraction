#!/usr/bin/env python3
# pyright: basic
import hashlib
import json
import logging
import math
import random
import re
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, cast

import openai
import pandas as pd
import typer
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import Choice
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

from evaluation import log, metrics

logger = logging.getLogger("evaluation.gpt")


def parse_instance(answer: str) -> tuple[dict[str, list[str]], str | None]:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return {
            "Cause": [],
            "Effect": [],
        }, "cause"
    causes, relation, effects = matches[0]
    causes = sorted(c.strip() for c in causes.split("|") if c.strip())
    effects = sorted(e.strip() for e in effects.split("|") if e.strip())
    relation = relation.strip()

    return {
        "Cause": causes,
        "Effect": effects,
    }, relation


MODEL_COSTS = {
    "gpt-3.5-turbo": (  # in: $0.001 / 1K tokens, out: $0.002 / 1K tokens
        0.0000005,
        0.0000015,
    ),
    "gpt-3.5-turbo-1106": (  # in: $0.001 / 1K tokens, out: $0.002 / 1K tokens
        0.0000005,
        0.0000015,
    ),
    "gpt-4": (  # in: $0.03 / 1K tokens, out: $0.06 / 1K tokens
        0.00003,
        0.00006,
    ),
    "gpt-4-1106-preview": (  # in: $0.01 / 1K tokens, out: $0.03 / 1K tokens
        0.00001,
        0.00003,
    ),
    "gpt-4-0125-preview": (  # in: $0.01 / 1K tokens, out: $0.03 / 1K tokens
        0.00001,
        0.00003,
    ),
}

SUPPORTED_MODES = ("qa", "extraction")


def calculate_cost(model: str, response: Any) -> float:
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_input, cost_output = MODEL_COSTS[model]
    return input_tokens * cost_input + output_tokens * cost_output


def dbg_gpt(messages: list[ChatCompletionMessageParam], result: str | None) -> None:
    output = ["\nINPUT:"]
    for msg in messages:
        output.extend(
            (
                f'>>> {msg["role"]}',
                f"{msg.get('content')}",
                "",
            )
        )
    output.extend(
        (
            "OUTPUT:",
            (result or "<empty>"),
            "-" * 80,
            "",
        )
    )
    logger.info("\n".join(output))


AZURE_FILTER_MESSAGE = "The response was filtered due to the prompt triggering Azure OpenAI's content management policy"


def make_filter_response(messages: list[dict[str, str]]) -> ChatCompletion:
    return ChatCompletion(
        object="chat.completion",
        id="",
        model="",
        choices=[
            Choice(
                finish_reason="content_filter",
                index=0,
                message=ChatCompletionMessage(
                    content=json.dumps(messages, indent=2), role="assistant"
                ),
            )
        ],
        created=int(time.time()),
        usage=CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )


@dataclass
class GptResult:
    results: list[str]
    cost: float
    model_used: str
    filtered: bool


def make_chat_request(
    client: openai.OpenAI, **kwargs: Any
) -> tuple[ChatCompletion, bool]:
    calls_per_minute = 3500  # OpenAI full plan

    # Ignores (mypy): untyped decorator makes function untyped
    @sleep_and_retry  # type: ignore[misc]
    @limits(calls=calls_per_minute, period=60)  # type: ignore[misc]
    def _make_chat_request(**kwargs: Any) -> tuple[ChatCompletion, bool]:
        attempts = 0
        while True:
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                if isinstance(e, openai.BadRequestError):
                    message = cast(dict[str, str], e.body).get("message", "")
                    if AZURE_FILTER_MESSAGE in message:
                        return make_filter_response(kwargs["messages"]), True

                logger.info(f'Error - {type(e)} - "{e}" / Attempt {attempts + 1}')
                attempts += 1

                if isinstance(e, openai.APIStatusError) and e.status_code == 429:
                    logger.info("Rate limit exceeded. Waiting 10 seconds.")
                    time.sleep(10)
            else:
                return response, False

    # This cast is necessary because of the sleep_and_retry and limits decorators,
    # which make the function untyped.
    return cast(tuple[ChatCompletion, bool], _make_chat_request(**kwargs))


def run_gpt(
    client: openai.OpenAI,
    model: str,
    message: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_samples: int | None,
    debug: bool,
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
        temperature=temperature,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=0,
        n=num_samples,
    )
    results = [c.message.content or "<empty>" for c in response.choices]
    cost = calculate_cost(model, response)

    if debug:
        for result in results:
            dbg_gpt(messages, result)

    return GptResult(
        results=results,
        cost=cost,
        model_used=response.model,
        filtered=filtered,
    )


def split_data(
    data: list[dict[str, Any]], n: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    "Split the data into n positive and n negative examples."
    valids = [item for item in data if item["valid"]][:n]
    invalids = [item for item in data if not item["valid"]][:n]

    return valids, invalids


class DataMode(str, Enum):
    qa = "qa"
    extraction = "extraction"


def format_data(item: dict[str, Any], mode: DataMode) -> str:
    match mode:
        case DataMode.extraction:
            entities, _ = parse_instance(item["output"])
            context = f"Context:\n{item['input']}"
            answer = (
                "Extraction:\n"
                f"Cause: {' | '.join(entities['Cause'])}\n"
                f"Effect: {' | '.join(entities['Effect'])}"
            )
        case DataMode.qa:
            context = item["input"]
            answer = f"Answer: {item['output']}"

    score = f"Score: {5 if item['valid'] else 1}"
    return "\n".join([context, answer, score])


def make_message_extraction(item: dict[str, Any]) -> tuple[str, str]:
    extraction_entities, _ = parse_instance(item["output"])
    extraction = (
        "Extraction:\n"
        f"Cause: \n"
        f"Effect: {' | '.join(extraction_entities['Effect'])}\n"
    )

    gold_entities, _ = parse_instance(item["gold"])
    gold = (
        "GOLD:\n"
        f"Cause: {' | '.join(gold_entities['Cause'])}\n"
        f"Effect: {' | '.join(gold_entities['Effect'])}\n"
    )

    return gold, extraction


class ResultMode(str, Enum):
    "Result mode for the GPT model."
    valid = "valid"
    score = "score"
    likert = "likert"

    def extract_result(self, result: str) -> int:
        regex = "(validity|valid|score):"
        last_line = result.splitlines()[-1].lower()
        last_line = re.sub(regex, "", last_line).strip()

        match self:
            case ResultMode.valid:
                if "true" in last_line:
                    return 1
                elif "false" in last_line:
                    return 0
            case ResultMode.score | ResultMode.likert:
                if last_line.isdigit():
                    return int(last_line)

        logger.warning(f"Invalid result: {last_line}")
        return 0

    def get_gold(self, item: dict[str, Any]) -> int:
        match self:
            case ResultMode.likert:
                return item["score"]
            case ResultMode.valid | ResultMode.score:
                return int(item["valid"])

    def get_pred(self, gpt_reward: int) -> int:
        match self:
            case ResultMode.score:
                return int(gpt_reward >= 4)
            case ResultMode.valid | ResultMode.likert:
                return gpt_reward

    @property
    def mse(self) -> bool:
        match self:
            case ResultMode.likert:
                return True
            case ResultMode.valid | ResultMode.score:
                return False

    @property
    def average_method(self) -> str:
        match self:
            case ResultMode.likert:
                return "macro"
            case ResultMode.valid | ResultMode.score:
                return "binary"


@dataclass
class Message:
    item: dict[str, Any]
    answer_msg: str
    gpt_msg: str
    gold_label: int


def make_messages(
    sampled_data: list[dict[str, Any]], mode: DataMode, result_mode: ResultMode
) -> list[Message]:
    messages: list[Message] = []

    for item in sampled_data:
        match mode:
            case DataMode.extraction:
                context = f"Context: {item['input']}"
                gold, answer = make_message_extraction(item)
            case DataMode.qa:
                context = item["input"]
                answer = f"answer: {item['output']}"
                gold = f"Gold: {item['gold']}"

        answer_msg = "\n".join([gold, answer]).strip()
        gpt_msg = "\n".join([context, answer]).strip()

        gold_label = result_mode.get_gold(item)
        messages.append(Message(item, answer_msg, gpt_msg, gold_label))

    return messages


def render_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        f">> {msg['role'].upper()}\n{textwrap.indent(msg['content'], ' ' * 4)}"
        for msg in messages
    )


@dataclass
class ModelResult:
    output_data: list[dict[str, Any]]
    results: dict[tuple[int, int], int]
    total_cost: float


def most_common(lst: list[int]) -> int:
    "Return the most common element in the list."
    return max(set(lst), key=lst.count)


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
    debug: bool,
) -> ModelResult:
    results: defaultdict[tuple[int, int], int] = defaultdict(int)
    total_cost = 0
    filtered = 0
    output_data: list[dict[str, Any]] = []

    for msg in tqdm(messages):
        gpt_result = run_gpt(
            client=client,
            model=model,
            message=msg.gpt_msg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            num_samples=num_samples,
            debug=debug,
        )
        total_cost += gpt_result.cost

        if gpt_result.filtered:
            filtered += 1
            logger.info(f"Content filtered. Occurrences: {filtered}.")

            if print_messages:
                logger.info(
                    f"  Prompt:\n{render_messages(json.loads(gpt_result.results[0]))}"
                )

        parsed_results = [result_mode.extract_result(r) for r in gpt_result.results]
        result = most_common(parsed_results)
        results[msg.gold_label, result] += 1

        output_data.append(msg.item | {"gpt_reward": result})

        if print_messages:
            output = [
                "-" * 80,
                system_prompt,
                "-" * 80,
                user_prompt,
                "-" * 80,
                msg.gpt_msg,
                "-" * 80,
                "\nGPT: ",
                *(f"{i+1} - {r}" for i, r in enumerate(gpt_result.results)),
                f"\nParsed: {result}",
                f"NOT SENT {'~' * 50}",
                msg.answer_msg,
                "*" * 80,
                "",
            ]
            logger.info("\n".join(output))

    logger.info(f"Total filtered: {filtered}")
    return ModelResult(output_data, results, total_cost)


def reformat_output(
    output_data: list[dict[str, Any]], result_mode: ResultMode
) -> list[dict[str, Any]]:
    "Reformat the output data to match the format of other models."
    return [
        {
            "gold": result_mode.get_gold(item),
            "pred": result_mode.get_pred(item["gpt_reward"]),
            "passage": item["input"],
            "output": item["output"],
            "annotation": item["gold"],
            "tag": None,
        }
        for item in output_data
    ]


def calculate_metrics(
    data: list[dict[str, Any]], result_mode: ResultMode
) -> dict[str, float]:
    """Calculate main metrics for the GPT result.

    If `result_mode` is `ResultMode.likert`, the metrics are calculated using the
    mean squared error and macro averaging. Otherwise, the metrics are calculated using
    binary averaging.
    """
    return metrics.calc_metrics(
        metrics.EvaluationResult(
            golds=[d["gold"] for d in data],
            preds=[d["pred"] for d in data],
            passages=[d["passage"] for d in data],
            outputs=[d["output"] for d in data],
            annotations=[d["annotation"] for d in data],
            loss=math.nan,  # no loss available from GPT
        ),
        average=result_mode.average_method,
        mse=result_mode.mse,
    )


def confusion_matrix(results: dict[tuple[int, int], int]) -> pd.DataFrame:
    "Generate confusion matrix from the predictions and annotation counts."
    df = pd.DataFrame(
        list(results.items()), columns=pd.Series(["Combination", "Count"])
    )
    df[["gold", "pred"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop("Combination", axis="columns")
    df["Count"] = df["Count"].astype(int)
    df = df.pivot_table(index="gold", columns="pred", values="Count", fill_value=0)

    return df


def init_client(
    api_type: str, config: dict[str, dict[str, Any]]
) -> tuple[openai.OpenAI, str | None]:
    "Create client for OpenAI or Azure API, depending on the configuration."
    api_config = config[api_type]

    if api_type.startswith("azure"):
        logger.info("Using Azure OpenAI API")
        return (
            openai.AzureOpenAI(
                api_key=api_config["key"],
                api_version=api_config["api_version"],
                azure_endpoint=api_config["endpoint"],
                azure_deployment=api_config["deployment"],
            ),
            api_config["model"],
        )
    elif api_type.startswith("openai"):
        logger.info("Using OpenAI API")
        return openai.OpenAI(api_key=api_config["key"]), None
    else:
        raise ValueError(f"Unknown API type: {api_type}")


def hash_file(file: Path) -> str:
    return hashlib.md5(file.read_bytes()).hexdigest()


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
        True,
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
    all_data: bool = typer.Option(
        False,
        help="Whether to run the entire dataset. If true, n is ignored.",
    ),
    data_mode: DataMode = typer.Option(DataMode.extraction, help="Data mode."),
    result_mode: ResultMode = typer.Option(ResultMode.score, help="Result mode."),
    temperature: float = typer.Option(
        0.0, help="Temperature for the GPT model.", min=0.0, max=1.0
    ),
    num_samples: Optional[int] = typer.Option(
        None, help="Number of samples to generate."
    ),
) -> None:  # sourcery skip: low-code-quality
    "Run a GPT model on the given data and evaluate the results."

    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {tuple(MODEL_COSTS)}")
    if num_samples is not None and temperature == 0:
        raise ValueError("Number of samples is set but temperature is 0.")
    if (num_samples is None or num_samples == 1) and temperature != 0:
        raise ValueError("Temperature is set but number of samples is not.")

    reproduction_info = {
        "command": sys.argv,
        "data_hash": hash_file(file),
    }

    if run_name is None:
        run_name = f"{model}-sys_{system_prompt_path.name}-user_{user_prompt_path.name}"
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
        logger, metrics_, "GPT", mse=result_mode is ResultMode.likert
    )

    with Path("cost.csv").open("a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{model_result.total_cost}\n")

    logger.info(f"\n{confusion_matrix(model_result.results)}\n")
    logger.info(f"Total cost: ${model_result.total_cost}")


if __name__ == "__main__":
    typer.run(main)
