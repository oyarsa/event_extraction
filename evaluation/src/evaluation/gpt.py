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


class FilterStatus(Enum):
    "Filter status for the GPT model output."
    UNFILTERED = 0
    FILTERED = 1


@dataclass
class GptResult:
    results: list[str]
    cost: float
    model_used: str
    filtered: FilterStatus


def make_chat_request(
    client: openai.OpenAI, **kwargs: Any
) -> tuple[ChatCompletion, FilterStatus]:
    calls_per_minute = 3500  # OpenAI full plan

    # Ignores (mypy): untyped decorator makes function untyped
    @sleep_and_retry  # type: ignore[misc]
    @limits(calls=calls_per_minute, period=60)  # type: ignore[misc]
    def _make_chat_request(**kwargs: Any) -> tuple[ChatCompletion, FilterStatus]:
        attempts = 0
        while True:
            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                if isinstance(e, openai.BadRequestError):
                    message = cast(dict[str, str], e.body).get("message", "")
                    if AZURE_FILTER_MESSAGE in message:
                        return (
                            make_filter_response(kwargs["messages"]),
                            FilterStatus.FILTERED,
                        )

                attempts += 1

                if isinstance(e, openai.APIStatusError) and e.status_code == 429:
                    logger.info(
                        f"Rate limit exceeded. Waiting 10 seconds."
                        f" Attempts: {attempts + 1}"
                    )
                    time.sleep(10)
                else:
                    message = e.message if isinstance(e, openai.APIError) else str(e)
                    logger.info(f'Error: "{message}" / Attempt {attempts + 1}')
            else:
                return response, FilterStatus.UNFILTERED

    # This cast is necessary because of the sleep_and_retry and limits decorators,
    # which make the function untyped.
    return cast(tuple[ChatCompletion, FilterStatus], _make_chat_request(**kwargs))


@dataclass
class Chain:
    input: str
    answer: str
    chain: str
    score: int


def load_chains(chains_path: Path) -> list[Chain]:
    chains = json.loads(chains_path.read_text())
    return [
        Chain(input=c["input"], answer=c["answer"], chain=c["chain"], score=c["score"])
        for c in chains
    ]


def run_gpt(
    client: openai.OpenAI,
    model: str,
    message: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_samples: int | None,
    chain_prompt: str | None,
    debug: bool,
) -> GptResult:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if chain_prompt:
        messages.append({"role": "user", "content": chain_prompt})
    messages.append({"role": "user", "content": message})

    response, filtered = make_chat_request(
        client,
        model=model,
        messages=messages,
        temperature=temperature,
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
    QA = "qa"
    EXTRACTION = "extraction"


def format_data(item: dict[str, Any], mode: DataMode) -> str:
    match mode:
        case DataMode.EXTRACTION:
            entities, _ = parse_instance(item["output"])
            context = f"Context:\n{item['input']}"
            answer = (
                "Extraction:\n"
                f"Cause: {' | '.join(entities['Cause'])}\n"
                f"Effect: {' | '.join(entities['Effect'])}"
            )
        case DataMode.QA:
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
    VALID = "valid"
    SCORE = "score"
    LIKERT = "likert"

    def extract_result(self, result: str) -> int:
        regex = "(validity|valid|score):"
        last_line = result.splitlines()[-1].lower()
        last_line = re.sub(regex, "", last_line).strip()

        match self:
            case ResultMode.VALID:
                if "true" in last_line:
                    return 1
                elif "false" in last_line:
                    return 0
            case ResultMode.SCORE | ResultMode.LIKERT:
                if last_line.isdigit():
                    return int(last_line)

        logger.warning(f"Invalid result: {last_line}")
        return 0

    def get_gold(self, item: dict[str, Any]) -> int:
        "Gold label that we use to evaluate the GPT output."
        match self:
            case ResultMode.LIKERT:
                return item["score"]
            case ResultMode.VALID | ResultMode.SCORE:
                return int(item["valid"])

    def get_pred(self, gpt_reward: int) -> int:
        "Prediction label from the GPT output."
        match self:
            case ResultMode.SCORE:
                return int(gpt_reward >= 4)
            case ResultMode.VALID | ResultMode.LIKERT:
                return gpt_reward

    def get_input_score(self, item: dict[str, Any]) -> int:
        """Score that we will send to the GPT model as the gold.

        This is different from the gold label that we use to evaluate the GPT output
        (see `ResultMode.get_gold`). For the "score" model, we want the model to output
        an integer, and we convert that integer to a boolean value for evaluation.
        """
        match self:
            case ResultMode.VALID:
                return int(item["valid"])
            case ResultMode.SCORE | ResultMode.LIKERT:
                return item["score"]

    @property
    def mse(self) -> bool:
        match self:
            case ResultMode.LIKERT:
                return True
            case ResultMode.VALID | ResultMode.SCORE:
                return False

    @property
    def average_method(self) -> str:
        match self:
            case ResultMode.LIKERT:
                return "macro"
            case ResultMode.VALID | ResultMode.SCORE:
                return "binary"

    @property
    def display(self) -> str:
        match self:
            case ResultMode.VALID:
                return "Valid"
            case ResultMode.SCORE | ResultMode.LIKERT:
                return "Score"

    def convert_score(self, score: int) -> str:
        match self:
            case ResultMode.VALID:
                return "true" if score == 1 else "false"
            case ResultMode.SCORE | ResultMode.LIKERT:
                return str(score)


@dataclass
class Message:
    item: dict[str, Any]
    answer_msg: str
    gpt_msg: str
    gold_label: int


def make_chain_prompt(chains: list[Chain], result_mode: ResultMode) -> str:
    return "\n\n".join(
        f"""\
Example {i}:
{c.input}

Answer: {c.answer}

Explanation:
{c.chain}

{result_mode.display}: {result_mode.convert_score(c.score)}
"""
        for i, c in enumerate(chains, 1)
    )


def make_messages(
    sampled_data: list[dict[str, Any]], mode: DataMode, result_mode: ResultMode
) -> list[Message]:
    messages: list[Message] = []

    for item in sampled_data:
        match mode:
            case DataMode.EXTRACTION:
                context = f"Context: {item['input']}"
                gold, answer = make_message_extraction(item)
            case DataMode.QA:
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
    chains: list[Chain],
    print_chains: bool,
    debug: bool,
) -> ModelResult:
    results: defaultdict[tuple[int, int], int] = defaultdict(int)
    total_cost = 0
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

        if gpt_result.filtered is FilterStatus.FILTERED:
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
            ]
            if (
                chain_prompt
                and print_chains
                and gpt_result.filtered is FilterStatus.UNFILTERED
            ):
                output.extend(["-" * 80, chain_prompt])
            if gpt_result.filtered is FilterStatus.UNFILTERED:
                output.extend(
                    [
                        "-" * 80,
                        msg.gpt_msg,
                        "-" * 80,
                        "\nGPT: ",
                        *(
                            format_result(i, r)
                            for i, r in enumerate(gpt_result.results)
                        ),
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

    logger.info(f"Total filtered: {filtered}")
    return ModelResult(output_data, results, total_cost)


def extract_lines(text: str) -> str:
    pattern = r"(?i)(?:explanation|analysis):\s*(.*?)\s*(?:validity|valid|score):"
    if match := re.search(pattern, text, re.DOTALL):
        return match[1].strip()
    return ""


def count_steps(result: str) -> int:
    steps = extract_lines(result)
    return sum(bool(line.strip()) for line in steps.splitlines())


def format_result(i: int, result: str) -> str:
    num_steps = count_steps(result)
    return f"#{i+1} Num steps: {num_steps}\n{result}"


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
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{model_result.total_cost}\n")

    logger.info(f"\n{confusion_matrix(model_result.results)}\n")
    logger.info(f"Total cost: ${model_result.total_cost}")


if __name__ == "__main__":
    typer.run(main)
