# pyright: basic
import hashlib
import json
import logging
import math
import re
import textwrap
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict, cast

import openai
import pandas as pd
from openai.types import CompletionUsage
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion import Choice
from ratelimit import limits, sleep_and_retry

from evaluation import metrics

logger = logging.getLogger("evaluation.gpt")


# sourcery skip: snake-case-variable-declarations
class Instance(TypedDict):
    Cause: list[str]
    Effect: list[str]


def parse_instance(answer: str) -> tuple[Instance, str | None]:
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
    "gpt-4-turbo-2024-04-09": (  # in: $10.00 / 1M tokens, out: $30.00 / 1M tokens
        0.00001,
        0.00003,
    ),
}


def calculate_cost(model: str, response: Any) -> float:
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_input, cost_output = MODEL_COSTS[model]
    return input_tokens * cost_input + output_tokens * cost_output


def dbg_gpt(messages: list[ChatCompletionMessageParam], result: str | None) -> None:
    output = ["\nINPUT:"]
    for msg in messages:
        output.extend((
            f'>>> {msg["role"]}',
            f"{msg.get('content')}",
            "",
        ))
    output.extend((
        "OUTPUT:",
        (result or "<empty>"),
        "-" * 80,
        "",
    ))
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
        top_p=None if temperature == 0 else 1,
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
        f"Cause: {' | '.join(extraction_entities['Cause'])}\n"
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
        """Extract the validity/score result from the GPT output.

        The last line is expected to contain the result. This function is
        case-insensitive and ignores whitespace. It supports the following formats:

        For ResultMode.VALID:
        - "validity: <true/false>"
        - "valid: <true/false>"

        For ResultMode.LIKERT and ResultMode.SCORE:
        - "score: <integer>"

        If the format is not recognized, a warning is logged and 0 is returned.
        """
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

    def get_pred(self, result: int) -> int:
        "Prediction label from the GPT output."
        match self:
            case ResultMode.SCORE:
                return int(result >= 4)
            case ResultMode.VALID | ResultMode.LIKERT:
                return result

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
    gold: str
    answer: str
    context: str
    gold_label: int
    item: dict[str, Any]

    @property
    def answer_msg(self) -> str:
        return "\n".join([self.gold, self.answer]).strip()

    @property
    def gpt_msg(self) -> str:
        return "\n".join([self.context, self.answer]).strip()


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

        gold_label = result_mode.get_gold(item)
        messages.append(
            Message(
                item=item,
                gold=gold,
                answer=answer,
                context=context,
                gold_label=gold_label,
            )
        )

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
    model_used: str


def most_common(lst: list[int]) -> int:
    "Return the most common element in the list."
    return max(set(lst), key=lst.count)


def extract_lines(text: str) -> str:
    """Extract lines between explanation/analysis and the score."""
    pattern = r"(?i)(?:explanation|analysis):\s*(.*?)\s*(?:validity|valid|score):"
    if match := re.search(pattern, text, re.DOTALL):
        return match[1].strip()
    return ""


def starts_with_number(text: str) -> bool:
    """Line starts with any whitespace followed by a number and a word boundary."""
    return bool(re.match(r"^\s*\d+\b", text))


def count_steps(result: str) -> int:
    steps = extract_lines(result)
    return sum(starts_with_number(line) for line in steps.splitlines())


def format_result(i: int, result: str) -> str:
    num_steps = count_steps(result)
    return f"#{i + 1} Num steps: {num_steps}\n{result}"


def reformat_output(
    output_data: list[dict[str, Any]], result_mode: ResultMode
) -> list[dict[str, Any]]:
    "Reformat the output data to match the format of other models."
    return [
        {
            "gold": result_mode.get_gold(item),
            "pred": result_mode.get_pred(item["gpt_result"]),
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
        if c.chain and c.chain != "<empty>"
    )
