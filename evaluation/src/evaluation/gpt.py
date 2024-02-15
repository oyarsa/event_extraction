#!/usr/bin/env python3
# pyright: basic
import json
import logging
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import openai
import pandas as pd
import typer
from openai.types.chat import ChatCompletionMessageParam
from tqdm import tqdm

from evaluation import log, metrics

logger = logging.getLogger("classifier")

# Controls whether to print debug information in some functions.
DEBUG = False


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
    "gpt-4": (  # in: $0.03 / 1K tokens, out: $0.06 / 1K tokens
        0.00003,
        0.00006,
    ),
    "gpt-4-1106-preview": (  # in: $0.01 / 1K tokens, out: $0.03 / 1K tokens
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


def run_gpt(
    client: openai.OpenAI,
    model: str,
    message: str,
    system_prompt: str,
    user_prompt: str,
    ctx_prompt: str | None = None,
) -> tuple[str, float]:  # sourcery skip: extract-method
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if ctx_prompt:
        messages.append({"role": "user", "content": ctx_prompt})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
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

    if DEBUG:
        dbg_gpt(messages, result)

    return result or "<empty>", cost


SYSTEM_PROMPTS = {
    "simple": """\
You are a helpful assistant that can take a context and an extraction composed of a \
cause and effect, and determine whether that extraction is valid.\
"""
}
USER_PROMPTS = {
    "simple": """\
Given the context, is the extraction (cause and effect) valid? \
Respond with either 'true' or 'false'.\
    """,
    "instructions": """\
Given the context, how valid is the extraction? The extraction is composed of a cause \
and effect. The cause and effect are spans of the context.

Evaluate the extraction based on the following criteria:

1. Read the extraction and compare it to the context. Check if the extraction contains
the cause and effect mentioned in the context.
2. Make sure that the extraction clauses only contain the necessary information.
3. Penalize extractions that are too long or too short.
4. Penalize extractions that include more information than necessary for the clause.
5. Assign a score for validity on a scale from 1 to 5, where 1 is the lowest and \
5 is the highest based on the Evaluation Criteria.

Respond with the following format:
Explanation: <text explanating the score>
Score: <score from 1 to 5>\
""",
}


def split_data(
    data: list[dict[str, Any]], n: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    "Split the data into n positive and n negative examples."
    valids = [item for item in data if item["valid"]][:n]
    invalids = [item for item in data if not item["valid"]][:n]

    return valids, invalids


def format_data(item: dict[str, Any], mode: str) -> str:
    context = f"Context:\n{item['input']}"

    if mode == "extraction":
        entities, _ = parse_instance(item["output"])
        answer = (
            "Extraction:\n"
            f"Cause: {' | '.join(entities['Cause'])}\n"
            f"Effect: {' | '.join(entities['Effect'])}"
        )
    else:
        answer = f"Answer: {item['output']}"

    score = f"Score: {5 if item['valid'] else 1}"
    return "\n".join([context, answer, score])


def build_context(data: list[dict[str, str]], n: int, mode: str) -> str:
    "Build a context prompt from data. Uses n positive and n negative examples."
    valids, invalids = split_data(data, n)
    valid_msg = [
        "Some examples of _valid_ extractions:",
        *(format_data(item, mode) for item in valids),
    ]
    invalid_msg = [
        "Some examples of _invalid_ extractions:",
        *(format_data(item, mode) for item in invalids),
    ]
    return "\n\n".join(valid_msg + invalid_msg)


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


def make_messages(
    sampled_data: list[dict[str, Any]], mode: str
) -> list[tuple[dict[str, Any], str, str, bool]]:
    messages: list[tuple[dict[str, Any], str, str, bool]] = []
    for item in sampled_data:
        context = f"Context: {item['input']}"
        answer = f"Valid?: {item['valid']}"

        if mode == "extraction":
            gold, answer = make_message_extraction(item)
        else:
            answer = f"Answer: {item['output']}"
            gold = f"Gold: {item['gold']}"

        answer_msg = "\n".join([gold, answer]).strip()
        gpt_msg = "\n".join([context, answer]).strip()
        messages.append((item, answer_msg, gpt_msg, item["valid"]))

    return messages


def run_model(
    messages: list[tuple[dict[str, Any], str, str, bool]],
    model: str,
    client: openai.OpenAI,
    system_prompt: str,
    user_prompt: str,
    ctx_prompt: str | None,
    print_messages: bool,
) -> tuple[list[dict[str, Any]], float, dict[tuple[bool, int], int]]:
    results: defaultdict[tuple[bool, int], int] = defaultdict(int)
    total_cost = 0
    output_data: list[dict[str, Any]] = []

    for item, answer_msg, gpt_msg, valid in tqdm(messages):
        result_s, cost = run_gpt(
            client=client,
            model=model,
            message=gpt_msg,
            system_prompt=SYSTEM_PROMPTS[system_prompt],
            user_prompt=USER_PROMPTS[user_prompt],
            ctx_prompt=ctx_prompt,
        )
        total_cost += cost

        last_line = result_s.splitlines()[-1].replace("Score:", "").strip()
        result = int(last_line) if last_line.isdigit() else 0
        results[valid, result] += 1

        output_data.append(item | {"gpt_reward": result, "gpt_response": result_s})

        if print_messages:
            output = [
                ctx_prompt or "",
                "-" * 80,
                SYSTEM_PROMPTS[system_prompt],
                "-" * 80,
                USER_PROMPTS[user_prompt],
                "-" * 80,
                gpt_msg,
                "-" * 80,
                f"\nGPT: '{result_s}'",
                "",
                f"NOT SENT {'~' * 50}",
                answer_msg,
                "*" * 80,
                "",
            ]
            logger.info("\n".join(output))

    return output_data, total_cost, results


def reformat_output(output_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    "Reformat the output data to match the format of other models."
    return [
        {
            "gold": int(item["valid"]),
            "pred": int(item["gpt_reward"] >= 4),
            "passage": item["input"],
            "output": item["output"],
            "annotation": item["gold"],
            "tag": None,
        }
        for item in output_data
    ]


def calculate_metrics(data: list[dict[str, Any]]) -> dict[str, float]:
    "Calculate main metrics for the GPT result."
    return metrics.calc_metrics(
        metrics.EvaluationResult(
            golds=[d["gold"] for d in data],
            preds=[d["pred"] for d in data],
            passages=[d["passage"] for d in data],
            outputs=[d["output"] for d in data],
            annotations=[d["annotation"] for d in data],
            loss=math.nan,  # no loss available from GPT
        )
    )


def confusion_matrix(results: dict[tuple[bool, int], int]) -> pd.DataFrame:
    "Generate confusion matrix from the predictions and annotation counts."
    df = pd.DataFrame(
        list(results.items()), columns=pd.Series(["Combination", "Count"])
    )
    df[["gold", "pred"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop("Combination", axis="columns")
    df["Count"] = df["Count"].astype(int)
    df = df.pivot_table(index="gold", columns="pred", values="Count", fill_value=0)

    return df


def init_client(api_type: str, config: dict[str, Any]) -> openai.OpenAI:
    "Create client for OpenAI or Azure API, depending on the configuration."
    config = config[api_type]

    if api_type == "azure":
        logger.info("Using Azure OpenAI API")
        return openai.AzureOpenAI(
            api_key=config["key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
            azure_deployment=config["deployment"],
        )
    elif api_type == "openai":
        logger.info("Using OpenAI API")
        return openai.OpenAI(api_key=config["key"])
    else:
        raise ValueError(f"Unknown API type: {config['api_type']}")


def main(
    file: Path = typer.Argument(
        ...,
        help="Path to the json file containing the data (list of objects with keys"
        " 'input', 'output', 'gold', 'valid').",
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
    openai_config_path: Path = typer.Option(
        Path("config.json"),
        help="Path to the file containing the OpenAI API keys.",
    ),
    api_type: str = typer.Option(
        "openai",
        help="API type, defaults to 'openai'.",
    ),
    model: str = typer.Option(
        "gpt-4",
        help="Which GPT model to use (e.g., 'gpt-3.5-turbo', 'gpt-4').",
    ),
    system_prompt: str = typer.Option(
        "simple",
        help="Which system prompt to use (only 'simple' for now).",
    ),
    user_prompt: str = typer.Option(
        "instructions",
        help="Which user prompt to use ('simple' or 'instructions').",
    ),
    use_context: bool = typer.Option(
        False,
        help="Whether to use the context prompt.",
    ),
    context_size: int = typer.Option(
        2,
        help="Context size if context is used.",
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
    ),
    run_name: Optional[str] = typer.Option(
        None,
        help="Name of the run. Generated from the model and prompts if not provided.",
    ),
    all_data: bool = typer.Option(
        False,
        help="Whether to run the entire dataset. If true, n is ignored.",
    ),
    mode: str = typer.Option(
        "extraction",
        help=f"Mode of the data. One of {SUPPORTED_MODES}.",
    ),
) -> None:  # sourcery skip: low-code-quality
    "Run a GPT model on the given data and evaluate the results."
    global DEBUG  # noqa: PLW0603
    DEBUG = debug

    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {list(MODEL_COSTS.keys())}")
    if system_prompt not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system prompt. Options: {SYSTEM_PROMPTS.keys()}")
    if user_prompt not in USER_PROMPTS:
        raise ValueError(f"Invalid user prompt. Options: {USER_PROMPTS.keys()}")
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Invalid mode. Options: {SUPPORTED_MODES}")

    if run_name is None:
        run_name = f"{model}-sys_{system_prompt}-user_{user_prompt}-n{n}"
        if use_context:
            run_name += f"-context{context_size}"
        if rand:
            run_name += "-rand"

    output_path = output_dir / run_name
    output_path.mkdir(exist_ok=True, parents=True)
    log.setup_logger(logger, output_path)

    openai_config = json.loads(openai_config_path.read_text())
    client = init_client(api_type, openai_config)

    data = json.loads(file.read_text())
    if rand:
        random.shuffle(data)

    if use_context:
        ctx_prompt = build_context(data, context_size, mode)
        data = data[context_size * 2 :]
    else:
        ctx_prompt = None

    if all_data:
        sampled_data = data
    else:
        # Get same number of valid and invalid examples
        valids, invalids = split_data(data, math.ceil(n / 2))
        sampled_data = valids + invalids

    messages = make_messages(sampled_data, mode)
    output_data, total_cost, results = run_model(
        messages,
        model,
        client,
        system_prompt,
        user_prompt,
        ctx_prompt,
        print_messages,
    )
    formatted_output = reformat_output(output_data)
    metrics_ = calculate_metrics(formatted_output)

    (output_path / "full_output.json").write_text(json.dumps(output_data, indent=2))
    (output_path / "results.json").write_text(json.dumps(formatted_output, indent=2))
    (output_path / "metrics.json").write_text(json.dumps(metrics_, indent=2))
    metrics.report_metrics(logger, metrics_, "GPT")

    with Path("cost.csv").open("a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{total_cost}\n")

    logger.info(f"\n{confusion_matrix(results)}\n")
    logger.info(f"Total cost: ${total_cost}")


if __name__ == "__main__":
    typer.run(main)
