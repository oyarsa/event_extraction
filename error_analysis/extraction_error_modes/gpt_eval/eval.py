#!/usr/bin/env python3
# pyright: basic
import json
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
        0.000001,
        0.000002,
    ),
    "gpt-4": (0.00003, 0.00006),  # in: $0.03 / 1K tokens, out: $0.06 / 1K tokens
}


def calculate_cost(model: str, response: Any) -> float:
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_input, cost_output = MODEL_COSTS[model]
    return input_tokens * cost_input + output_tokens * cost_output


def dbg_gpt(messages: list[ChatCompletionMessageParam], result: str | None) -> None:
    print("INPUT:")
    for msg in messages:
        print(f'>>> {msg["role"]}')
        print(f"{msg['content']}")
        print()
    print("OUTPUT:")
    print(result)
    print("-" * 80)
    print()


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


def format_data(item: dict[str, Any]) -> str:
    context = f"Context: {item['input']}"

    entities, _ = parse_instance(item["output"])
    extraction = (
        "Extraction:\n"
        f"Cause: {' | '.join(entities['Cause'])}\n"
        f"Effect: {' | '.join(entities['Effect'])}"
    )

    answer = f"Score: {5 if item['valid'] else 1}"
    return "\n".join([context, extraction, answer])


def build_context(data: list[dict[str, str]], n: int = 2) -> str:
    "Build a context prompt from data. Uses n positive and n negative examples."
    valids, invalids = split_data(data, n)
    valid_msg = ["Some examples of _valid_ extractions:", *map(format_data, valids)]
    invalid_msg = [
        "Some examples of _invalid_ extractions:",
        *map(format_data, invalids),
    ]
    return "\n\n".join(valid_msg + invalid_msg)


def main(
    file: Path,
    n: int = 10,
    rand: bool = True,
    key_file: Path = Path("key"),
    model: str = "gpt-4",
    system_prompt: str = "simple",
    user_prompt: str = "instructions",
    use_context: bool = False,
    context_size: int = 2,
    print_messages: bool = True,
    debug: bool = False,
    output_dir: Path = Path("output"),
    run_name: Optional[str] = None,
) -> None:
    """
    Run a GPT model on the given data and evaluate the results.

    \b
    - file: Path to the json file containing the data (list of objects with keys
        'input', 'output', 'gold', 'valid')
    - n: Number of examples to run. Should be even. If not, the number is rounded up.
    - rand: Whether to shuffle the data before selecting n examples
    - key_file: Path to the file containing the OpenAI API key (simple text file
        containing only the key)
    - model: Which GPT model to use (gpt-3.5-turbo or gpt-4)
    - system_prompt: Which system prompt to use (only 'simple' for now)
    - user_prompt: Which user prompt to use ('simple' or 'instructions')
    - use_context: Whether to use the context prompt
    - print_messages: Whether to print the prompt, context, gold and prediction. If
        false, only the progress bar and evaluation results are printed.
    """
    global DEBUG  # noqa: PLW0603
    DEBUG = debug

    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {list(MODEL_COSTS.keys())}")
    if system_prompt not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system prompt. Options: {SYSTEM_PROMPTS.keys()}")
    if user_prompt not in USER_PROMPTS:
        raise ValueError(f"Invalid user prompt. Options: {USER_PROMPTS.keys()}")

    if run_name is None:
        run_name = f"{model}-sys_{system_prompt}-user_{user_prompt}-n{n}"
        if use_context:
            run_name += f"-context{context_size}"
        if rand:
            run_name += "-rand"

    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{run_name}.json"

    api_key = key_file.read_text().strip()

    data = json.loads(file.read_text())
    if rand:
        random.shuffle(data)

    if use_context:
        ctx_prompt = build_context(data, context_size)
        data = data[context_size * 2 :]
    else:
        ctx_prompt = None

    n = n or len(data)
    valids, invalids = split_data(data, math.ceil(n / 2))
    sampled_data = valids + invalids

    messages = make_messages(sampled_data)
    output_data, total_cost, results = run_model(
        messages,
        model,
        api_key,
        system_prompt,
        user_prompt,
        ctx_prompt,
        print_messages,
    )

    output_path.write_text(json.dumps(output_data, indent=4))
    with Path("cost.csv").open("a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{total_cost}\n")

    print(confusion_matrix(results))
    print(f"\nTotal cost: ${total_cost}")


def make_messages(
    sampled_data: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], str, str, bool]]:
    messages: list[tuple[dict[str, Any], str, str, bool]] = []
    for item in sampled_data:
        context = f"Context: {item['input']}"

        entities, _ = parse_instance(item["output"])
        extraction = (
            "Extraction:\n"
            f"Cause: {' | '.join(entities['Cause'])}\n"
            f"Effect: {' | '.join(entities['Effect'])}\n"
        )

        entities, _ = parse_instance(item["gold"])
        gold = (
            "GOLD:\n"
            f"Cause: {' | '.join(entities['Cause'])}\n"
            f"Effect: {' | '.join(entities['Effect'])}\n"
        )

        answer = f"Valid?: {item['valid']}"

        answer_msg = "\n".join([gold, answer]).strip()
        gpt_msg = "\n".join([context, extraction]).strip()
        messages.append((item, answer_msg, gpt_msg, item["valid"]))

    return messages


def run_model(
    messages: list[tuple[dict[str, Any], str, str, bool]],
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    ctx_prompt: str | None,
    print_messages: bool,
) -> tuple[list[dict[str, Any]], float, dict[tuple[bool, int], int]]:
    client = openai.OpenAI(api_key=api_key)
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
            print(ctx_prompt)
            print("-" * 80)
            print(SYSTEM_PROMPTS[system_prompt])
            print("-" * 80)
            print(USER_PROMPTS[user_prompt])
            print("-" * 80)
            print(gpt_msg)
            print("-" * 80)
            print(f"\nGPT: '{result_s}'")
            print()
            print("NOT SENT", "~" * 50)
            print(answer_msg)
            print("*" * 80)
            print()

    return output_data, total_cost, results


def confusion_matrix(results: dict[tuple[bool, int], int]) -> pd.DataFrame:
    df = pd.DataFrame(list(results.items()), columns=["Combination", "Count"])
    df[["gold", "pred"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop("Combination", axis="columns")
    df["Count"] = df["Count"].astype(int)
    df = df.pivot_table(index="gold", columns="pred", values="Count", fill_value=0)

    return df


if __name__ == "__main__":
    typer.run(main)
