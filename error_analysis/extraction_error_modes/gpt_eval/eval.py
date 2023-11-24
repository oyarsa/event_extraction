#!/usr/bin/env python3
# pyright: basic
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from openai import OpenAI
from tqdm import tqdm


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


def run_gpt(
    client: OpenAI, model: str, system_prompt: str, message: str
) -> tuple[str, float]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=0,
    )
    result = response.choices[0].message.content
    cost = calculate_cost(model, response)
    return result or "<empty>", cost


SYSTEM_PROMPTS = {
    "simple": """\
You are a helpful assistant that can take a context and an extraction composed of a
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


def main(
    file: Path,
    n: int = 10,
    rand: bool = True,
    key_file: Path = Path("key"),
    model: str = "gpt-4",
    system_prompt: str = "simple",
    user_prompt: str = "instructions",
    print_messages: bool = True,
) -> None:
    """
    Run a GPT model on the given data and evaluate the results.

    \b
    - file: Path to the json file containing the data (list of objects with keys
        'input', 'output', 'gold', 'valid')
    - n: Number of examples to run
    - rand: Whether to shuffle the data before selecting n examples
    - key_file: Path to the file containing the OpenAI API key (simple text file
        containing only the key)
    - model: Which GPT model to use (gpt-3.5-turbo or gpt-4)
    - system_prompt: Which system prompt to use (only 'simple' for now)
    - user_prompt: Which user prompt to use ('simple' or 'instructions')
    - print_messages: Whether to print the prompt, context, gold and prediction. If
        false, only the progress bar and evaluation results are printed.
    """

    if model not in MODEL_COSTS:
        raise ValueError(f"Invalid model. Options: {list(MODEL_COSTS.keys())}")
    if system_prompt not in SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system prompt. Options: {SYSTEM_PROMPTS.keys()}")
    if user_prompt not in USER_PROMPTS:
        raise ValueError(f"Invalid user prompt. Options: {USER_PROMPTS.keys()}")

    api_key = key_file.read_text().strip()

    data = json.loads(file.read_text())
    if rand:
        random.shuffle(data)

    n = n or len(data)
    if n % 2 != 0:
        n += 1

    valids = [item for item in data if item["valid"]][: n // 2]
    invalids = [item for item in data if not item["valid"]][: n // 2]
    sampled_data = valids + invalids

    messages: list[tuple[str, str, bool]] = []
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
            "Gold:\n"
            f"Cause: {' | '.join(entities['Cause'])}\n"
            f"Effect: {' | '.join(entities['Effect'])}\n"
        )

        answer = f"Valid?: {item['valid']}"

        display_msg = "\n\n".join([context, extraction, gold, answer]).strip()
        gpt_msg = "\n\n".join([USER_PROMPTS[user_prompt], context, extraction]).strip()
        messages.append((display_msg, gpt_msg, item["valid"]))

    client = OpenAI(api_key=api_key)
    results: defaultdict[tuple[bool, int], int] = defaultdict(int)
    total_cost = 0

    for display_msg, gpt_msg, valid in tqdm(messages):
        result_s, cost = run_gpt(client, model, SYSTEM_PROMPTS[system_prompt], gpt_msg)
        total_cost += cost

        last_line = result_s.splitlines()[-1].replace("Score:", "").strip()
        result = int(last_line) if last_line.isdigit() else 0
        results[(valid, result)] += 1

        if print_messages:
            print(display_msg)
            print(f"\nGPT: '{result_s}'")
            print("-" * 80)
            print()

    with Path("cost.csv").open("a") as f:
        ts = datetime.now(timezone.utc).isoformat()
        f.write(f"{ts},{total_cost}\n")

    # Format results as a confusion matrix
    df = pd.DataFrame(list(results.items()), columns=["Combination", "Count"])
    df[["gold", "pred"]] = pd.DataFrame(df["Combination"].tolist(), index=df.index)
    df = df.drop("Combination", axis="columns")
    df["Count"] = df["Count"].astype(int)
    df = df.pivot_table(index="gold", columns="pred", values="Count", fill_value=0)

    print(df)
    print(f"\nTotal cost: ${total_cost}")


if __name__ == "__main__":
    typer.run(main)
