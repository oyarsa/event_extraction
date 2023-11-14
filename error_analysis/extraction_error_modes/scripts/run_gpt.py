# pyright: basic
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import typer
from openai import OpenAI


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


def run_gpt(client: OpenAI, system_prompt: str, message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
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
    return result or "<empty>"


def main(
    file: Path, n: int = 10, rand: bool = True, key_file: Path = Path("key")
) -> None:
    """
    \b
    - file: Path to the json file containing the data (list of objects with keys 'input',
        'output', 'gold', 'valid')
    - n: Number of examples to run
    - rand: Whether to shuffle the data before selecting n examples
    - key_file: Path to the file containing the OpenAI API key (simple text file
        containing only the key)
    """
    system_prompt = "You are a helpful assistant that can take a context and a cause, effect and relation extraction, and determine whether that extraction is valid."
    user_prompt = "Given the context, is the extraction (cause, effect and relation) valid? Respond with either 'true' or 'false'."

    data = json.loads(file.read_text())
    if rand:
        random.shuffle(data)

    n = n or len(data)
    if n % 2 != 0:
        n += 1

    valids = [item for item in data if item["valid"]][: n // 2]
    invalids = [item for item in data if not item["valid"]][: n // 2]

    messages: list[tuple[str, str, bool]] = []
    for item in valids + invalids:
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

        full_msg = "\n\n".join([user_prompt, context, extraction, gold, answer]).strip()
        gpt_msg = "\n\n".join([user_prompt, context, extraction]).strip()
        messages.append((full_msg, gpt_msg, item["valid"]))

    client = OpenAI(api_key=key_file.read_text().strip())
    results: dict[tuple[bool, bool], int] = defaultdict(int)
    for full_msg, gpt_msg, valid in messages:
        print(full_msg)
        result = run_gpt(client, system_prompt, gpt_msg)
        print(f"\nGPT: '{result}'")
        print("-" * 80)
        print()

        results[(valid, "true" in result.lower().strip())] += 1

    print(json.dumps({str(k): v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    typer.run(main)
