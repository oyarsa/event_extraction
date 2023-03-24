import argparse
import json
from pathlib import Path
from typing import Any

import tiktoken

MODEL_COSTS = {
    "gpt-3.5-turbo": 0.000002,  # $0.002 / 1K tokens
}


def parse_prompt(instance: dict[str, Any]) -> list[str]:
    return [msg["content"] for msg in instance["params"]["messages"][:-1]]


def parse_log(data: list[dict[str, Any]]) -> list[list[str]]:
    output: list[list[str]] = []
    for instances in data:
        content: list[str] = []
        for msg in instances["params"]["messages"]:
            content.append(msg["content"])
        content.append(instances["response"]["choices"][0]["message"]["content"])

        output.append(content)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    with args.input.open() as f:
        data: list[dict[str, Any]] = [json.loads(line) for line in f]

    encoding = tiktoken.encoding_for_model(args.model)
    log_data = parse_log(data)

    log_tokens: list[int] = []
    for instances in log_data:
        log_tokens.append(sum(len(encoding.encode(inst)) for inst in instances))

    print("Tokens per instance (sample):")
    print("\n".join(str(i) for i in log_tokens[:3] + log_tokens[-3:]))

    total_tokens = sum(log_tokens)
    cost = total_tokens * MODEL_COSTS[args.model]
    print(f"Estimated cost: ${cost:.2f} for {total_tokens} tokens")

    prompt = parse_prompt(data[1])
    prompt_tokens = sum(len(encoding.encode(msg)) for msg in prompt)
    prompt_cost = prompt_tokens * MODEL_COSTS[args.model]
    print(f"Estimated cost: ${prompt_cost} for {prompt_tokens} prompt tokens")


if __name__ == "__main__":
    main()
