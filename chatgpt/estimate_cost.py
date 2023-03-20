import argparse
import json
from pathlib import Path

import tiktoken

# $0.002 / 1K tokens
PRICE_PER_TOKEN = 0.000002
DEFAULT_PROMPT = "What are the causes, effects and relations in the following text?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, default="../data/raw")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    args = parser.parse_args()

    splits = ["train", "dev", "test"]
    data: list[dict[str, str]] = []
    for split in splits:
        split_data = json.loads(
            (args.input / f"event_dataset_{split}.json").read_text()
        )
        data.extend(split_data)

    encoding = tiktoken.encoding_for_model(args.model)

    data_tokens = sum(len(encoding.encode(d["info"])) for d in data)
    prompt_tokens = len(encoding.encode(args.prompt)) * len(data)

    total_tokens = data_tokens + prompt_tokens
    cost = total_tokens * PRICE_PER_TOKEN

    print(f"Estimated cost: ${cost:.2f} for {total_tokens} tokens")


if __name__ == "__main__":
    main()
