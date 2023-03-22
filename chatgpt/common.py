import argparse
import json
from pathlib import Path
from typing import Any, cast

import openai


class ExchangeLogger:
    def __init__(self) -> None:
        self.file: Path | None = None
        self.print_log = False

    def config(self, file: Path, print_log: bool = False) -> None:
        self.file = file
        self.print_log = print_log

    def log_exchange(self, params: dict[str, Any], response: dict[str, Any]) -> None:
        if self.file is None:
            raise ValueError("Must call config() before logging exchanges.")

        log = {"params": params, "response": response}

        with self.file.open("a") as f:
            json.dump(log, f)
            f.write("\n")

        if self.print_log:
            print(json.dumps(log, indent=2))
            print()


logger = ExchangeLogger()


def get_key(key_file: Path, key_name: str) -> str:
    keys = json.loads(key_file.read_text())
    return keys[key_name]


def make_msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def make_chat_request(**kwargs: Any) -> dict[str, Any]:
    response = cast(dict[str, Any], openai.ChatCompletion.create(**kwargs))
    logger.log_exchange(kwargs, response)
    return response


def get_result(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


MODEL_COSTS = {
    "gpt-3.5-turbo": 0.000002,  # $0.002 / 1K tokens
}


def calculate_cost(model: str, response: dict[str, Any]) -> float:
    num_tokens = response["usage"]["total_tokens"]
    return MODEL_COSTS[model] * num_tokens


def print_args(args: argparse.Namespace) -> None:
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()


def init_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("key_file", type=Path)
    parser.add_argument("key_name", type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--log-file", type=Path, default="chat_log.jsonl")
    parser.add_argument("--input", "-i", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--prompt", type=int, default=0)
    parser.add_argument("--metrics-path", type=Path)
    return parser
