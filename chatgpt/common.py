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


def log_args(args: argparse.Namespace, path: Path | None) -> None:
    args_dict = vars(args).copy()
    for key, value in args_dict.items():
        if isinstance(value, Path):
            args_dict[key] = str(value)

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(args_dict))
    else:
        print(json.dumps(args_dict, indent=2))


def init_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("key_file", type=Path, help="Path to JSON file with API keys")
    parser.add_argument("key_name", type=str, help="Name of key to use")
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="Model to use"
    )
    parser.add_argument(
        "--print-logs", action="store_true", help="Print logs to stdout"
    )
    parser.add_argument(
        "--log-file", type=Path, default="chat_log.jsonl", help="Log file"
    )
    parser.add_argument("--input", "-i", type=Path, help="Input file")
    parser.add_argument("--output", "-o", type=Path, help="Output file for predictions")
    parser.add_argument(
        "--prompt",
        type=int,
        default=0,
        help="Prompt index to use for the chat session",
    )
    parser.add_argument("--metrics-path", type=Path, help="Path where to save metrics")
    parser.add_argument("--args-path", type=Path, help="Path where to save args")
    return parser
