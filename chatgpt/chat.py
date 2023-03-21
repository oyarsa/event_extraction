import argparse
import json
from pathlib import Path
from typing import Any, NamedTuple, cast

import openai
from tqdm import tqdm

from metrics import (
    MetricPrediction,
    MetricReference,
    StructureFormat,
    calculate_metrics,
)


class ExchangeLogger:
    def __init__(self) -> None:
        self.file: Path | None = None
        self.print_log = False

    def config(self, file: Path, print_log: bool) -> None:
        self.file = file
        self.print_log = print_log

    def log_exchange(self, params: dict[str, Any], response: dict[str, Any]) -> None:
        assert self.file is not None, "Must call config() before logging exchanges."

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


DEFAULT_CONTADICTION_PROMPT = "Generate a sentence that contradicts the following:"


def gen_contradiction(
    model: str, sentence: str, prompt: str = DEFAULT_CONTADICTION_PROMPT
) -> dict[str, Any]:
    response = make_chat_request(
        model=model,
        messages=[
            make_msg(
                "system", "You are a helpful assistant that generates contradictions."
            ),
            make_msg("user", prompt),
            make_msg("user", sentence),
        ],
    )

    return response


DEFAULT_EXTRACTION_PROMPT = (
    "What are the causes, effects and relations in the following text?"
)


def make_extraction_request(
    model: str,
    text: str,
    examples: list[dict[str, str]],
    prompt: str = DEFAULT_EXTRACTION_PROMPT,
) -> dict[str, Any]:
    response = make_chat_request(
        model=model, messages=generate_extraction_messages(text, examples, prompt)
    )

    return response


def gen_extraction_example_exchange(
    examples: list[dict[str, str]], prompt: str
) -> list[dict[str, str]]:
    prompt_msg = make_msg("user", prompt)
    messages: list[dict[str, str]] = []

    for example in examples:
        messages.extend(
            [
                prompt_msg,
                make_msg("user", example["context"]),
                make_msg("assistant", example["answers"]),
            ]
        )

    return messages


def generate_extraction_messages(
    text: str, examples: list[dict[str, str]], prompt: str
) -> list[dict[str, str]]:
    return [
        make_msg(
            "system",
            "You are a helpful assistant that extract causes, effects, and"
            " relations.",
        ),
        *gen_extraction_example_exchange(examples, prompt),
        make_msg("user", prompt),
        make_msg("user", text),
    ]


def get_result(response: dict[str, Any]) -> str:
    return response["choices"][0]["message"]["content"]


MODEL_COSTS = {
    "gpt-3.5-turbo": 0.000002,  # $0.002 / 1K tokens
}


def calculate_cost(model: str, response: dict[str, Any]) -> float:
    num_tokens = response["usage"]["total_tokens"]
    return MODEL_COSTS[model] * num_tokens


class ExtractionResult(NamedTuple):
    responses: list[dict[str, Any]]
    predictions: list[MetricPrediction]
    metrics: dict[str, float]


def extract_clauses(
    model: str,
    demonstration_examples: list[dict[str, str]],
    inputs: list[MetricReference],
    extraction_mode: StructureFormat,
) -> ExtractionResult:
    responses: list[dict[str, Any]] = []
    predictions: list[MetricPrediction] = []

    for example in tqdm(inputs):
        response = make_extraction_request(
            model, example["context"], demonstration_examples
        )
        pred: MetricPrediction = {
            "id": example["id"],
            "prediction_text": get_result(response),
        }
        responses.append(response)
        predictions.append(pred)

    metrics = calculate_metrics(predictions, inputs, extraction_mode)

    return ExtractionResult(responses, predictions, metrics)


def run_extraction(
    model: str,
    demonstration_examples_path: Path,
    input_path: Path,
    output_path: Path | None,
    extraction_mode: StructureFormat,
) -> None:
    demonstration_examples = json.loads(demonstration_examples_path.read_text())
    examples: list[MetricReference] = json.loads(input_path.read_text())["data"]

    responses, predictions, metrics = extract_clauses(
        model, demonstration_examples, examples, extraction_mode
    )

    output = [
        {
            "id": example["id"],
            "text": example["context"],
            "pred": pred["prediction_text"],
            "answer": example["answers"],
        }
        for example, pred in zip(examples, predictions)
    ]
    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    cost = sum(calculate_cost(model, response) for response in responses)
    print(f"\nTotal cost: ${cost}")


def run_contradiction(model: str) -> None:
    response = gen_contradiction(model, "I am a cat.")
    print("Result:")
    print(get_result(response))
    print(f"\nCost: ${calculate_cost(model, response)}")


def print_args(args: argparse.Namespace) -> None:
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key_file", type=Path)
    parser.add_argument("key_name", type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--log-file", type=Path, default="chat_log.jsonl")
    parser.add_argument(
        "--task",
        type=str,
        default="extraction",
        choices=["contradiction", "extraction"],
    )
    parser.add_argument("--extraction-examples", type=Path)
    parser.add_argument("--input", "-i", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument(
        "--extraction-mode",
        default="tags",
        choices=["tags", "lines"],
        type=StructureFormat,
    )
    args = parser.parse_args()
    print_args(args)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.task == "extraction":
        run_extraction(
            args.model,
            args.extraction_examples,
            args.input,
            args.output,
            args.extraction_mode,
        )
    else:
        run_contradiction(args.model)


if __name__ == "__main__":
    main()
