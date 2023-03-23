import json
from pathlib import Path
from typing import Any, NamedTuple

import openai
from tqdm import tqdm

from common import (
    calculate_cost,
    get_key,
    get_result,
    init_argparser,
    log_args,
    logger,
    make_chat_request,
    make_msg,
)
from metrics import (
    MetricPrediction,
    MetricReference,
    StructureFormat,
    calculate_metrics,
)

EXTRACTION_PROMPTS = [
    "What are the causes, effects and relations in the following text?",  # 0
]


def make_extraction_request(
    model: str,
    text: str,
    examples: list[dict[str, str]],
    prompt: str,
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


class ExtractionResult(NamedTuple):
    responses: list[dict[str, Any]]
    predictions: list[MetricPrediction]
    metrics: dict[str, float]


def extract_clauses(
    model: str,
    demonstration_examples: list[dict[str, str]],
    inputs: list[MetricReference],
    extraction_mode: StructureFormat,
    prompt: str,
) -> ExtractionResult:
    responses: list[dict[str, Any]] = []
    predictions: list[MetricPrediction] = []

    for example in tqdm(inputs):
        response = make_extraction_request(
            model, example["context"], demonstration_examples, prompt
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
    metric_path: Path | None,
    extraction_mode: StructureFormat,
    prompt: str,
) -> None:
    demonstration_examples = json.loads(demonstration_examples_path.read_text())
    examples: list[MetricReference] = json.loads(input_path.read_text())["data"]

    responses, predictions, metrics = extract_clauses(
        model=model,
        demonstration_examples=demonstration_examples,
        inputs=examples,
        extraction_mode=extraction_mode,
        prompt=prompt,
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

    metrics["cost"] = sum(calculate_cost(model, response) for response in responses)
    if metric_path is None:
        print("\nMetrics:")
        print(json.dumps(metrics, indent=2))
    else:
        metric_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    parser = init_argparser()
    parser.add_argument(
        "--examples",
        type=Path,
        default="data/tags/extraction_examples_3.json",
    )
    parser.add_argument(
        "--mode",
        default="tags",
        choices=["tags", "lines"],
        type=StructureFormat,
    )
    args = parser.parse_args()
    log_args(args, path=args.args_path)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.prompt < 0 or args.prompt >= len(EXTRACTION_PROMPTS):
        raise IndexError(f"Invalid prompt index: {args.prompt}")
    run_extraction(
        model=args.model,
        demonstration_examples_path=args.examples,
        input_path=args.input,
        output_path=args.output,
        metric_path=args.metrics_path,
        extraction_mode=args.mode,
        prompt=EXTRACTION_PROMPTS[args.prompt],
    )


if __name__ == "__main__":
    if not hasattr(__builtins__, "__IPYTHON__"):
        main()
