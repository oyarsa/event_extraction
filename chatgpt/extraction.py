import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import openai
from tqdm import tqdm

from common import (
    MODEL_COSTS,
    ChatCompletion,
    calculate_cost,
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
    parse_instance,
)

USER_PROMPTS = [
    # 0
    "What are the causes, effects and relations in the following text?",
    # 1
    """\
What are the causes, effects and relation in the following text?
The relation MUST BE one of "cause", "enable", or "prevent".
The causes and effects must be spans of the text. There is only one relation.

The response should be formatted as this:
Cause: <text>
Effect: <text>
Relation: <text>

The relation MUST BE one of "cause", "enable", or "prevent". If it is anything else \
besides those three words, you will be penalized.

When there are multiple causes or effects, separate them by " | ". Don't add quotes \
around the extractions.
""",
    # 2
    """\
What are the causes, effects and relation in the following text?
The relation must be one of "cause", "enable", or "prevent".
The causes and effects must be spans of the text. There is only one relation.

The response should be formatted as this:
`[Cause] <cause text> [Relation] <relation> [Effect] <effect text>`

When there are multiple causes or effects, separate them by " | ". Don't add quotes \
around the extractions.
""",
    # 3
    """\
What are the causes, effects and relation in the following text? The relation must be \
one of "cause", "enable", or "prevent". The causes and effects must be spans of the \
text. There is only one relation.

The response should be formatted as this:
Cause: <text>
Effect: <text>
Relation: <text>

When there are multiple causes or effects, separate them by " | ". Don't add quotes
around the extractions.
""",
    # 4
    """\
What are the causes, effects and relations in the following text? \
The relation must be one of "cause", "enable", or "prevent".
""",
    # 5 - FinCausal (no relation)
    """\
What are the causes and effects in the following text?
The causes and effects must be substrings of the text with no external words.

The response should be formatted as this:
Cause: <text>
Effect: <text>
""",
    # 6 - MAVEN (cause only)
    """\
What is the cause in the following text?
The cause must be a substring of the text with no external words.

The response should be formatted as this:
<cause text>
""",
]
SYSTEM_PROMPTS = [
    # 0
    "You are a helpful assistant that extract causes, effects, and"
    " relations from text. The format is `[Cause] <cause text> [Relation]"
    " <relation> [Effect] <effect text>`.",
    # 1
    "You are a helpful assistant that extract causes, effects, and a relation from"
    " text.",
    # 2
    "You are a helpful assistant that extract causes, effects, and a relation from"
    " text. The relation must be one of 'cause', 'enable', or 'prevent'.",
    # 3
    "You are a helpful assistant that extract causes and effects from text.",
]


def make_extraction_request(
    client: openai.OpenAI,
    model: str,
    text: str,
    examples: list[dict[str, str]],
    user_prompt: str,
    system_prompt: str,
    appendix: Sequence[str] = (),
) -> ChatCompletion:
    return make_chat_request(
        client=client,
        model=model,
        messages=generate_extraction_messages(
            text, examples, user_prompt, system_prompt, appendix=appendix
        ),
        temperature=0,
        seed=0,
    )


def gen_extraction_example_exchange(
    examples: list[dict[str, str]], prompt: str
) -> Iterator[dict[str, str]]:
    prompt_msg = make_msg("user", prompt)
    for example in examples:
        yield prompt_msg
        yield make_msg("user", example["context"])
        yield make_msg("assistant", example["answers"])


def generate_extraction_messages(
    text: str,
    examples: list[dict[str, str]],
    user_prompt: str,
    system_prompt: str,
    appendix: Sequence[str] = (),
) -> list[dict[str, str]]:
    return [
        make_msg(
            "system",
            system_prompt,
        ),
        *gen_extraction_example_exchange(examples, user_prompt),
        make_msg("user", user_prompt),
        make_msg("user", text),
        *(make_msg("user", s) for s in appendix),
    ]


class ExtractionResult(NamedTuple):
    responses: list[ChatCompletion]
    predictions: list[MetricPrediction]
    metrics: dict[str, float]


def retry_relation(
    client: openai.OpenAI,
    model: str,
    example: MetricReference,
    demonstration_examples: list[dict[str, str]],
    user_prompt: str,
    system_prompt: str,
    relation: str,
    max_retries: int = 5,
) -> str:
    """
    Retry getting a relation from the user until we get 'cause', 'enable' or 'prevent'
    or we reach the maximum number of retries.
    """
    num_tries = 0
    appendix: list[str] = []

    while relation not in ["cause", "enable", "prevent"] and num_tries < max_retries:
        num_tries += 1
        appendix.append(
            f'Invalid relation: "{relation}". Provide a valid relation (one of "cause",'
            ' "enable", "prevent"). The output should be only the relation.'
        )

        response = make_extraction_request(
            client,
            model,
            example["context"],
            demonstration_examples,
            user_prompt,
            system_prompt,
            appendix=appendix,
        )
        relation = get_result(response).lower().strip()

    if num_tries >= max_retries:
        print(f"Failed to get a valid relation after {max_retries} retries")
        print("\n".join(appendix))
        print()

    return relation


def construct_instance_tags(entities: dict[str, list[str]], relation: str) -> str:
    tags = [
        f'[Cause] {" | ".join(entities["Cause"])}',
        f"[Relation] {relation}",
        f'[Effect] {" | ".join(entities["Effect"])}',
    ]
    return " ".join(tags)


def construct_instance_lines(entities: dict[str, list[str]], relation: str) -> str:
    rows = [
        f'Cause: {" | ".join(entities["Cause"])}',
        f'Effect: {" | ".join(entities["Effect"])}',
        f"Relation: {relation}",
    ]
    return "\n".join(rows)


def replace_relation(result: str, relation: str, mode: StructureFormat) -> str:
    entities, _ = parse_instance(result, mode)

    match mode:
        case StructureFormat.TAGS:
            return construct_instance_tags(entities, relation)
        case StructureFormat.LINES:
            return construct_instance_lines(entities, relation)
        case StructureFormat.LINES_NO_RELATION | StructureFormat.STRAIGHT:
            return result


def extract_clauses(
    client: openai.OpenAI,
    model: str,
    demonstration_examples: list[dict[str, str]],
    inputs: list[MetricReference],
    extraction_mode: StructureFormat,
    user_prompt: str,
    system_prompt: str,
) -> ExtractionResult:
    responses: list[ChatCompletion] = []
    predictions: list[MetricPrediction] = []

    for example in tqdm(inputs):
        response = make_extraction_request(
            client,
            model,
            example["context"],
            demonstration_examples,
            user_prompt,
            system_prompt,
        )
        result = get_result(response)
        relation = parse_instance(result, extraction_mode)[1].lower()

        if relation not in ["cause", "enable", "prevent"]:
            relation = retry_relation(
                client,
                model,
                example,
                demonstration_examples,
                user_prompt,
                system_prompt,
                relation,
            )
            result = replace_relation(result, relation, extraction_mode)

        pred: MetricPrediction = {
            "id": example["id"],
            "prediction_text": result,
        }
        responses.append(response)
        predictions.append(pred)

    metrics = calculate_metrics(predictions, inputs, extraction_mode)

    return ExtractionResult(responses, predictions, metrics)


def run_extraction(
    client: openai.OpenAI,
    model: str,
    demonstration_examples_path: Path | None,
    input_path: Path,
    output_path: Path | None,
    metric_path: Path | None,
    extraction_mode: StructureFormat,
    user_prompt: str,
    system_prompt: str,
) -> None:
    if demonstration_examples_path is not None:
        demonstration_examples = json.loads(demonstration_examples_path.read_text())
    else:
        demonstration_examples = []

    data = json.loads(input_path.read_text())
    if "data" in data:
        data = data["data"]
    examples: list[MetricReference] = data

    responses, predictions, metrics = extract_clauses(
        client,
        model=model,
        demonstration_examples=demonstration_examples,
        inputs=examples,
        extraction_mode=extraction_mode,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
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
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    if metric_path is not None:
        metric_path.write_text(json.dumps(metrics, indent=2))


def init_client(api_type: str, config: dict[str, Any]) -> openai.OpenAI:
    "Create client for OpenAI or Azure API, depending on the configuration."
    config = config[api_type]

    if api_type.startswith("azure"):
        print("Using Azure OpenAI API")
        return openai.AzureOpenAI(
            api_key=config["key"],
            api_version=config["api_version"],
            azure_endpoint=config["endpoint"],
            azure_deployment=config["deployment"],
        )
    elif api_type.startswith("openai"):
        print("Using OpenAI API")
        return openai.OpenAI(api_key=config["key"])
    else:
        raise ValueError(f"Unknown API type: {config['api_type']}")


def list_prompts() -> None:
    print("User prompts:")
    for i, prompt in enumerate(USER_PROMPTS):
        print(i)
        print("----")
        print(prompt)
        print()
    print()
    print("=" * 80)
    print()
    print("System prompts:")
    for i, prompt in enumerate(SYSTEM_PROMPTS):
        print(i)
        print("----")
        print(prompt)
        print()


def main() -> None:
    parser = init_argparser()
    parser.add_argument(
        "--examples",
        type=Path,
        help="The path to file with the demonstration examples.",
    )
    parser.add_argument(
        "--mode",
        default="tags",
        choices=["tags", "lines", "lines_no_relation", "straight"],
        type=StructureFormat,
        help="The format of the structured output.",
    )
    parser.add_argument("--list-prompts", action="store_true", help="List prompts")
    args = parser.parse_args()

    if args.list_prompts:
        list_prompts()
        return

    log_args(args, path=args.args_path)
    logger.config(args.log_file, args.print_logs)

    api_config = json.loads(args.key_file.read_text())
    client = init_client(args.key_name, api_config)

    if args.prompt < 0 or args.prompt >= len(USER_PROMPTS):
        raise IndexError(f"Invalid user prompt index: {args.prompt}")

    if args.sys_prompt < 0 or args.sys_prompt >= len(USER_PROMPTS):
        raise IndexError(f"Invalid system prompt index: {args.prompt}")

    if args.model not in MODEL_COSTS:
        raise ValueError(f"Model {args.model} not found in costs table.")

    run_extraction(
        client,
        model=args.model,
        demonstration_examples_path=args.examples,
        input_path=args.input,
        output_path=args.output,
        metric_path=args.metrics_path,
        extraction_mode=args.mode,
        user_prompt=USER_PROMPTS[args.prompt],
        system_prompt=SYSTEM_PROMPTS[args.sys_prompt],
    )


if __name__ == "__main__":
    main()
