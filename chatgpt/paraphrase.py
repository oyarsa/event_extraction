import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

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

PARAPHRASE_PROMPTS = [
    "Given the following causes and effects, generate a sentence:",  # 0
]


def gen_example_exchange(
    examples: list[dict[str, str]], prompt: str
) -> Iterator[dict[str, str]]:
    prompt_msg = make_msg("user", prompt)
    for example in examples:
        yield prompt_msg
        yield make_msg("user", example["context"])
        yield make_msg("assistant", example["answers"])


def gen_paraphrase(
    model: str, sentence: str, prompt: str, examples: list[dict[str, str]]
) -> dict[str, Any]:
    response = make_chat_request(
        model=model,
        messages=[
            make_msg(
                "system",
                "You are a helpful assistant that generates sentences from causes,"
                " effects and relations.",
            ),
            *gen_example_exchange(examples, prompt),
            make_msg("user", prompt),
            make_msg("user", sentence),
        ],
    )

    return response


def run_paraphrase(
    model: str,
    examples_path: Path,
    prompt: str,
    input_path: Path,
    output_path: Path | None,
) -> None:
    data = json.loads(input_path.read_text())
    demonstration_examples = json.loads(examples_path.read_text())

    inputs = [inst["sentence2"] for inst in data]
    responses = [
        gen_paraphrase(
            model=model, sentence=inst, prompt=prompt, examples=demonstration_examples
        )
        for inst in tqdm(inputs)
        if inst is not None
    ]

    output_sentences = (get_result(response) for response in responses)
    output = deepcopy(data)
    for inst, sentence in zip(output, output_sentences):
        inst["sentence2_orig"] = inst["sentence2"]
        inst["sentence2"] = sentence

    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))

    cost = sum(calculate_cost(model, response) for response in responses)
    print(f"\nCost: ${cost}")


def main() -> None:
    parser = init_argparser()
    parser.add_argument(
        "--examples",
        type=Path,
        default="./data/paraphrase/paraphrase_examples_3.json",
        help="Path to the demonstration examples.",
    )
    args = parser.parse_args()
    log_args(args, args.args_path)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.prompt < 0 or args.prompt >= len(PARAPHRASE_PROMPTS):
        raise IndexError(
            f"Invalid prompt index: {args.prompt}. Choose one between 0 and"
            f" {len(PARAPHRASE_PROMPTS)})"
        )

    run_paraphrase(
        model=args.model,
        examples_path=args.examples,
        input_path=args.input,
        output_path=args.output,
        prompt=PARAPHRASE_PROMPTS[args.prompt],
    )


if __name__ == "__main__":
    if not hasattr(__builtins__, "__IPYTHON__"):
        main()
