import json
from pathlib import Path
from typing import Any

import openai

from common import (
    calculate_cost,
    get_key,
    get_result,
    init_argparser,
    logger,
    make_chat_request,
    make_msg,
    print_args,
)

CONTRADICTION_SENTENCE_PROMPTS = [
    "Generate a sentence that contradicts the following:",  # 0
]
CONTRADICTION_STRUCTURED_PROMPTS = [
    "Given the following causes and effects, generate a sentence:",  # 0
]


def gen_contradiction_sentence(
    model: str, sentence: str, prompt: str
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


def run_contradiction_sentence(
    model: str,
    prompt: str,
    input_path: Path,
    output_path: Path | None,
) -> None:
    response = gen_contradiction_sentence(model, "I am a cat.", prompt)
    # _examples = json.loads(input_path.read_text())["data"]

    output = [get_result(response)]
    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))

    print(f"\nCost: ${calculate_cost(model, response)}")


def main() -> None:
    parser = init_argparser()
    args = parser.parse_args()
    print_args(args)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.prompt < 0 or args.prompt >= len(CONTRADICTION_SENTENCE_PROMPTS):
        raise ValueError(f"Invalid prompt index: {args.prompt}")
    run_contradiction_sentence(
        model=args.model,
        input_path=args.input,
        output_path=args.output,
        prompt=CONTRADICTION_SENTENCE_PROMPTS[args.prompt],
    )


if __name__ == "__main__":
    if not hasattr(__builtins__, "__IPYTHON__"):
        main()
