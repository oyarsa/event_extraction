import json
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

PARAPHRASE_PROMPT = "Paraphrase the following text:"


def gen_paraphrase(model: str, sentence: str, prompt: str) -> dict[str, Any]:
    response = make_chat_request(
        model=model,
        messages=[
            make_msg(
                "system",
                "You are a helpful assistant that paraphrases sentences using different"
                " words but the same meaning.",
            ),
            make_msg("user", prompt),
            make_msg("user", sentence),
        ],
    )

    return response


def run_paraphrase(
    model: str,
    input_path: Path,
    output_path: Path | None,
) -> None:
    data = json.loads(input_path.read_text())
    sentences = [d["sentence2"] for d in data if d["sentence2"].strip()]

    responses: list[dict[str, Any]] = []
    for sentence in tqdm(sentences):
        responses.append(
            gen_paraphrase(model=model, sentence=sentence, prompt=PARAPHRASE_PROMPT)
        )

    output_sentences = (get_result(response) for response in responses)
    output = deepcopy(data)
    for sentence, new_sentence in zip(output, output_sentences):
        sentence["sentence2_orig"] = sentence["sentence2"]
        sentence["sentence2"] = new_sentence

    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))

    cost = sum(calculate_cost(model, response) for response in responses)
    print(f"\nCost: ${cost}")


def main() -> None:
    parser = init_argparser(prompt=False)
    args = parser.parse_args()
    log_args(args, args.args_path)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    run_paraphrase(model=args.model, input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
