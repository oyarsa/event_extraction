import argparse
import json
from pathlib import Path
from typing import Any, cast

import openai


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


DEFAULT_EXTRACTION_EXAMPLES = [
    {
        "text": (
            "If one or more of Ecolab's customers were to experience a disastrous"
            " outcome, the firm's reputation could suffer and it could lose multiple"
            " customers as a result."
        ),
        "result": (
            "[Cause] one or more of Ecolab's customers were to experience a disastrous"
            " outcome [Relation] cause [Effect] the firm's reputation could suffer and"
            " it could lose multiple customers"
        ),
    },
    {
        "text": (
            "As grocery customers regularly visit the store, they are continually"
            " exposed to the firm's higher margin offerings, spurring lucrative general"
            " merchandise sales."
        ),
        "result": (
            "[Cause] they are continually exposed to the firm's higher margin offerings"
            " [Relation] cause [Effect] spurring lucrative general merchandise sales"
        ),
    },
    {
        "text": (
            "We think that QuickBooks exhibits high switching costs given the regularity"
            " in bookkeeping and the pain of transferring a business' accounting record"
            " as well as learning a new software to record entries."
        ),
        "result": (
            "[Cause] learning a new software | the regularity in bookkeeping | the"
            " pain of transferring a business' accounting record [Relation] cause"
            " [Effect] QuickBooks exhibits high switching costs"
        ),
    },
    {
        "text": (
            "We think Lululemon protects the integrity of its brand by selling only"
            " high-quality apparel and it has continued to grow and improve margins"
            " despite the introduction of competing products by others."
        ),
        "result": (
            "[Cause] the introduction of competing products by others [Relation] prevent"
            " [Effect] it has continued to grow and improve margins"
        ),
    },
    {
        "text": (
            "In times of crises a AAA or AA rating is imperative within reinsurance as a"
            " flight to quality for renewals takes hold."
        ),
        "result": (
            "[Cause] a flight to quality for renewals takes hold [Relation] enable"
            " [Effect] In times of crises a AAA or AA rating is imperative within"
            " reinsurance"
        ),
    },
]
DEFAULT_EXTRACTION_PROMPT = (
    "What are the causes, effects and relations in the following text?"
)


def extract_clauses(
    model: str,
    text: str,
    examples: list[dict[str, str]] | None = None,
    prompt: str = DEFAULT_EXTRACTION_PROMPT,
) -> dict[str, Any]:
    if examples is None:
        examples = DEFAULT_EXTRACTION_EXAMPLES

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
                make_msg("user", example["text"]),
                make_msg("assistant", example["result"]),
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


def calculate_metrics(result: str, expected: str) -> dict[str, float]:
    return {}


def run_extraction(model: str) -> dict[str, Any]:
    text = (
        "Nevertheless, with voices amplified through structural shifts like the rise"
        " of digital media, consumers have more agency than ever: if they want"
        " LaCroix (or any other National Beverage brand), retailers eventually have"
        " to oblige."
    )
    expected = (
        "[Cause] voices amplified through structural shifts [Relation] cause"
        " [Effect] consumers have more agency"
    )

    response = extract_clauses(model, text)
    metrics = calculate_metrics(get_result(response), expected)

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"\t{metric}: {value}")

    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("key_file", type=Path)
    parser.add_argument("key_name", type=str)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--print-logs", action="store_true")
    parser.add_argument("--log-file", type=Path, default="chat_log.jsonl")
    parser.add_argument(
        "--mode",
        type=str,
        default="contradiction",
        choices=["contradiction", "extraction"],
    )
    args = parser.parse_args()

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.mode == "extraction":
        response = run_extraction(args.model)
    else:
        response = gen_contradiction(args.model, "I am a cat.")

    print("Result:")
    print(get_result(response))
    print(f"\nCost: ${calculate_cost(args.model, response)}")


if __name__ == "__main__":
    main()
