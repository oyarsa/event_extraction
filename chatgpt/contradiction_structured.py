import json
from pathlib import Path
from typing import Any

import openai

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
from metrics import StructureFormat, parse_instance_tags

CONTRADICTION_STRUCTURED_PROMPTS = [
    "Given the following causes and effects, generate a sentence:",  # 0
]


def gen_contradiction_structured(
    model: str, sentence: str, prompt: str
) -> dict[str, Any]:
    response = make_chat_request(
        model=model,
        messages=[
            make_msg(
                "system",
                "You are a helpful assistant that generates sentences from causes,"
                " effects and relations.",
            ),
            make_msg("user", prompt),
            make_msg("user", sentence),
        ],
    )

    return response


def tag_sort_key(tag: str) -> tuple[int, str]:
    for i, c in enumerate("CRE"):
        if tag.startswith("[" + c):
            return i, tag
    assert False, f"Unknown tag: {tag}"


def generate_answer_lines(events: dict[str, list[str]], relation: str) -> str:
    out: list[str] = []
    for ev_type, evs in events.items():
        event = f"{ev_type}: " + " | ".join(evs)
        out.append(event)
    if relation:
        out.append(f"Relation: {relation}")
    return "\n".join(sorted(out))


def generate_answer_tags(events: dict[str, list[str]], relation: str) -> str:
    out: list[str] = []
    for ev_type, evs in events.items():
        event = f"[{ev_type}] " + " | ".join(evs)
        out.append(event.strip())
    if relation:
        out.append(f"[Relation] {relation}")
    return " ".join(sorted(out, key=tag_sort_key))


def swap_clauses(instance: str, format: StructureFormat) -> str:
    entities, relation = parse_instance_tags(instance)
    assert relation is not None

    entities["Cause"], entities["Effect"] = entities["Effect"], entities["Cause"]
    if format == StructureFormat.TAGS:
        return generate_answer_tags(entities, relation)
    else:
        return generate_answer_lines(entities, relation)


def run_contradiction_structured(
    model: str,
    prompt: str,
    input_path: Path,
    output_path: Path | None,
    format: StructureFormat,
) -> None:
    data = json.loads(input_path.read_text())["data"]
    inputs = [swap_clauses(i["answers"], format) for i in data]
    responses = [gen_contradiction_structured(model, inst, prompt) for inst in inputs]
    output = [get_result(response) for response in responses]

    if output_path is not None:
        output_path.write_text(json.dumps(output, indent=2))
    else:
        print(json.dumps(output, indent=2))

    cost = sum(calculate_cost(model, response) for response in responses)
    print(f"\nCost: ${cost}")


def main() -> None:
    parser = init_argparser()
    args = parser.parse_args()
    args.add_argument(
        "--structured-format",
        type=StructureFormat,
        default="tags",
        choices=["tags", "lines"],
        help="The format of the structured input.",
    )
    log_args(args, args.args_path)

    logger.config(args.log_file, args.print_logs)
    openai.api_key = get_key(args.key_file, args.key_name)

    if args.prompt < 0 or args.prompt >= len(CONTRADICTION_STRUCTURED_PROMPTS):
        raise IndexError(
            f"Invalid prompt index: {args.prompt}. Choose one between 0 and"
            f" {len(CONTRADICTION_STRUCTURED_PROMPTS)})"
        )

    run_contradiction_structured(
        model=args.model,
        # Input should be tag-based, regardless of the structured format.
        input_path=args.input,
        output_path=args.output,
        prompt=CONTRADICTION_STRUCTURED_PROMPTS[args.prompt],
        format=args.structured_format,
    )


if __name__ == "__main__":
    if not hasattr(__builtins__, "__IPYTHON__"):
        main()
