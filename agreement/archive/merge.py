import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Result:
    gold: int
    pred: int
    passage: str
    output: str
    annotation: str


@dataclass
class ModelEntry:
    input: str
    annotation: str
    output: str
    reward_label: str


def convert_model(data: dict[str, Any]) -> ModelEntry:
    # sourcery skip: default-get
    """Convert model output to match base data format.

    The data is expected to be in the following format:
    - 'context' or 'input': the context or input.
    - 'answers' or 'gold': the gold answer.
    - 'reward_label': the model's output.

    All values are strings.
    """
    return ModelEntry(
        # The explicit check is necessary because 'input' and 'gold' might not exist,
        # which would break the 'get' version
        input=data["context"] if "context" in data else data["input"],
        annotation=data["answers"] if "answers" in data else data["gold"],
        output=data["output"] if "output" in data else data["rl_extract_txt"],
        reward_label=data["reward_label"].casefold().strip(),
    )


def match_data(
    base: list[dict[str, Any]], model: list[ModelEntry], true_class: str
) -> list[Result]:
    matches: list[Result] = []

    count = defaultdict(int)

    for b in base:
        for m in model:
            if (
                b["input"] == m.input
                and b["gold"] == m.annotation
                and b["output"] == m.output
            ):
                count[b["input"] + b["gold"] + b["output"]] += 1
                if count[b["input"] + b["gold"] + b["output"]] > 1:
                    continue
                matches.append(
                    Result(
                        gold=int(b["valid"]),
                        pred=int(m.reward_label == true_class),
                        passage=b["input"],
                        output=m.output,
                        annotation=b["gold"],
                    )
                )
        if count[b["input"] + b["gold"] + b["output"]] == 0:
            print("No match for:")
            print("Input:", b["input"])
            print("Gold:", b["gold"])
            print("Output:", b["output"])
            print()

    print([(k, v) for k, v in count.items() if v > 1])
    print("base", len(base))
    print("model", len(model))
    print("matches", len(matches))
    return matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("human_file", help="Path to human data.", type=Path)
    parser.add_argument("model_file", help="Path to model data.", type=Path)
    parser.add_argument("output_file", help="Path to output file.", type=Path)
    parser.add_argument(
        "--true-class", help="True class for model data.", default="valid"
    )
    args = parser.parse_args()

    human = json.loads(args.human_file.read_text())
    model = [convert_model(d) for d in json.loads(args.model_file.read_text())]

    matched = match_data(human, model, args.true_class)
    args.output_file.write_text(json.dumps([asdict(m) for m in matched], indent=2))


if __name__ == "__main__":
    main()
