import json
import random
from pathlib import Path
from typing import Any


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def tag(data: list[dict[str, Any]], tag: str) -> list[dict[str, Any]]:
    return [{"tag": tag, **d} for d in data]


def main() -> None:
    random.seed(0)

    hand_path = Path("./hand_labelled.json")
    predicted_path = Path("./predicted.json")
    rule_path = Path("./rule_labelled.json")
    resampled_path = Path("./resampled.json")

    hand_data = read_json(hand_path)
    predicted_data = read_json(predicted_path)
    rule_data = read_json(rule_path)

    hand_valid = [d for d in hand_data if d["valid"]]
    hand_invalid = [d for d in hand_data if not d["valid"]]
    em = [d for d in rule_data if d["valid"]]
    nonsubstr = [d for d in rule_data if not d["valid"]]
    predicted_invalid = [d for d in predicted_data if not d["valid"]]

    substr_invalid = hand_invalid + predicted_invalid

    n = 1700
    substr_valid_s = random.sample(hand_valid, n)
    substr_invalid_s = random.sample(substr_invalid, n)
    em_s = random.sample(em, n)
    nonsubstr_s = random.sample(nonsubstr, n)

    tagged = [
        tag(d, t)
        for d, t in zip(
            [substr_valid_s, substr_invalid_s, em_s, nonsubstr_s],
            ["substr_valid", "substr_invalid", "em", "nonsubstr"],
        )
    ]
    result = sum(tagged, [])
    random.shuffle(result)

    assert len(result) == 4 * 1700
    resampled_path.write_text(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
