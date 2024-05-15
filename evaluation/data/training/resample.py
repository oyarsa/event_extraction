"""Resample tagged data file with N examples per tag."""

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def main(n: int, seed: int) -> None:
    random.seed(seed)

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

    substr_valid_s = random.sample(hand_valid, n)
    substr_invalid_s = random.sample(substr_invalid, n)
    em_s = random.sample(em, n)
    nonsubstr_s = random.sample(nonsubstr, n)

    tagged = [
        [{"tag": tag, **d} for d in data]
        for data, tag in zip(
            [substr_valid_s, substr_invalid_s, em_s, nonsubstr_s],
            ["substr_valid", "substr_invalid", "em", "nonsubstr"],
        )
    ]
    result = [d for data in tagged for d in data]
    random.shuffle(result)

    assert len(result) == 4 * 1700
    resampled_path.write_text(json.dumps(result, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", type=int, default=1700, help="Number of examples per tag."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()
    main(args.n, args.seed)
