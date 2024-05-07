"""Split dataset into subsets for annotation, including a common subset between them.

Ensures that the resulting datasets have a subset of data that is the same between them
so we can calculate the inter-annotator agreement.
"""
import argparse
import json
import random
from pathlib import Path
from typing import Any


def split_data(
    data: list[dict[str, Any]], num_subsets: int, common_pct: float
) -> list[list[dict[str, Any]]]:
    data_ = data.copy()
    random.shuffle(data_)

    common_size = int(len(data_) * common_pct)
    common, remaining = data_[:common_size], data_[common_size:]
    size = len(remaining) // num_subsets

    return [common + remaining[i * size : (i + 1) * size] for i in range(num_subsets)]


def load_and_process_data(input: Path) -> list[dict[str, Any]]:
    return [
        {
            "text": item["input"],
            "annotation": item["gold"],
            "model": item["output"],
        }
        for item in json.loads(input.read_text())
    ]


def save_splits(
    name: str, output_dir: Path, splits: list[list[dict[str, Any]]]
) -> None:
    for i, subset in enumerate(splits):
        path = output_dir / f"{name}_subset_{i}.json"
        path.write_text(json.dumps(subset, indent=2))


def main(
    input: Path, output_dir: Path, num_subsets: int, common_pct: float, seed: int
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_process_data(input)
    splits = split_data(data, num_subsets, common_pct)
    print(f"Split length: {len(splits[0])} x {num_subsets} ({common_pct:.0%} common)")

    save_splits(input.name, output_dir, splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input JSON file")
    parser.add_argument(
        "output_dir", type=Path, help="Output directory to save the split datasets"
    )
    parser.add_argument(
        "--num-subsets",
        "-n",
        type=int,
        default=2,
        metavar="N",
        help="Number of subsets to create",
    )
    parser.add_argument(
        "--common-pct",
        "-p",
        type=float,
        default=0.2,
        metavar="PCT",
        help="Percentage of data to share between subsets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    main(args.input, args.output_dir, args.num_subsets, args.common_pct, args.seed)
