"""Take multiple files, combine them into one, and split for annotation."""
import argparse
import random
from pathlib import Path
from typing import Any

from evaluation.annotation.split_data import (
    load_and_process_data,
    save_splits,
    split_data,
)


def main(
    file_models: list[str],
    output_dir: Path,
    num_subsets: int,
    common_pct: float,
    seed: int,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data: list[dict[str, Any]] = []
    for fm in file_models:
        file, model = fm.split(",")
        data = load_and_process_data(Path(file))
        all_data.extend(add_metadata(item, model) for item in data)

    splits = split_data(all_data, num_subsets, common_pct)
    save_splits("test_combined", output_dir, splits)


def add_metadata(item: dict[str, Any], model: str) -> dict[str, Any]:
    meta = "".join(item[f] for f in ["text", "model", "annotation"]) + model
    return item | {"meta": meta, "hash": hash(meta)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "file_models",
        type=Path,
        nargs="+",
        help="List of comma separated pairs. Each pair is file,model.",
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
    parser.add_argument("model", type=str, help="The model to use for the annotation.")
    args = parser.parse_args()
    main(
        args.file_models, args.output_dir, args.num_subsets, args.common_pct, args.seed
    )
