"""Split the input data into multiple files for different users.

The input data file should only include data that needs to be annotated. For example,
it shouldn't include data like exact matches.

Use `scripts/filter_data.py` to generate the input data.
"""

import argparse
import hashlib
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def split_data(
    data: list[dict[str, Any]], num_subsets: int, overlap: int
) -> list[list[dict[str, Any]]]:
    data_ = data.copy()
    random.shuffle(data_)

    common_size = min(overlap, len(data_))
    common, remaining = data_[:common_size], data_[common_size:]
    size = len(remaining) // num_subsets

    return [common + remaining[i * size : (i + 1) * size] for i in range(num_subsets)]


def report_splits(
    overlap: int,
    num_subsets: int,
    splits: list[list[dict[str, Any]]],
) -> None:
    print(f"Split length: {len(splits[0])} x {num_subsets} ({overlap} common)")


def hash_data(data: list[dict[str, Any]]) -> str:
    """Generate a hash from the data."""
    data_txt = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data_txt).hexdigest()[:8]


def backup_directory(dir: Path) -> None:
    """Backup the directory to a new directory with the current timestamp."""
    if not dir.exists():
        return

    backup_dir = dir.with_name(f"{dir.name}_{datetime.now().isoformat()}")
    shutil.copytree(dir, backup_dir)
    print(f"Backed up {dir} to {backup_dir}")


def main(
    data_file: Path,
    num_splits: int,
    data_output_dir: Path,
    overlap: int,
    seed: int,
    max_size: int | None,
) -> None:
    backup_directory(data_output_dir)
    shutil.rmtree(data_output_dir, ignore_errors=True)

    random.seed(seed)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(data_file.read_text())[:max_size]

    data_splits = split_data(data, num_splits, overlap)
    split_ids = [hash_data(split) for split in data_splits]
    for split_id, split in zip(split_ids, data_splits):
        (data_output_dir / f"{split_id}.json").write_text(json.dumps(split, indent=2))

    split_to_user = {split_id: None for split_id in split_ids}
    (data_output_dir / "split_to_user.json").write_text(
        json.dumps(split_to_user, indent=2)
    )

    report_splits(overlap, num_splits, data_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_file", type=Path, help="Path to the data file")
    parser.add_argument(
        "num_splits",
        type=int,
        help="Number of splits to create",
    )
    parser.add_argument(
        "--data-output-dir",
        type=Path,
        help="Path to the output dir for the split data files",
        default="data/inputs",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Number of common instances between users",
        default=200,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the random number generator",
        default=0,
    )
    parser.add_argument(
        "--max-size",
        type=int,
        help="Maximum size of the split files",
        default=None,
    )
    args = parser.parse_args()
    main(
        args.data_file,
        args.num_splits,
        args.data_output_dir,
        args.overlap,
        args.seed,
        args.max_size,
    )
