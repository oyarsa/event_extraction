#!/usr/bin/env python3
"""Split the input data into multiple files for different users.

The input data file should only include data that needs to be annotated. For example,
it shouldn't include data like exact matches.

Use `scripts/filter_data.py` to generate the input data.
"""

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def split_data(
    data: list[dict[str, Any]], num_splits: int, overlap: int
) -> list[list[dict[str, Any]]]:
    overlap = min(overlap, len(data))
    common, remaining = data[:overlap], data[overlap:]
    if not remaining:
        print("No remaining data to split")
        return [common] * num_splits

    split_len = math.ceil(len(remaining) / num_splits)

    splits = [
        common + remaining[i : i + split_len]
        for i in range(0, len(remaining), split_len)
    ]
    assert len(splits) == num_splits
    return splits


def report_splits(
    overlap: int,
    splits: list[list[dict[str, Any]]],
) -> None:
    split_lens = ", ".join(str(len(split)) for split in splits)
    overlap = min(overlap, *(len(split) for split in splits))
    print(f"Split lengths: {split_lens} ({overlap} common)")


def hash_data(i: int, data: list[dict[str, Any]]) -> str:
    """Generate a hash from the data.

    We need to include the index `i` to ensure that the hash is unique for each split,
    even if they have same data. That happens when we're generating splits that
    have the same data, like when we're testing the annotation tool.
    """
    data_txt = str(i) + json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_txt.encode("utf-8")).hexdigest()[:8]


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
    output_dir: Path,
    overlap: int,
    max_size: int | None,
    backup: bool,
) -> None:
    if backup:
        backup_directory(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(data_file.read_text())[:max_size]

    data_splits = split_data(data, num_splits, overlap)
    split_ids = [hash_data(i, split) for i, split in enumerate(data_splits)]
    for split_id, split in zip(split_ids, data_splits):
        (output_dir / f"{split_id}.json").write_text(json.dumps(split, indent=2))

    split_to_user = {split_id: None for split_id in split_ids}
    (output_dir / "split_to_user.json").write_text(json.dumps(split_to_user, indent=2))

    report_splits(overlap, data_splits)


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
        "output_dir",
        type=Path,
        help="Path to the output dir for the split data files",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        help="Number of common instances between users",
        default=200,
    )
    parser.add_argument(
        "--max-size",
        type=int,
        help="Maximum number of examples before splitting",
        default=None,
    )
    parser.add_argument(
        "--backup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Backup the output directory before creating the splits",
    )
    args = parser.parse_args()
    main(
        args.data_file,
        args.num_splits,
        args.output_dir,
        args.overlap,
        args.max_size,
        args.backup,
    )
