"""Take multiple files, combine them into one, and split for annotation."""

import argparse
import random
from pathlib import Path
from typing import Any

from evaluation.annotation.split_training import (
    annotate,
    calc_annotation_ratios,
    load_and_reshape_data,
    remove_auto_annotated,
    report_splits,
    save_annotation_results,
    save_splits,
    show_annotation_results,
    split_data,
)


def main(
    files: list[Path],
    output_dir: Path,
    num_subsets: int,
    common_pct: float,
    seed: int,
    min_subseq_length: int,
) -> None:
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data: list[dict[str, Any]] = []
    for file in files:
        data = load_and_reshape_data(Path(file))
        all_data.extend(add_metadata(item, file.stem) for item in data)

    annotated_data = annotate(all_data, min_subseq_length)

    annotation_ratios = calc_annotation_ratios(annotated_data)
    print(show_annotation_results(annotation_ratios), end="\n\n")
    save_annotation_results(output_dir / "annotation_ratios.json", annotation_ratios)

    needs_annotation_data = remove_auto_annotated(annotated_data)
    splits = split_data(needs_annotation_data, num_subsets, common_pct)

    save_splits("test_combined", output_dir, splits)
    report_splits(needs_annotation_data, common_pct, num_subsets, splits)


def add_metadata(item: dict[str, Any], model: str) -> dict[str, Any]:
    meta = "".join(item[f] for f in ["text", "model", "annotation"]) + model
    return item | {"meta": meta, "hash": hash(meta)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        help="List of comma separated pairs. Each pair is file,model.",
    )
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
    parser.add_argument(
        "--min-subseq-length",
        type=int,
        default=1,
        help="Minimum subsequence length for intersection check",
    )
    args = parser.parse_args()
    main(
        args.files,
        args.output_dir,
        args.num_subsets,
        args.common_pct,
        args.seed,
        args.min_subseq_length,
    )
