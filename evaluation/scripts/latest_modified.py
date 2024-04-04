#!/usr/bin/env python3
import argparse
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any


def get_latest_modified_file(directory: Path) -> tuple[Path, datetime] | None:
    """
    Get the latest modified file in the given directory.

    Args:
        directory (Path): The path to the directory.

    Returns:
        Path: The path to the latest modified file.
        datetime: The modification time of the latest modified file.
    """
    latest_file: Path | None = None
    latest_modified_time: float | None = None

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        modified_time = file_path.stat().st_mtime
        if latest_modified_time is None or modified_time > latest_modified_time:
            latest_file = file_path
            latest_modified_time = modified_time

    if latest_file is not None and latest_modified_time is not None:
        return latest_file, datetime.fromtimestamp(latest_modified_time)
    return None


def generate_table(headers: list[str], values: list[list[Any]]) -> str:
    # Calculate the maximum length for each column, considering both headers and the
    # data in values
    max_lengths = [
        max(len(str(row[i])) if i < len(row) else 0 for row in [headers, *values])
        for i in range(len(headers))
    ]

    # Create the format string for each column based on its maximum length
    fmt_parts = [f"{{{i}:<{len}}}" for i, len in enumerate(max_lengths)]
    fmt_string = " | ".join(fmt_parts)

    def format_row(row: list[Any]) -> str:
        return fmt_string.format(*(map(str, row)))

    # Generate header line
    header_line = format_row(headers)
    # Generate separator line
    separator_line = " | ".join("-" * length for length in max_lengths)
    # Generate table rows
    rows = [format_row(row) for row in values]

    # Combine all parts and return as a single string
    return "\n".join([header_line, separator_line, *rows])


def main() -> None:
    """
    Iterate over all subdirectories in the current working directory and print the
    latest modified file and its modification time for each directory.
    """
    parser = argparse.ArgumentParser(
        description="Print the latest modified file in each subdirectory."
    )
    parser.add_argument(
        "path", type=Path, default=".", nargs="?", help="The path to the directory."
    )
    args = parser.parse_args()

    result: list[list[str]] = []
    for subdir in args.path.glob("*"):
        if not subdir.is_dir():
            continue
        if modified := get_latest_modified_file(subdir):
            latest_file, modified_time = modified
            modified_time_fmt = modified_time.strftime("%Y-%m-%d %H:%M:%S")
            result.append([str(subdir), latest_file.name, modified_time_fmt])

    result.sort(key=lambda x: x[2])

    print(generate_table(["Directory", "File", "Modified Time"], result))


if __name__ == "__main__":
    main()
