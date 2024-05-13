"""Initialise user credentials and data files.

The input data file should only include data that needs to be annotated. Don't
include data like exact match.
"""

import argparse
import random
import string
from pathlib import Path
from typing import Any

import yaml


def split_data(
    data: list[dict[str, Any]], num_subsets: int, common_pct: float
) -> list[list[dict[str, Any]]]:
    data_ = data.copy()
    random.shuffle(data_)

    common_size = int(len(data_) * common_pct)
    common, remaining = data_[:common_size], data_[common_size:]
    size = len(remaining) // num_subsets

    return [common + remaining[i * size : (i + 1) * size] for i in range(num_subsets)]


def report_splits(
    all_data: list[dict[str, Any]],
    common_pct: float,
    num_subsets: int,
    splits: list[list[dict[str, Any]]],
) -> None:
    print(
        f"Split length: {len(splits[0])} x {num_subsets} ({common_pct:.0%} common ="
        f" {int(len(all_data) * common_pct):.0f} items)"
    )


def generate_random_string(length: int) -> str:
    """Generate a random string of the given length.

    The strings contain ASCII letters and digits.
    """
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def generate_credentials(length: int) -> tuple[str, str]:
    """Generate a random username and password."""
    return generate_random_string(length), generate_random_string(length)


def main(
    auth_file: Path,
    data_file: Path,
    data_output_dir: Path,
    num_users: int,
    username_length: int,
    common_pct: float,
    seed: int,
) -> None:
    random.seed(seed)

    auth = yaml.safe_load(auth_file.read_text())
    data = yaml.safe_load(data_file.read_text())

    data_splits = split_data(data, num_users, common_pct)

    for split in data_splits:
        username, password = generate_credentials(username_length)
        auth["users"] = {
            "name": username,
            "password": password,
            "logged_in": False,
        }
        (data_output_dir / f"{username}.yaml").write_text(yaml.dump(split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("auth-file", type=Path, help="Path to the authentication file")
    parser.add_argument("data-file", type=Path, help="Path to the data file")
    parser.add_argument("num-users", type=int, help="Number of users to initialise")
    parser.add_argument(
        "--data-output-dir",
        type=Path,
        help="Path to the output dir for the split data files",
        default="data/inputs",
    )
    parser.add_argument(
        "--username-length",
        type=int,
        help="Length of the generated usernames",
        default=8,
    )
    parser.add_argument(
        "--common-pct",
        type=float,
        help="Percentage of common data between users",
        default=0.2,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for the random number generator",
        default=8,
    )
    args = parser.parse_args()
    main(
        args.auth_file,
        args.data_file,
        args.data_output_dir,
        args.num_users,
        args.username_length,
        args.common_pct,
        args.seed,
    )
