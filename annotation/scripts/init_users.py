"""Initialise user credentials and data files.

The input data file should only include data that needs to be annotated. Don't
include data like exact match.
"""

import argparse
import json
import random
import shutil
import string
from datetime import datetime
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


def backup_auth_file(auth_file: Path) -> None:
    """Backup the authentication file."""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_file = auth_file.with_name(f"auth.{ts}.yaml")
    shutil.copy(auth_file, backup_file)
    print(f"Backed up {auth_file} to {backup_file}")


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
    data_output_dir.mkdir(parents=True, exist_ok=True)

    backup_auth_file(auth_file)
    auth = yaml.safe_load(auth_file.read_text())
    data = yaml.safe_load(data_file.read_text())

    data_splits = split_data(data, num_users, common_pct)
    users = auth["credentials"]["usernames"]
    new_users = {"admin": users.pop("admin")}
    creds: list[str] = []

    for split in data_splits:
        username, password = generate_credentials(username_length)
        new_users[username] = {
            "name": username,
            "password": password,
            "logged_in": False,
        }
        (data_output_dir / f"{username}.json").write_text(json.dumps(split))
        creds.append(f"{username},{password}")

    auth["credentials"]["usernames"] = new_users

    auth_file.write_text(yaml.dump(auth))
    (auth_file.parent / "credentials.csv").write_text("\n".join(creds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("auth_file", type=Path, help="Path to the authentication file")
    parser.add_argument("data_file", type=Path, help="Path to the data file")
    parser.add_argument("num_users", type=int, help="Number of users to initialise")
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
