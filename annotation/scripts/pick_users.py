"""Pick data files for users from Prolific IDs."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def backup_and_write(file: Path, txt: str) -> None:
    """Back up a text file with the timestamp, then overwrite the original."""
    if file.exists():
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_file = file.with_suffix(f".{ts}{file.suffix}")
        shutil.copy(file, backup_file)
        print(f"Backed up {file} to {backup_file}")
    file.write_text(txt)


def main(split_to_user_file: Path, prolific_file: Path) -> None:
    split_to_user = json.loads(split_to_user_file.read_text())
    prolific_ids = [
        id for id_ in prolific_file.read_text().splitlines() if (id := id_.strip())
    ]

    if len(prolific_ids) != len(split_to_user):
        raise ValueError(
            f"Number of Prolific IDs ({len(prolific_ids)}) does not match number of"
            f" splits ({len(split_to_user)})"
        )

    for split_id, prolific_id in zip(split_to_user, prolific_ids):
        split_to_user[split_id] = prolific_id

    backup_and_write(split_to_user_file, json.dumps(split_to_user, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "split_to_user_file",
        type=Path,
        help="Path to the split to user mapping file",
    )
    parser.add_argument(
        "prolific_file",
        type=Path,
        help="Path to the file containing Prolific IDs",
    )
    args = parser.parse_args()
    main(args.split_to_user_file, args.prolific_file)
