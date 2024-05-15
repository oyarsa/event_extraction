import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NewType

from annotation.util import backup_and_write

logger = logging.getLogger("annotation.model")


@dataclass
class ParsedInstance:
    cause: str
    effect: str


def parse_instance(answer: str) -> ParsedInstance | None:
    matches = re.findall(r"\[Cause\](.*?)\[Relation\].*?\[Effect\](.*?)$", answer)
    if not matches:
        return None

    causes, effects = matches[0]
    causes = " AND ".join(sorted(c.strip() for c in causes.split("|") if c.strip()))
    effects = " AND ".join(sorted(e.strip() for e in effects.split("|") if e.strip()))

    return ParsedInstance(causes, effects)


class Answer(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass
class AnnotationInstance:
    id: str
    text: str
    annotation: ParsedInstance
    model: ParsedInstance
    data: dict[str, Any]


@dataclass
class UserProgressItem:
    id: str
    data: dict[str, Any]
    answer: Answer | None


ItemIndex = NewType("ItemIndex", int)


@dataclass
class UserProgress:
    username: str
    items: list[UserProgressItem]

    @classmethod
    def from_unannotated_data(
        cls, username: str, data: list[AnnotationInstance]
    ) -> "UserProgress":
        """Initialise from the original, unannotated data."""
        return cls(
            username=username,
            items=[UserProgressItem(id=d.id, data=d.data, answer=None) for d in data],
        )

    def set_answer(self, idx: ItemIndex, answer: Answer) -> None:
        """Sets the answer for the item at the given index.

        The index should come from the items list itself, so it should be safe.
        """
        self.items[idx].answer = answer


def save_progress(
    username: str,
    answer_path: Path,
    idx: ItemIndex,
    answer: Answer,
    input_data: list[AnnotationInstance],
) -> None:
    user_data = load_user_progress(username, answer_path, input_data)
    user_data.set_answer(idx, answer)

    answer_path.write_text(json.dumps(asdict(user_data), indent=2))


def get_data_file(
    data_dir: Path, split_to_user_file: Path, username: str
) -> Path | None:
    """Get the allocated file for the user in `data_dir` if it assigned.

    This applies to both annotation and answer files since the mapping is the same.

    Returns:
        Path to the user's file if found in the `split_to_user_file` mapping, otherwise
        None.
    """
    split_to_user = json.loads(split_to_user_file.read_text())
    user_to_split = {v: k for k, v in split_to_user.items()}
    if filename := user_to_split.get(username):
        return data_dir / f"{filename}.json"
    return None


class DataFileError(Exception):
    pass


def allocate_input_file(
    input_dir: Path, split_to_user_file: Path, username: str
) -> Path:
    """Allocate a unallocated input file to the user.

    If it's a new allocation, the split_to_user_file is updated.

    Returns:
        Allocated file path.

    Raises:
        DataFileError: If there are no files left to be allocated.
    """
    split_to_user = json.loads(split_to_user_file.read_text())
    unallocated_file = next(
        (k for k, v in split_to_user.items() if v is None),
        None,
    )
    # No unallocated files left
    if unallocated_file is None:
        raise DataFileError(f"No unallocated files left for user {username}")

    # Allocate the file to the user
    split_to_user[unallocated_file] = username
    backup_and_write(split_to_user_file, json.dumps(split_to_user, indent=2))
    logger.info("[user %s] Assigned file %s", username, unallocated_file)

    return input_dir / f"{unallocated_file}.json"


def _get_or_allocate_input_file(
    input_dir: Path, split_to_user_file: Path, username: str
) -> Path:
    """Try to get or allocate the input file for the user.

    Returns:
        The user's input file.

    Raises:
        DataFileError: If there are no files left to be allocated.
        DataFileError: If the file is allocated but doesn't exist.
    """
    if file := get_data_file(input_dir, split_to_user_file, username):
        if not file.exists():
            raise DataFileError(f"Allocated file {file} doesn't exist")
        return file

    file = allocate_input_file(input_dir, split_to_user_file, username)
    if not file.exists():
        raise DataFileError(f"Allocated file {file} doesn't exist")

    return file


def get_or_allocate_input_file(
    input_dir: Path, split_to_user_file: Path, username: str
) -> Path | None:
    """Get the allocated input file for the user, or allocate a new one if needed.

    If the user needed a new file but there were none left, returns None and logs an
    error.

    Returns:
        The user's input file, or None if there are no files left to be allocated.
    """
    try:
        return _get_or_allocate_input_file(input_dir, split_to_user_file, username)
    except DataFileError:
        logger.exception("[user %s] Input file error", username)
        return None


def load_user_progress(
    username: str,
    answer_path: Path,
    input_data: list[AnnotationInstance],
) -> UserProgress:
    """Loads the user's progress from the answer file."""
    if not answer_path.exists():
        return UserProgress.from_unannotated_data(username, input_data)

    data = json.loads(answer_path.read_text())
    return UserProgress(
        username=data["username"],
        items=[
            UserProgressItem(
                id=item["id"],
                data=item["data"],
                answer=Answer[item["answer"].upper()]
                if item["answer"] is not None
                else None,
            )
            for item in data["items"]
        ],
    )


def load_answer(
    instance_id: str,
    username: str,
    answer_path: Path,
    annotation_data: list[AnnotationInstance],
) -> Answer | None:
    """Loads user progress and retrieves the answer from the given instance ID.

    Returns None if a matching instance ID is not found.
    """
    user_data = load_user_progress(username, answer_path, annotation_data)
    return next(
        (item.answer for item in user_data.items if item.id == instance_id),
        None,
    )


def hash_instance(instance: dict[str, Any], length: int = 8) -> str:
    """Hashes the string JSON representation of the dict.

    Length is because we don't need the entire hash to be unique, just enough to
    differentiate between instances. Shorter hashes are easier to read and compare.
    """
    return hashlib.sha256(json.dumps(instance).encode()).hexdigest()[:length]


def load_data(path: Path) -> list[AnnotationInstance]:
    return [
        AnnotationInstance(
            id=hash_instance(d),
            text=d["text"],
            annotation=ann,
            model=model,
            data=d,
        )
        for d in json.loads(path.read_text())
        if (ann := parse_instance(d["reference"]))
        and (model := parse_instance(d["model"]))
    ]


def find_last_entry_idx(
    username: str,
    answer_path: Path,
    annotation_data: list[AnnotationInstance],
) -> ItemIndex:
    """If there is a last entry, return its index, otherwise return 0.

    The 0 means the user is starting now.
    """
    if not answer_path.exists():
        return ItemIndex(0)

    user_progress = load_user_progress(username, answer_path, annotation_data)
    # First non-None (i.e. first unanswered) answer
    idx = next(
        (i for i, item in enumerate(user_progress.items) if item.answer is None), None
    )
    # If there are no answered questions, idx will be None, so return 0
    return ItemIndex(idx if idx is not None else 0)
