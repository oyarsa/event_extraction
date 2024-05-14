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
    matches = re.findall(r"\[Cause\](.*?)\[Relation\](.*?)\[Effect\](.*?)$", answer)
    if not matches:
        return None
    causes, _, effects = matches[0]
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
    # Keeping the original data just in case
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
    answer_dir: Path,
    split_to_user_file: Path,
    idx: ItemIndex,
    answer: Answer,
    input_data: list[AnnotationInstance],
) -> None:
    user_data = load_user_progress(username, answer_dir, split_to_user_file, input_data)
    user_data.set_answer(idx, answer)

    get_user_path(answer_dir, split_to_user_file, username).write_text(
        json.dumps(asdict(user_data), indent=2)
    )


def get_user_path(dir: Path, split_to_user_file: Path, username: str) -> Path:
    """Get the path to a user's file. Applicable to both annotation and answers file.

    There is a mapping (see Config.split_to_user_file) that maps the data split name to
    the Prolific ID. This function uses that mapping to get the correct file.

    This applies to both the annotation and the answers file. Ensure that you're using
    the correct `dir` to get what you need.

    If the user is not found in the mapping, it will assign the user to the first free
    slot and return that file path, updating the mapping file.

    If there aren't any free slots, it will raise ValueError.
    TODO: Return None instead of raising an error.
    """
    split_to_user = json.loads(split_to_user_file.read_text())
    user_to_split = {v: k for k, v in split_to_user.items()}

    if filename := user_to_split.get(username):
        return dir / f"{filename}.json"

    free = next(
        (k for k, v in split_to_user.items() if v is None),
        None,
    )
    if free is None:
        logger.error(
            "No free slots available to assign data file. Username: %s", username
        )
        raise ValueError("No data file available")

    split_to_user[free] = username
    backup_and_write(split_to_user_file, json.dumps(split_to_user, indent=2))
    logger.info("Assigned file %s to %s", free, username)

    return dir / f"{free}.json"


def load_user_progress(
    username: str,
    answer_dir: Path,
    split_to_user_file: Path,
    input_data: list[AnnotationInstance],
) -> UserProgress:
    """Loads the user's progress from the answer file."""
    user_path = get_user_path(answer_dir, split_to_user_file, username)
    if not user_path.exists():
        return UserProgress.from_unannotated_data(username, input_data)

    data = json.loads(user_path.read_text())
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
    answer_dir: Path,
    split_to_user_file: Path,
    annotation_data: list[AnnotationInstance],
) -> Answer | None:
    """Loads user progress and retrieves the answer from the given instance ID.

    Returns None if a matching instance ID is not found.
    """
    user_data = load_user_progress(
        username, answer_dir, split_to_user_file, annotation_data
    )
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
    answer_dir: Path,
    split_to_user_file: Path,
    annotation_data: list[AnnotationInstance],
) -> ItemIndex:
    """If there is a last entry, return its index, otherwise return 0.

    The 0 means the user is starting now.
    """
    user_path = get_user_path(answer_dir, split_to_user_file, username)
    if not user_path.exists():
        return ItemIndex(0)

    user_progress = load_user_progress(
        username, answer_dir, split_to_user_file, annotation_data
    )
    # First non-None (i.e. first unanswered) answer
    idx = next(
        (i for i, item in enumerate(user_progress.items) if item.answer is None), None
    )
    # If there are no answered questions, idx will be None, so return 0
    return ItemIndex(idx if idx is not None else 0)


def read_user_data(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text())


def write_user_data(path: Path, data: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2))
