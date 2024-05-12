import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NewType


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
    prolific_id: str
    items: list[UserProgressItem]

    @classmethod
    def from_unannotated_data(
        cls, prolific_id: str, data: list[AnnotationInstance]
    ) -> "UserProgress":
        """Initialise from the original, unannotated data."""
        return cls(
            prolific_id=prolific_id,
            items=[UserProgressItem(id=d.id, data=d.data, answer=None) for d in data],
        )

    def set_answer(self, idx: ItemIndex, answer: Answer) -> None:
        """Sets the answer for the item at the given index.

        The index should come from the items list itself, so it should be safe.
        """
        self.items[idx].answer = answer


def save_progress(
    prolific_id: str,
    answer_dir: Path,
    idx: ItemIndex,
    answer: Answer,
    input_data: list[AnnotationInstance],
) -> None:
    user_data = load_user_progress(prolific_id, answer_dir, input_data)
    user_data.set_answer(idx, answer)

    get_user_path(answer_dir, prolific_id).write_text(
        json.dumps(asdict(user_data), indent=2)
    )


def get_user_path(answer_dir: Path, prolific_id: str) -> Path:
    return answer_dir / f"{prolific_id}.json"


def load_user_progress(
    prolific_id: str, answer_dir: Path, input_data: list[AnnotationInstance]
) -> UserProgress:
    """Loads the user's progress from the answer file."""
    user_path = get_user_path(answer_dir, prolific_id)
    if not user_path.exists():
        return UserProgress.from_unannotated_data(prolific_id, input_data)

    data = json.loads(user_path.read_text())
    return UserProgress(
        prolific_id=data["prolific_id"],
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
    prolific_id: str,
    answer_dir: Path,
    annotation_data: list[AnnotationInstance],
) -> Answer | None:
    user_data = load_user_progress(prolific_id, answer_dir, annotation_data)
    return next(
        (item.answer for item in user_data.items if item.id == instance_id),
        None,
    )


def hash_instance(instance: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(instance).encode()).hexdigest()[:8]


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
        if (ann := parse_instance(d["annotation"]))
        and (model := parse_instance(d["model"]))
    ]


def find_last_entry_idx(
    prolific_id: str, answer_dir: Path, annotation_data: list[AnnotationInstance]
) -> ItemIndex:
    """If there is a last entry, return its index, otherwise return 0.

    The 0 means the user is starting now.
    """
    user_path = get_user_path(answer_dir, prolific_id)
    if not user_path.exists():
        return ItemIndex(0)

    user_progress = load_user_progress(prolific_id, answer_dir, annotation_data)
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


def reset_user_data(prolific_id: str, answer_dir: Path) -> None:
    user_path = get_user_path(answer_dir, prolific_id)
    if not user_path.exists():
        return

    user_data = read_user_data(user_path)
    for item in user_data:
        item["answer"] = None

    write_user_data(user_path, user_data)
