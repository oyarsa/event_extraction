"""Streamlit app to annotate evaluation data."""

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NewType

import streamlit as st

from annotation.common import ask_login, get_config, get_prolific_id, setup_logger

logger = logging.getLogger("annotation.pages.3_Annotation")


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


def render_clauses(header: str, reference: str, model: str) -> None:
    st.subheader(header)
    st.markdown(f"**Reference**: {reference}")
    st.markdown(f"**Model**: {model}")


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


def answer_instance(
    instance: AnnotationInstance,
    prolific_id: str,
    answer_dir: Path,
    annotation_data: list[AnnotationInstance],
) -> Answer | None:
    """Renders the instance and returns True if the user has selected a valid answer."""
    st.subheader("Source Text")
    st.write(instance.text)

    render_clauses("Cause", instance.annotation.cause, instance.model.cause)
    render_clauses("Effect", instance.annotation.effect, instance.model.effect)

    previous_answer = load_answer(instance.id, prolific_id, answer_dir, annotation_data)
    if instance.id not in st.session_state:
        st.session_state[answer_state_id(instance.id)] = previous_answer

    radio_index = None
    if previous_answer is not None:
        radio_index = 0 if previous_answer is Answer.VALID else 1

    label = "Is the model output valid relative to the reference?"
    st.subheader(label)
    answer = st.radio(
        label,
        ["Valid", "Invalid"],
        label_visibility="collapsed",
        index=radio_index,
        horizontal=True,
        key=answer_radio_id(instance.id),
    )

    if answer is not None:
        return Answer[answer.upper()]

    st.write("Please select an answer.")
    return None


def answer_radio_id(instance_id: str) -> str:
    return f"answer_radio_{instance_id}"


def answer_state_id(instance_id: str) -> str:
    return f"answer_state_{instance_id}"


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


def goto_page(page_idx: int) -> None:
    st.session_state["page_idx"] = page_idx
    st.rerun()


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


def goto_latest(
    prolific_id: str, answer_dir: Path, annotation_data: list[AnnotationInstance]
) -> ItemIndex:
    page_idx = find_last_entry_idx(prolific_id, answer_dir, annotation_data)
    st.session_state["page_idx"] = page_idx
    return page_idx


def render_page(annotation_dir: Path, answer_dir: Path) -> None:
    prolific_id = get_prolific_id()
    if not prolific_id:
        ask_login()
        return

    # Data is divided in files, one per Prolific ID.
    annotation_path = annotation_dir / f"{prolific_id}.json"
    if not annotation_path.exists():
        ask_login()
        return

    annotation_data = load_data(annotation_path)

    # Find the first unanswered question so the user can continue from they left off.
    # If there are no unanswered questions, start from the beginning.
    if st.button("Go to latest"):
        page_idx = goto_latest(prolific_id, answer_dir, annotation_data)
    # If the key exists, the user is currently annotating, or is coming back from the
    # same session.
    elif "page_idx" in st.session_state:
        page_idx: ItemIndex = st.session_state["page_idx"]
    # Otherwise, automatically go to the latest entry.
    else:
        page_idx = goto_latest(prolific_id, answer_dir, annotation_data)

    if page_idx >= len(annotation_data):
        st.subheader("You have answered all questions.")
        return

    st.title("Annotate the data")
    st.header(f"Entry {page_idx + 1} of {len(annotation_data)}")

    instance = annotation_data[page_idx]
    answer = answer_instance(instance, prolific_id, answer_dir, annotation_data)

    prev_col, next_col = st.columns(2)
    if page_idx > 0 and prev_col.button("Previous"):
        goto_page(page_idx - 1)

    if answer is not None and next_col.button("Save & Next"):
        save_progress(prolific_id, answer_dir, page_idx, answer, annotation_data)
        goto_page(page_idx + 1)


def main() -> None:
    config = get_config()
    config.answer_dir.mkdir(exist_ok=True, parents=True)

    setup_logger(logger, config.log_path)
    render_page(config.annotation_dir, config.answer_dir)


if __name__ == "__main__":
    main()
