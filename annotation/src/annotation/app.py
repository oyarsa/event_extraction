"""Streamlit app to annotate evaluation data."""

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar

import streamlit as st
from typing_extensions import override

logger = logging.getLogger(__name__)

DEBUG = True


class ColourFormatter(logging.Formatter):
    """Logging colored formatter.

    Adapted from https://stackoverflow.com/a/56944256/3638629
    """

    formats: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\x1b[38;21m",  # grey
        logging.INFO: "\x1b[37m",  # white
        logging.WARNING: "\x1b[38;5;226m",  # yellow
        logging.ERROR: "\x1b[38;5;196m",  # red
        logging.CRITICAL: "\x1b[31;1m",  # bold red
    }
    reset: ClassVar[str] = "\x1b[0m"

    @override
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if colour := self.formats.get(record.levelno):
            return colour + msg + self.reset
        else:
            return msg


def setup_logger(
    logger: logging.Logger,
    output_dir: Path,
    file_name: str = "train.log",
    mode: str = "a",
    level: str = "info",
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.setLevel(logging.getLevelName(level.upper()))

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(output_dir / file_name, mode=mode)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColourFormatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)


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
    answer: bool | None


@dataclass
class UserProgress:
    prolific_id: str
    items: list[UserProgressItem]

    @classmethod
    def from_data(
        cls, prolific_id: str, data: list[AnnotationInstance]
    ) -> "UserProgress":
        return cls(
            prolific_id=prolific_id,
            items=[UserProgressItem(id=d.id, data=d.data, answer=None) for d in data],
        )

    def set_answer(self, idx: int, answer: bool) -> None:
        self.items[idx].answer = answer


def heading(text: str, level: int) -> None:
    st.markdown(f"{'#' * level} {text}")


def render_clauses(header: str, instance: ParsedInstance) -> None:
    heading(header, 3)
    st.markdown(f"**Causes**: {instance.cause}")
    st.markdown(f"**Effects**: {instance.effect}")


def save_progress(
    prolific_id: str,
    answer_dir: Path,
    idx: int,
    answer: bool,
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
        return UserProgress.from_data(prolific_id, input_data)

    data = json.loads(user_path.read_text())
    return UserProgress(
        prolific_id=data["prolific_id"],
        items=[
            UserProgressItem(id=item["id"], data=item["data"], answer=item["answer"])
            for item in data["items"]
        ],
    )


def set_answer(instance_id: str) -> None:
    st.session_state[instance_id] = (
        0 if st.session_state[radio_id(instance_id)] == "Valid" else 1
    )


def render_instance(
    instance: AnnotationInstance,
    prolific_id: str,
    answer_dir: Path,
    annotation_data: list[AnnotationInstance],
) -> bool:
    """Renders the instance and returns True if the user has selected a valid answer."""
    heading("Text", 3)
    st.write(instance.text)

    render_clauses("Reference answer", instance.annotation)
    render_clauses("Model answer", instance.model)

    previous_answer = load_answer(instance.id, prolific_id, answer_dir, annotation_data)
    if instance.id not in st.session_state:
        st.session_state[instance.id] = previous_answer

    label = "Is the model output valid relative to the reference?"
    heading(label, 3)
    valid = st.radio(
        label,
        ["Valid", "Invalid"],
        label_visibility="collapsed",
        index=None if previous_answer is None else int(previous_answer),
        horizontal=True,
        key=radio_id(instance.id),
        on_change=set_answer,
        kwargs={"instance_id": instance.id},
    )

    if DEBUG:
        st.write("Valid? ", valid)

    if valid is None:
        st.write("Please select an answer.")
        return False
    return True


def radio_id(instance_id: str) -> str:
    return f"rd_{instance_id}"


def load_answer(
    instance_id: str,
    prolific_id: str,
    answer_dir: Path,
    annotation_data: list[AnnotationInstance],
) -> bool | None:
    user_data = load_user_progress(prolific_id, answer_dir, annotation_data)
    return next(
        (item.answer for item in user_data.items if item.id == instance_id),
        None,
    )


def hash_instance(instance: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(instance).encode()).hexdigest()


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


def setup_prolific(prolific_id_param: str) -> str | None:
    """Sets up the Prolific ID. If it's not set, the page will be disabled."""
    if "prolific_id" in st.query_params:
        st.session_state["prolific_id"] = st.query_params[prolific_id_param]

    if prolific_id := st.text_input(
        "Enter your Prolific ID", key="prolific_id", placeholder="Prolific ID"
    ):
        st.write("Your Prolific ID is:", prolific_id)
        return prolific_id

    st.write("Please enter your Prolific ID to start.")
    return None


def find_last_entry_idx(
    prolific_id: str, answer_dir: Path, annotation_data: list[AnnotationInstance]
) -> int | None:
    user_path = get_user_path(answer_dir, prolific_id)
    if not user_path.exists():
        return None

    user_progress = load_user_progress(prolific_id, answer_dir, annotation_data)
    # First non-None (i.e. first unanswered) answer
    return next(
        (i for i, item in enumerate(user_progress.items) if item.answer is None), None
    )


def progress(
    prolific_id: str,
    answer_dir: Path,
    annotation_data: list[AnnotationInstance],
    instance_id: str,
    page_idx: int,
) -> None:
    col1, col2 = st.columns(2)

    if col1.button("Previous"):
        goto_page(page_idx - 1)
    if col2.button("Save & Next"):
        checkbox_answer = st.session_state[instance_id]
        save_progress(
            prolific_id, answer_dir, page_idx, checkbox_answer, annotation_data
        )
        goto_page(page_idx + 1)


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


def get_page_idx(
    annotation_data: list[AnnotationInstance], answer_dir: Path, prolific_id: str
) -> int:
    if "page_idx" in st.session_state:
        return st.session_state["page_idx"]

    # Find the first unanswered question so the user can continue from they left off
    first_unanswered_idx = find_last_entry_idx(prolific_id, answer_dir, annotation_data)

    # User starting now
    if first_unanswered_idx is None:
        page_idx = 0
    else:
        page_idx = first_unanswered_idx

    st.session_state["page_idx"] = page_idx
    return page_idx


def render_page(
    annotation_data: list[AnnotationInstance], answer_dir: Path, prolific_id_param: str
) -> None:
    prolific_id = setup_prolific(prolific_id_param)
    if prolific_id is None:
        return

    page_idx = get_page_idx(annotation_data, answer_dir, prolific_id)

    if page_idx >= len(annotation_data):
        heading("You have answered all questions.", 2)
        return

    heading("Annotate the data", 1)
    heading(f"#{page_idx + 1}", 2)

    instance = annotation_data[page_idx]
    if render_instance(instance, prolific_id, answer_dir, annotation_data):
        progress(prolific_id, answer_dir, annotation_data, instance.id, page_idx)


def main(
    log_path: Path, annotation_data_path: Path, answer_dir: Path, prolific_id_param: str
) -> None:
    setup_logger(logger, log_path)
    annotation_data = load_data(annotation_data_path)
    answer_dir.mkdir(exist_ok=True, parents=True)

    with st.sidebar:
        st.title("Instructions")
        st.write("TODO")
        st.write(f"Number of instances: {len(annotation_data)}")

    render_page(annotation_data, answer_dir, prolific_id_param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-file", type=Path, default="data/test.json")
    parser.add_argument("--answer-dir", type=Path, default="data/answers")
    parser.add_argument("--log-path", type=Path, default="logs")
    parser.add_argument("--prolific-id-param", default="prolific_id")
    args = parser.parse_args()

    main(args.log_path, args.data_file, args.answer_dir, args.prolific_id_param)
