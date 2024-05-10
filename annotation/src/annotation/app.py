"""Streamlit app to annotate evaluation data."""

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import streamlit as st
from typing_extensions import override

logger = logging.getLogger(__name__)


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
    original: dict[str, str]


def heading(text: str, level: int) -> None:
    st.markdown(f"{'#' * level} {text}")


def render_clauses(header: str, instance: ParsedInstance) -> None:
    heading(header, 3)
    st.markdown(f"**Causes**: {instance.cause}")
    st.markdown(f"**Effects**: {instance.effect}")


def save(answer_path: Path, item: dict[str, Any]) -> None:
    with answer_path.open("a") as f:
        print(json.dumps(item), file=f, flush=True)


def save_answer(instance: AnnotationInstance, key: str, answer_path: Path) -> None:
    valid = st.session_state[key]
    prolific_id = st.session_state["prolific_id"]

    logger.info(f"{instance.id} - {prolific_id} - {valid}")

    item = {
        "id": instance.id,
        "prolific_id": prolific_id,
        "ts": datetime.now().isoformat(),
        "answer": valid,
        "original": instance.original,
    }
    save(answer_path, item)


def render_instance(
    instance: AnnotationInstance, num: int, answer_path: Path, enabled: bool
) -> None:
    heading("Text", 3)
    st.write(instance.text)

    render_clauses("Reference answer", instance.annotation)
    render_clauses("Model answer", instance.model)

    key = f"box_{num}"
    valid = st.checkbox(
        "Valid?",
        key=key,
        disabled=not enabled,
        on_change=save_answer,
        kwargs={
            "instance": instance,
            "key": key,
            "answer_path": answer_path,
        },
    )
    st.write("Valid? ", valid)


def hash_instance(instance: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(instance).encode()).hexdigest()


def load_data(path: Path) -> list[AnnotationInstance]:
    return [
        AnnotationInstance(
            id=hash_instance(d),
            text=d["text"],
            annotation=ann,
            model=model,
            original=d,
        )
        for d in json.loads(path.read_text())
        if (ann := parse_instance(d["annotation"]))
        and (model := parse_instance(d["model"]))
    ]


def setup_prolific() -> bool:
    """Sets up the Prolific ID. If it's not set, the page will be disabled."""
    if st.query_params.get("prolific_id") is not None:
        st.session_state["prolific_id"] = st.query_params["prolific_id"]

    if prolific_id := st.text_input(
        "Enter your Prolific ID", key="prolific_id", placeholder="Prolific ID"
    ):
        st.write("Your Prolific ID is:", prolific_id)
        return True

    st.write("Please enter your Prolific ID to start.")
    return False


def render_page(annotation_data, answer_path: Path) -> None:
    enabled = setup_prolific()

    heading("Annotate the data", 1)
    for i, instance in enumerate(annotation_data):
        heading(f"#{i + 1}", 2)
        render_instance(instance, i, answer_path, enabled)


def main(log_path: Path, data_path: Path, answer_path: Path) -> None:
    setup_logger(logger, log_path)
    annotation_data = load_data(data_path)

    render_page(annotation_data, answer_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-path", type=Path, default="data/test.json")
    parser.add_argument("--answer-path", type=Path, default="data/answers.json")
    parser.add_argument("--log-path", type=Path, default="logs")
    args = parser.parse_args()

    main(args.log_path, args.data_path, args.answer_path)
