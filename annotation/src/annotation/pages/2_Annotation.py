"""Streamlit app to annotate evaluation data."""

import logging
from pathlib import Path

import streamlit as st

from annotation.components import escape, get_annotation_path, get_username
from annotation.model import (
    AnnotationInstance,
    Answer,
    ItemIndex,
    find_last_entry_idx,
    load_answer,
    load_data,
    save_progress,
)
from annotation.util import get_config, setup_logger

logger = logging.getLogger("annotation.pages.3_Annotation")


def render_clauses(header: str, reference: str, model: str) -> None:
    st.subheader(header)
    st.markdown(f"**Reference**: {escape(reference)}")
    st.markdown(f"**Model**: {escape(model)}")


def answer_instance(
    instance: AnnotationInstance,
    username: str,
    answer_dir: Path,
    split_to_user_file: Path,
    annotation_data: list[AnnotationInstance],
) -> Answer | None:
    """Renders the instance and returns True if the user has selected a valid answer."""
    st.subheader("Source Text")
    st.write(escape(instance.data["text"]))

    render_clauses("Cause", instance.annotation.cause, instance.model.cause)
    render_clauses("Effect", instance.annotation.effect, instance.model.effect)

    previous_answer = load_answer(
        instance.id, username, answer_dir, split_to_user_file, annotation_data
    )
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


_PAGE_IDX_KEY = "page_idx"


def goto_page(page_idx: int) -> None:
    st.session_state[_PAGE_IDX_KEY] = page_idx
    st.rerun()


def render_page(
    annotation_dir: Path, answer_dir: Path, split_to_user_file: Path
) -> None:
    username = get_username()
    if not username:
        return

    annotation_path = get_annotation_path(annotation_dir, username)
    if annotation_path is None:
        return

    annotation_data = load_data(annotation_path)
    page_idx: ItemIndex = st.session_state.get(_PAGE_IDX_KEY, 0)

    if page_idx >= len(annotation_data):
        st.subheader("You have answered all questions.")
        if st.button("Go to start"):
            goto_page(0)
        return

    st.title(f"Annotate ({page_idx + 1} of {len(annotation_data)})")

    instance = annotation_data[page_idx]
    answer = answer_instance(
        instance, username, answer_dir, split_to_user_file, annotation_data
    )

    prev_col, next_col, first_col, latest_col = st.columns([2, 2, 2, 5])
    if page_idx > 0 and prev_col.button("Previous"):
        goto_page(page_idx - 1)

    if answer is not None and next_col.button("Save & Next"):
        save_progress(
            username, answer_dir, split_to_user_file, page_idx, answer, annotation_data
        )
        goto_page(page_idx + 1)

    if page_idx > 0 and first_col.button("Go to first"):
        goto_page(0)

    # Find the first unanswered question so the user can continue from they left off.
    # If there are no unanswered questions, start from the beginning.
    if latest_col.button("Go to next unanswered"):
        page_idx = find_last_entry_idx(
            username, answer_dir, split_to_user_file, annotation_data
        )
        goto_page(page_idx)


def main() -> None:
    config = get_config()
    config.answer_dir.mkdir(exist_ok=True, parents=True)

    setup_logger(logger, config.log_path)
    render_page(config.annotation_dir, config.answer_dir, config.split_to_user)


if __name__ == "__main__":
    main()
