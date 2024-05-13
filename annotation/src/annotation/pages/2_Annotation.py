"""Streamlit app to annotate evaluation data."""

import logging
from pathlib import Path

import streamlit as st

from annotation.components import get_annotation_path, get_prolific_id
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
    st.markdown(f"**Reference**: {reference}")
    st.markdown(f"**Model**: {model}")


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


def set_page(page_idx: int) -> None:
    st.session_state["page_idx"] = page_idx


def goto_page(page_idx: int) -> None:
    set_page(page_idx)
    st.rerun()


def render_page(annotation_dir: Path, answer_dir: Path) -> None:
    prolific_id = get_prolific_id("annotation")
    if not prolific_id:
        return

    annotation_path = get_annotation_path(annotation_dir, prolific_id)
    if annotation_path is None:
        return

    annotation_data = load_data(annotation_path)

    # Find the first unanswered question so the user can continue from they left off.
    # If there are no unanswered questions, start from the beginning.
    # If the key exists, the user is currently annotating, or is coming back from the
    # same session.
    if "page_idx" in st.session_state:
        page_idx: ItemIndex = st.session_state["page_idx"]
    # Otherwise, automatically go to the latest entry.
    else:
        page_idx = find_last_entry_idx(prolific_id, answer_dir, annotation_data)

    if page_idx >= len(annotation_data):
        st.subheader("You have answered all questions.")
        if st.button("Go to start"):
            goto_page(0)
        return

    st.title(f"Annotate ({page_idx + 1} of {len(annotation_data)})")

    instance = annotation_data[page_idx]
    answer = answer_instance(instance, prolific_id, answer_dir, annotation_data)

    prev_col, next_col, latest_col = st.columns([2, 2, 5])
    if page_idx > 0 and prev_col.button("Previous"):
        goto_page(page_idx - 1)

    if answer is not None and next_col.button("Save & Next"):
        save_progress(prolific_id, answer_dir, page_idx, answer, annotation_data)
        goto_page(page_idx + 1)

    # Find the first unanswered question so the user can continue from they left off.
    # If there are no unanswered questions, start from the beginning.
    if latest_col.button("Go to latest"):
        page_idx = find_last_entry_idx(prolific_id, answer_dir, annotation_data)
        goto_page(page_idx)


def main() -> None:
    config = get_config()
    config.answer_dir.mkdir(exist_ok=True, parents=True)

    setup_logger(logger, config.log_path)
    render_page(config.annotation_dir, config.answer_dir)


if __name__ == "__main__":
    main()
