"""Components encapsulating data behaviour that also interact with the user."""

import logging
from pathlib import Path

import streamlit as st

from annotation.model import get_or_allocate_input_file

logger = logging.getLogger("annotation.components")


def get_or_allocate_annotation_path(
    annotation_dir: Path, split_to_user_file: Path, username: str
) -> Path | None:
    """Finds the path to the annotation file for the given user, or returns None.

    Data is divided in files, one per username. Annotation paths _must_ exist. This is
    different for the answer paths, which can be created on the fly.

    If the path doesn't exist or there are no free slots, renders an st.error and
    returns None.
    """
    path = get_or_allocate_input_file(annotation_dir, split_to_user_file, username)
    if path is None or not path.exists():
        st.error("Internal error: contact the administrator")
        return None

    return path


# TODO: Use "username" instead of "prolific_id" to be platform-agnostic
_PROLIFIC_ID_KEY = "prolific_id"


def get_username() -> str | None:
    user: str | None = None
    if prolific_id := st.query_params.get(_PROLIFIC_ID_KEY):
        st.session_state[_PROLIFIC_ID_KEY] = prolific_id

    if prolific_id := st.session_state.get(_PROLIFIC_ID_KEY):
        st.query_params[_PROLIFIC_ID_KEY] = prolific_id
        user = prolific_id

    if user is None:
        st.markdown("Please enter your Prolific ID")

        if prolific_id := st.text_input("Prolific ID"):
            st.session_state[_PROLIFIC_ID_KEY] = prolific_id
            st.query_params[_PROLIFIC_ID_KEY] = prolific_id
            user = prolific_id

    if user is None:
        return None

    st.markdown(f"**Your username is:** `{user}`")

    return user
