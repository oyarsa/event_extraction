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


_USERNAME_KEY = "username"


def get_username() -> str | None:
    user: str | None = None
    if username := st.query_params.get(_USERNAME_KEY):
        st.session_state[_USERNAME_KEY] = username

    if username := st.session_state.get(_USERNAME_KEY):
        st.query_params[_USERNAME_KEY] = username
        user = username

    if user is None:
        st.markdown("Please enter your username:")

        if username := st.text_input("Username"):
            st.session_state[_USERNAME_KEY] = username
            st.query_params[_USERNAME_KEY] = username
            user = username

    if user is None:
        return None

    st.markdown(f"**Your username is:** `{user}`")

    return user
