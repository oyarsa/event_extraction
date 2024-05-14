from pathlib import Path
from typing import Literal, TypeAlias

import streamlit as st

from annotation.model import get_user_path


def escape(text: str) -> str:
    """Escape characters that have special meaning in LaTeX."""
    mapping = {
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "{": r"\{",
        "}": r"\}",
    }
    for char, escaped in mapping.items():
        text = text.replace(char, escaped)
    return text


Colour: TypeAlias = Literal[
    "blue", "green", "orange", "red", "violet", "grey", "rainbow"
]


def colour(text: str, fg: Colour | None = None, bg: Colour | None = None) -> str:
    """Supported colours: blue, green, orange, red, violet, grey, rainbow.

    Only one of fg or bg can be specified at the same time.
    """
    if fg and bg:
        raise ValueError("Cannot specify both foreground and background colours.")

    if fg is not None:
        return f":{fg}[{text}]"
    if bg is not None:
        return f":{bg}-background[{text}]"

    return text


def get_annotation_path(
    annotation_dir: Path, split_to_user_file: Path, username: str
) -> Path | None:
    """Data is divided in files, one per username."""
    annotation_path = get_user_path(annotation_dir, split_to_user_file, username)
    if annotation_path.exists():
        return annotation_path

    st.error(
        "Could not find a data file for you. Ensure you're using the correct"
        " username."
    )
    return None


_PROLIFIC_ID_KEY = "prolific_id"


def get_prolific_id() -> str | None:
    if prolific_id := st.query_params.get(_PROLIFIC_ID_KEY):
        st.session_state[_PROLIFIC_ID_KEY] = prolific_id

    if prolific_id := st.session_state.get(_PROLIFIC_ID_KEY):
        st.query_params[_PROLIFIC_ID_KEY] = prolific_id
        return prolific_id

    st.markdown("Please enter your Prolific ID")

    if prolific_id := st.text_input("Prolific ID"):
        st.session_state[_PROLIFIC_ID_KEY] = prolific_id
        st.query_params[_PROLIFIC_ID_KEY] = prolific_id
        return prolific_id

    return None


def standardise_page_name(name: str) -> str:
    return name.lower().replace("_", " ")


def logout_button() -> None:
    if st.button("Logout"):
        st.session_state.pop(_PROLIFIC_ID_KEY, None)
        st.query_params.clear()
        st.rerun()


def get_username() -> str | None:
    username: str | None = None
    if prolific_id := st.query_params.get(_PROLIFIC_ID_KEY):
        st.session_state[_PROLIFIC_ID_KEY] = prolific_id

    if prolific_id := st.session_state.get(_PROLIFIC_ID_KEY):
        st.query_params[_PROLIFIC_ID_KEY] = prolific_id
        username = prolific_id

    if username is None:
        st.markdown("Please enter your Prolific ID")

        if prolific_id := st.text_input("Prolific ID"):
            st.session_state[_PROLIFIC_ID_KEY] = prolific_id
            st.query_params[_PROLIFIC_ID_KEY] = prolific_id
            username = prolific_id

    if username is None:
        return None

    st.markdown(f"**Your username is:** `{username}`")

    return username
