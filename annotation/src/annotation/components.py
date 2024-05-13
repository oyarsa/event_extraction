from pathlib import Path
from typing import Literal, TypeAlias

import streamlit as st
from streamlit.runtime.scriptrunner import RerunData, RerunException
from streamlit.source_util import get_pages


def escape(text: str) -> str:
    """Escape characters that have special meaning in LaTeX."""
    mapping = {
        "$": r"\$",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
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


def get_annotation_path(annotation_dir: Path, username: str) -> Path | None:
    """Data is divided in files, one per username."""
    annotation_path = annotation_dir / f"{username}.json"
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

    st.write("Please enter your Prolific ID")
    if prolific_id := st.text_input("Prolific ID"):
        st.session_state[_PROLIFIC_ID_KEY] = prolific_id
        st.query_params[_PROLIFIC_ID_KEY] = prolific_id
        return prolific_id

    return None


def standardise_page_name(name: str) -> str:
    return name.lower().replace("_", " ")


def switch_page(page_name: str) -> None:
    """Switch page programmatically in a multipage app.

    Args:
        page_name: Target page name. `1_Annotation.py` -> "Annotation",
            `Start_Page.py` -> "Start Page"

    Copied from https://github.com/arnaudmiribel/streamlit-extras/blob/cbfb787bd94edaf0ad7a55a0ba3782e94ee7fbe2/src/streamlit_extras/__init__.py#L23
    """
    page_name_std = standardise_page_name(page_name)
    pages = get_pages("Start_Page.py")

    for page_hash, config in pages.items():
        if standardise_page_name(config["page_name"]) == page_name_std:
            raise RerunException(
                RerunData(
                    page_script_hash=page_hash,
                    page_name=page_name_std,
                )
            )

    raise ValueError(f"Page not found: {page_name}.")


def get_username(page: str) -> str | None:
    username: str | None = None
    if prolific_id := st.query_params.get(_PROLIFIC_ID_KEY):
        st.session_state[_PROLIFIC_ID_KEY] = prolific_id

    if prolific_id := st.session_state.get(_PROLIFIC_ID_KEY):
        st.query_params[_PROLIFIC_ID_KEY] = prolific_id
        username = prolific_id

    if username is None:
        st.write("Please enter your Prolific ID")
        if prolific_id := st.text_input("Prolific ID"):
            st.session_state[_PROLIFIC_ID_KEY] = prolific_id
            st.query_params[_PROLIFIC_ID_KEY] = prolific_id
            username = prolific_id

    if username is None:
        return None

    text_col, logout_col = st.columns([0.75, 0.25])
    with text_col:
        st.write(f"**Your username is:** `{username}`")

    with logout_col:
        if st.button("Logout", key=f"logout_{page}"):
            st.session_state.pop(_PROLIFIC_ID_KEY, None)
            st.query_params.pop(_PROLIFIC_ID_KEY, None)
            switch_page("start page")

    return username
