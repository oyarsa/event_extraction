from pathlib import Path
from typing import Literal, TypeAlias

import streamlit as st
import streamlit_authenticator as stauth
import yaml

from annotation.util import get_config


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


def get_username(page: str) -> str | None:
    authenticator = load_authenticator()
    name, status, username = authenticator.login()

    if status is None or name is None:
        st.warning("Please log in")
        return None
    elif status is False:
        st.error("Username/password is incorrect")
        return None

    text_col, logout_col = st.columns([0.75, 0.25])
    with text_col:
        st.write(f"**Your username is:** `{username}`")

    with logout_col:
        authenticator.logout(key=f"logout_{page}")

    return username


def load_authenticator() -> stauth.Authenticate:
    auth_config = yaml.safe_load(get_config().auth_file.read_text())
    return stauth.Authenticate(
        auth_config["credentials"],
        auth_config["cookie"]["name"],
        auth_config["cookie"]["key"],
        auth_config["cookie"]["expiry_days"],
    )
