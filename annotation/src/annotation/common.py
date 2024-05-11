import logging
import sys
from pathlib import Path
from typing import ClassVar

import streamlit as st
from typing_extensions import override

_PROLIFIC_STATE_KEY = "state_prolific"


def check_prolific_id() -> str | None:
    if prolific_id := st.session_state.get(PROLIFIC_STATE_KEY):
def set_prolific_id(prolific_id: str) -> None:
    st.session_state[_PROLIFIC_STATE_KEY] = prolific_id
        st.write(f"Your Prolific ID is: {prolific_id}")
        if st.button("Log out"):
            st.session_state.pop(PROLIFIC_STATE_KEY, None)
            st.rerun()
        return prolific_id
    return None


def ask_login() -> None:
    st.write("You're not logged in. Please log in with your Prolific ID.")
    st.page_link("pages/1_Welcome.py", label="Login page")


def heading(text: str, level: int) -> None:
    st.markdown(f"{'#' * level} {text}")


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
    colour: bool = False,
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.setLevel(logging.getLevelName(level.upper()))

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(output_dir / file_name, mode=mode)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    fmt_cls = ColourFormatter if colour else logging.Formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt_cls(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)
