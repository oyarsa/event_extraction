import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import streamlit as st
import yaml
from streamlit.runtime.scriptrunner import RerunData, RerunException
from streamlit.source_util import get_pages
from typing_extensions import override


def colour(text: str, fg: str | None = None, bg: str | None = None) -> str:
    if fg is not None:
        return f":{fg}[{text}]"
    if bg is not None:
        return f":{bg}-background[{text}]"
    return text


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


_PROLIFIC_STATE_KEY = "state_prolific"


def set_prolific_state_id(prolific_id: str) -> None:
    st.session_state[_PROLIFIC_STATE_KEY] = prolific_id


def get_prolific_id() -> str | None:
    if prolific_id := st.session_state.get(_PROLIFIC_STATE_KEY):
        subsubheader(f"**Your Prolific ID is:** `{prolific_id}`")
        if st.button("Log out", type="primary"):
            # Reset Prolific ID and page state
            st.session_state.clear()
            switch_page("Start Page")

        return prolific_id

    return None


@dataclass
class Config:
    log_path: Path
    annotation_dir: Path
    answer_dir: Path


def get_config() -> Config:
    with open("config/params.yaml") as f:
        config = yaml.safe_load(f)
    return Config(
        log_path=Path(config["log_path"]),
        annotation_dir=Path(config["data"]["input"]),
        answer_dir=Path(config["data"]["answers"]),
    )


def ask_login() -> None:
    msg, link = st.columns([0.75, 0.25])
    with msg:
        st.error("You're not logged in. Log in with your Prolific ID.")
    link.page_link("Start_Page.py", label=colour("Log In", bg="blue"))


def subsubheader(text: str) -> None:
    heading(text, 4)


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

    fmt = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(output_dir / file_name, mode=mode)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(file_handler)

    fmt_cls = ColourFormatter if colour else logging.Formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt_cls(fmt=fmt, datefmt=datefmt))
    logger.addHandler(console_handler)
