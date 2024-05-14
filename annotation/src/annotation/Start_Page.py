import logging
from pathlib import Path

import streamlit as st

from annotation.components import colour, get_annotation_path, get_username
from annotation.util import get_config, setup_logger

logger = logging.getLogger("annotation")


def main(annotation_dir: Path, split_to_user_file: Path) -> None:
    st.set_page_config(page_title="Event Extraction Annotation")
    st.title("Welcome to the Event Extraction Annotation tool")

    username = get_username()
    if not username:
        return

    instruction_col, annotation_col = st.columns([0.12, 0.88])
    instruction_col.page_link(
        "pages/1_Instructions.py",
        label=colour("Instructions", bg="red"),
    )
    annotation_col.page_link(
        "pages/2_Annotation.py", label=colour("Annotation page", bg="green")
    )

    if get_annotation_path(annotation_dir, split_to_user_file, username) is None:
        return


if __name__ == "__main__":
    config = get_config()
    setup_logger(logger, config.log_path)
    main(config.annotation_dir, config.split_to_user_file)
