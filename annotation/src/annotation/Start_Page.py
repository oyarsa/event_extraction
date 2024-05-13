from pathlib import Path

import streamlit as st

from annotation.components import colour, get_annotation_path, get_username
from annotation.util import get_config


def main(annotation_dir: Path) -> None:
    st.set_page_config(page_title="Event Extraction Annotation")
    st.title("Welcome to the Event Extraction Annotation tool")

    username = get_username("start_page")
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

    if get_annotation_path(annotation_dir, username) is None:
        return


if __name__ == "__main__":
    config = get_config()
    main(config.annotation_dir)
