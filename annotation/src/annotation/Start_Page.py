from pathlib import Path

import streamlit as st

from annotation.components import get_annotation_path, get_prolific_id
from annotation.util import colour, get_config


def main(annotation_dir: Path) -> None:
    st.set_page_config(page_title="Event Extraction Annotation")
    st.title("Welcome")

    prolific_id = get_prolific_id("start_page")
    if not prolific_id:
        return

    instruction_col, annotation_col = st.columns([0.12, 0.88])
    instruction_col.page_link(
        "pages/1_Instructions.py",
        label=colour("Instructions", bg="red"),
    )
    annotation_col.page_link(
        "pages/2_Annotation.py", label=colour("Annotation page", bg="green")
    )

    if get_annotation_path(annotation_dir, prolific_id) is None:
        return


if __name__ == "__main__":
    config = get_config()
    main(config.annotation_dir)
