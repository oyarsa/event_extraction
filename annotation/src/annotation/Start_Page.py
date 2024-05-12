import streamlit as st

from annotation.common import (
    colour,
    get_config,
    get_prolific_id,
    set_prolific_state_id,
)


def main() -> None:
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

    # Data is divided in files, one per Prolific ID.
    annotation_path = get_config().annotation_dir / f"{prolific_id}.json"
    if not annotation_path.exists():
        st.error("Invalid Prolific ID. Please try again.")
        return

    set_prolific_state_id(prolific_id)


if __name__ == "__main__":
    main()
