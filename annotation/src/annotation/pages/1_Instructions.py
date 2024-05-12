import streamlit as st

from annotation.common import colour, get_prolific_id
from annotation.instructions import render_instructions


def main() -> None:
    get_prolific_id()
    render_instructions()

    st.page_link(
        "pages/2_Annotation.py", label=colour("Go to annotation page", bg="green")
    )


if __name__ == "__main__":
    main()
