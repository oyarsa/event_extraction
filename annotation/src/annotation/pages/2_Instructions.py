import streamlit as st

from annotation.common import check_prolific_id
from annotation.instructions import render_instructions


def main() -> None:
    check_prolific_id()
    render_instructions()

    st.page_link("pages/3_Annotation.py", label="Data annotation")


if __name__ == "__main__":
    main()
