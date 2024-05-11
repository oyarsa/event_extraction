import streamlit as st

from annotation.common import ask_login, colour, get_prolific_id
from annotation.instructions import render_instructions


def main() -> None:
    st.set_page_config(page_title="Event Extraction Annotation")

    if not get_prolific_id():
        ask_login()

    st.title("Welcome")
    render_instructions()

    st.page_link(
        "pages/3_Annotation.py", label=colour("Go to annotation page", bg="grey")
    )


if __name__ == "__main__":
    main()
