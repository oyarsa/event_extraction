import streamlit as st

from annotation.components import colour
from annotation.util import get_config


def annotation_link():
    st.page_link(
        "pages/2_Annotation.py", label=colour("Go to annotation page", bg="green")
    )


def main() -> None:
    title_col, link_col = st.columns([0.5, 0.5])
    title_col.title("Instructions")
    with link_col:
        annotation_link()

    instructions = get_config().instructions_file.read_text()
    st.markdown(instructions)

    annotation_link()


if __name__ == "__main__":
    main()
