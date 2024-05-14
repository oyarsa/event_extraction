import streamlit as st

from annotation.components import colour, escape, get_username
from annotation.util import get_config


def annotation_link() -> None:
    st.page_link("pages/2_Annotation.py", label=colour("Annotation page", bg="green"))


def main() -> None:
    username = get_username()
    if not username:
        return

    title_col, link_col = st.columns([0.5, 0.5])
    title_col.title("Instructions")
    with link_col:
        annotation_link()

    instructions = get_config().instructions_file.read_text()
    st.markdown(escape(instructions))

    annotation_link()


if __name__ == "__main__":
    main()
