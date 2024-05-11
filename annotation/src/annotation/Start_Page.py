import streamlit as st

from annotation.common import ask_login, get_prolific_id, section_links


def main() -> None:
    st.set_page_config(page_title="Event Extraction Annotation")

    if not get_prolific_id():
        ask_login()

    st.title("Welcome")
    section_links()


if __name__ == "__main__":
    main()
