import streamlit as st

from annotation.common import get_prolific_id, section_links, set_prolific_id


def main() -> None:
    if get_prolific_id():
        st.write(
            "You're logged in. Move on to the annotation page or review the"
            " instructions."
        )

        section_links()
        return

    if prolific_id := st.text_input(
        "Enter your Prolific ID",
        key="prolific_input_id",
        placeholder="Prolific ID",
    ):
        set_prolific_id(prolific_id)
        st.rerun()


if __name__ == "__main__":
    main()