import streamlit as st

from annotation.common import PROLIFIC_INPUT_KEY, PROLIFIC_STATE_KEY, check_prolific_id


def main() -> None:
    if check_prolific_id():
        st.write(
            "You're logged in. Move on to the annotation page or review the"
            " instructions."
        )
        st.page_link("pages/2_Instructions.py", label="Instructions")
        st.page_link("pages/3_Annotation.py", label="Annotation page")
        return

    if prolific_id := st.text_input(
        "Enter your Prolific ID",
        key=PROLIFIC_INPUT_KEY,
        placeholder="Prolific ID",
    ):
        st.session_state[PROLIFIC_STATE_KEY] = prolific_id
        st.rerun()


if __name__ == "__main__":
    main()
