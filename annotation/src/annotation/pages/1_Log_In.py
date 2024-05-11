import streamlit as st

from annotation.common import colour, get_prolific_id, set_prolific_id


def main() -> None:
    if get_prolific_id():
        st.write(
            "You're logged in. Move on to the annotation page or review the"
            " instructions."
        )

        instructions, annotation, _ = st.columns([0.1, 0.1, 0.6])
        instructions.page_link(
            "pages/2_Instructions.py",
            label=colour("Instructions", bg="red"),
        )
        annotation.page_link(
            "pages/3_Annotation.py", label=colour("Annotation page", bg="green")
        )
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
