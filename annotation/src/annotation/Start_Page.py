import streamlit as st

from annotation.common import (
    get_config,
    get_prolific_id,
    reset_session_state,
    section_links,
    set_prolific_state_id,
)


def main() -> None:
    st.set_page_config(page_title="Event Extraction Annotation")

    st.title("Welcome")

    if get_prolific_id():
        st.write(
            "You're logged in. Move on to the annotation page or review the"
            " instructions."
        )
        if st.button("Log out", type="primary"):
            reset_session_state()
            st.rerun()

        section_links()
        return

    if prolific_id := st.text_input(
        "Enter your Prolific ID",
        key="prolific_input_id",
        placeholder="Prolific ID",
    ):
        config = get_config()
        # Data is divided in files, one per Prolific ID.
        annotation_path = config.annotation_dir / f"{prolific_id}.json"
        if not annotation_path.exists():
            st.error("Invalid Prolific ID. Please try again.")
            return

        set_prolific_state_id(prolific_id)
        st.rerun()


if __name__ == "__main__":
    main()
