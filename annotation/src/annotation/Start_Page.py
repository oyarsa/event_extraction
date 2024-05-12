import extra_streamlit_components as stx
import streamlit as st

from annotation.common import (
    get_prolific_id,
    reset_prolific_id,
    section_links,
    set_prolific_state_id,
)


# I'm not sure why the experimental flag is needed since it's not in the example.
# It _looks like_ it works.
@st.cache_resource(experimental_allow_widgets=True)
def get_manager() -> stx.CookieManager:
    return stx.CookieManager()


_PROLIFIC_COOKIE_KEY = "prolific_id"


def main() -> None:
    st.set_page_config(page_title="Event Extraction Annotation")

    cookies = get_manager()

    st.title("Welcome")

    if prolific_id := cookies.get(_PROLIFIC_COOKIE_KEY):
        set_prolific_state_id(prolific_id)

    if get_prolific_id():
        st.write(
            "You're logged in. Move on to the annotation page or review the"
            " instructions."
        )
        if st.button("Log out", type="primary"):
            cookies.delete(_PROLIFIC_COOKIE_KEY)
            reset_prolific_id()
            st.rerun()

        section_links()
        return

    if prolific_id := st.text_input(
        "Enter your Prolific ID",
        key="prolific_input_id",
        placeholder="Prolific ID",
    ):
        set_prolific_state_id(prolific_id)
        cookies.set(_PROLIFIC_COOKIE_KEY, prolific_id)
        st.rerun()


if __name__ == "__main__":
    main()
