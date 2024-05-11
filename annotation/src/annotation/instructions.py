import streamlit as st

from annotation.common import heading


def render_instructions() -> None:
    heading("Instructions", 1)
    st.write("TODO")
