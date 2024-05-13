import streamlit as st

from annotation.util import colour


def main() -> None:
    st.title("Instructions")
    st.write("TODO")

    st.page_link(
        "pages/2_Annotation.py", label=colour("Go to annotation page", bg="green")
    )


if __name__ == "__main__":
    main()
