import streamlit as st
from evaluation.gpt_common import hashlib

from annotation.common import get_prolific_id


def hashkey(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def main() -> None:
    prolific_id = get_prolific_id("annotation")
    if not prolific_id:
        return

    if prolific_id != "admin":
        st.error("You are not an admin")
        return

    st.header("Admin panel")
    st.write("TODO")


if __name__ == "__main__":
    main()
