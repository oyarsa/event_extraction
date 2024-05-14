import json

import streamlit as st

from annotation.components import get_username, logout_button
from annotation.util import check_auth, get_config


def validate_file(file_bytes: bytes, keys: list[str]) -> str | None:
    """Check if the file is a valid JSON with the correct format.

    Returns the error message if invalid, otherwise None.
    """
    try:
        data = json.loads(file_bytes)
    except json.JSONDecodeError:
        return "Invalid JSON file"

    if not isinstance(data, list):
        return "JSON file should be a list"

    valid = all(
        isinstance(item, dict) and all(key in item for key in keys) for item in data
    )
    if not valid:
        return f"Every item should be an object with keys {keys}"
    return None


def main() -> None:
    username = get_username()
    if not username:
        return

    password = st.text_input("Password", type="password")
    if not password:
        st.error("Please enter a password")
        return
    if not check_auth(password):
        st.error("Incorrect password")
        return

    logout_button()

    st.header("Admin panel")

    file = st.file_uploader("Choose a JSON data file")
    if not file:
        return

    file_path = get_config().annotation_dir / file.name
    if file_path.exists():
        st.warning(
            f"File {file_path} already exists. Confirm if you want to overwrite."
        )
        if not st.button("Overwrite", key=f"confirm_{file.name}"):
            return

    keys = ["text", "annotation", "model"]
    if error := validate_file(file.getvalue(), keys):
        st.error(error)
        return

    file_path.write_bytes(file.getvalue())
    st.write(f"Uploaded {file_path}")


if __name__ == "__main__":
    main()
