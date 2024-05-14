import json
import logging
from pathlib import Path

import streamlit as st

from annotation.components import get_annotation_path, get_username, logout_button
from annotation.util import backup_and_write, check_admin_password, get_config

logger = logging.getLogger("annotation.admin")


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
        isinstance(item, dict) and all(key in item for key in keys)
        for item in data  # pyright: ignore[reportUnknownVariableType]
    )
    if not valid:
        return f"Every item should be an object with keys {keys}"
    return None


def main(annotation_dir: Path, split_to_user_file: Path) -> None:
    st.header("Admin panel")

    username = get_username()
    if not username:
        return

    admin_password_key = "admin_password"
    password = st.session_state.get(admin_password_key) or st.text_input(
        "Password", type="password"
    )
    if not password:
        st.warning("Please enter a password")
        return
    if not check_admin_password(password):
        st.error("Incorrect password")
        return

    st.session_state[admin_password_key] = password
    logout_button()

    if path := get_annotation_path(annotation_dir, split_to_user_file, username):
        st.markdown(f"Your data file is `{path}`.")

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

    keys = ["text", "reference", "model"]
    if error := validate_file(file.getvalue(), keys):
        st.error(error)
        return

    file_path.write_bytes(file.getvalue())
    split_to_user = json.loads(split_to_user_file.read_text())
    split_to_user[file_path.stem] = None
    backup_and_write(split_to_user_file, json.dumps(split_to_user, indent=2))

    logger.info("Uploaded %s and updated the source-to-user file.", file_path)
    st.markdown(f"Uploaded {file_path}")


if __name__ == "__main__":
    config = get_config()
    main(config.annotation_dir, config.split_to_user_file)
