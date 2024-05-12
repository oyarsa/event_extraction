from pathlib import Path

import streamlit as st


def main() -> None:
    admin_path = Path("config/admin_key")
    if not admin_path.exists():
        st.error("Admin key not found")
        return

    admin_key = admin_path.read_text().strip()
    key = st.text_input("Enter the admin key")
    if not key:
        return

    if key != admin_key:
        st.error("Invalid admin key")
        return

    st.header("Admin panel")
    st.write("TODO")


if __name__ == "__main__":
    main()
