# noqa: N999
from annotation.common import ask_login, check_prolific_id
from annotation.instructions import heading, render_instructions


def main() -> None:
    heading("Welcome", 1)
    render_instructions()

    if not check_prolific_id():
        ask_login()


if __name__ == "__main__":
    main()
