import subprocess
import tempfile
from collections import Counter

import pandas as pd
import pixcat


def remove_commit_type(commit: str) -> str:
    parts = commit.strip().split(":", 1)
    return parts[0] if len(parts) == 1 else parts[1]


def commit_len(commit: str) -> int:
    return len(remove_commit_type(commit))


def get_commits() -> list[str]:
    git_output = subprocess.check_output("git shortlog", shell=True, text=True)
    return [
        remove_commit_type(commit.strip())
        for commit in git_output.strip().splitlines()
        if commit.startswith(" ")
    ]


def main() -> None:
    data = Counter(map(commit_len, get_commits()))
    df = pd.DataFrame(sorted(data.items()), columns=["cols", "commits"])

    plot = df.plot(x="cols", y="commits")

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        plot.figure.savefig(f.name)
        pixcat.Image(f.name).fit_screen(enlarge=True).show()


if __name__ == "__main__":
    main()
