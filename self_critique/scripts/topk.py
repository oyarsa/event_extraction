#!/usr/bin/env python
# pyright: basic
import sys

import pandas as pd
import typer


def main(column: str, k: int) -> None:
    col = int(column) if column.isdigit() else column

    df = pd.read_csv(sys.stdin, header=None)  # type: ignore
    df = df.sort_values(by=col, ascending=False)
    df = df.head(k)
    df.to_csv(sys.stdout, index=False)


if __name__ == "__main__":
    typer.run(main)
