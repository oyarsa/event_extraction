from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pixcat
import typer


def main(data_path: Path, n: int, y_label: str) -> None:
    data = pd.read_csv(data_path, header=None, names=["batch", y_label])
    plt.xlabel("batch")
    plt.ylabel(y_label)
    plt.xticks(
        np.arange(min(data.batch), max(data.batch) + 1, 10), fontsize=8, rotation=30
    )
    plt.grid()

    groups = list(data.groupby("batch"))
    chunks = [
        pd.concat([group for _, group in groups[i : i + n]])
        for i in range(0, len(groups), n)
    ]

    colormap = plt.colormaps.get_cmap("viridis")
    for i, chunk in enumerate(chunks):
        color = colormap(i / len(chunks))
        plt.plot(chunk["batch"], chunk[y_label], color=color)
        plt.plot([], [], color=color, label=f"Epoch {i+1}")

    plt.legend(loc="upper right", title="Legend")

    plt.savefig("output.png")
    pixcat.Image("output.png").fit_screen(enlarge=True).show()


if __name__ == "__main__":
    typer.run(main)
