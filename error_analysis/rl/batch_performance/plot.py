from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pixcat
import typer


def main(data_path: Path, batches_per_epoch: int, y_label: str) -> None:
    data = pd.read_csv(data_path, header=None, names=["batch", y_label])
    plt.xlabel("batch")
    plt.ylabel(y_label)
    plt.xticks(
        np.arange(min(data.batch), max(data.batch) + 1, 50), fontsize=8, rotation=30
    )
    plt.grid()

    bins = range(0, max(data.batch) + batches_per_epoch, batches_per_epoch)
    data["epoch"] = pd.cut(data["batch"], bins=bins, labels=False, include_lowest=True)

    colormap = plt.colormaps.get_cmap("viridis")
    for epoch_, group in data.groupby("epoch"):
        epoch = cast(int, epoch_)
        color = colormap(epoch / (len(bins) - 1))
        plt.plot(group["batch"], group[y_label], color=color, label=f"Epoch {epoch}")

    plt.legend(loc="best")
    plt.savefig("output.png", dpi=600)
    pixcat.Image("output.png").fit_screen(enlarge=True).show()


if __name__ == "__main__":
    typer.run(main)
