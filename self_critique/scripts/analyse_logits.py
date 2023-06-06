#!/usr/bin/env python3
from pathlib import Path
from typing import cast

import numpy as np
import torch
import typer


def main(path: Path) -> None:
    predictions = torch.from_numpy(np.load(path))

    soft = predictions.softmax(dim=1)
    maxs = cast(torch.Tensor, soft.max(dim=1)[0])
    orders = (maxs / soft.t()).t().log10()
    avg_order = orders.sum(dim=1)
    print(avg_order.reshape(-1, 1))


if __name__ == "__main__":
    typer.run(main)
