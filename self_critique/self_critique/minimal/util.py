# pyright: basic
import logging
import os
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import PreTrainedModel, PreTrainedTokenizer


def setup_logging(log_level: str) -> None:
    logging.basicConfig(
        level=logging.getLevelName(log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_metrics(metrics: dict[str, float], desc: str | None) -> None:
    desc = desc or "metrics"
    logging.info(f">>>> {desc.upper()}")

    padding = max(len(k) for k in metrics)
    for k, v in metrics.items():
        logging.info(f"    {k:>{padding}}: {v}")


def suppress_transformers_warnings() -> None:
    "Remove annoying messages about tokenisers and unititialised weights."
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", module="transformers.convert_slow_tokenizer")
    transformers.logging.set_verbosity_error()


def set_seed(seed: int) -> None:
    "Set random seed for reproducibility."
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_model(
    model: PreTrainedModel, tokeniser: PreTrainedTokenizer, output_dir: Path
) -> None:
    model.config.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    tokeniser.save_pretrained(output_dir)
