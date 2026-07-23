"""Small runtime helpers shared by the three benchmark pipelines."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn


def fix_seeds(seed: int = 3407) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_cudnn() -> None:
    cudnn.benchmark = True
    cudnn.deterministic = False


def get_logger(log_file: str | Path | None = None) -> logging.Logger:
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s: - %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
