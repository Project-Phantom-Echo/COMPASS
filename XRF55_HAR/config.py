"""Configuration used by the XRF55 COMPASS training scripts."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


BASE_CONFIG = {
    "exp_name": "compass-xrf55",
    "seed": 0,
    "device": "cuda",
    "gpu_ids": [0],
    "num_workers": 16,
    "data_dir": os.environ.get(
        "XRF55_DATA_ROOT",
        str(REPO_ROOT / "data" / "XRF55"),
    ),
    "save_dir": os.environ.get(
        "COMPASS_OUTPUT_DIR",
        str(REPO_ROOT / "outputs" / "XRF55"),
    ),
    "scene": "all",
    "class_num": 55,
    # 0.0 reproduces the released behaviour: the checkpoint is selected on the
    # test set. Set >0 to hold out that fraction of the training split for
    # selection instead, leaving the test set untouched until the final report.
    "val_ratio": 0.0,
    "batch_size": 32,
    "max_epoch": 100,
    "optim_type": "adamw",
    "learning_rate": 1e-4,
    "weight_decay": 5e-2,
    "scheduler": "warmuppolylr",
    "power": 0.9,
    "warmup": 5,
    "warmup_ratio": 0.1,
    "simulate_missing": True,
    "drop_prob": 0.7,
    "cmpt_dropout": 0.3,
    "cmpt_loss_weight": 0.2,
    "lambda_vicreg": 0.3,
    "vicreg_inv": 5.0,
    "vicreg_var": 25.0,
    "vicreg_cov": 1.0,
    "lambda_proxy_cls": 0.5,
    "lambda_recon": 0.5,
    "freeze_backbone": False,
    "fusion_type": "sum",
    "proxy_agg": "uniform",
    "conf_temperature": 1.0,
    "generator_type": "transformer",
    "generator_mode": "pairwise",
    "source_priority": None,
    "missing_fill": "proxy",
    "impute_hidden_dim": None,
    "impute_dropout": 0.3,
    "no_proxy": False,
    "wandb_exp_name": "XRF55",
}


NAMED_CONFIGS = {
    "task_finetune_xrf55": {},
}


def _parse_value(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_config(argv=None):
    """Parse the historical ``with task key=value`` command syntax."""
    if argv is None:
        argv = sys.argv[1:]

    task_name = None
    overrides = {}
    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg == "with":
            i += 1
            continue
        if arg == "--task":
            if i + 1 >= len(argv):
                raise ValueError("--task requires a value")
            task_name = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--"):
            key = arg[2:].replace("-", "_")
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a value")
            overrides[key] = _parse_value(argv[i + 1])
            i += 2
            continue
        if "=" in arg:
            key, value = arg.split("=", 1)
            overrides[key.replace("-", "_")] = _parse_value(value)
            i += 1
            continue
        if task_name is None and arg in NAMED_CONFIGS:
            task_name = arg
            i += 1
            continue
        raise ValueError(f"Unrecognized argument: {arg}")

    config = dict(BASE_CONFIG)
    if task_name is not None:
        if task_name not in NAMED_CONFIGS:
            raise ValueError(f"Unknown task: {task_name}")
        config.update(NAMED_CONFIGS[task_name])
    config.update(overrides)
    return config
