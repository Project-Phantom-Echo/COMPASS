from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.extracted_feature_dataset import (
    DEFAULT_FEATURES_PATH,
    extracted_feature_collate_fn,
    make_extracted_feature_dataset,
)
from models.baseline_model import OctoNetConcatBaseline
from models.cmpt_model_octonet import OctoNetCMPTNet
from shared.utils.optimizers import get_optimizer
from shared.utils.schedulers import get_scheduler
from util import (
    ACTIVITY_LIST,
    alignment_mse_loss,
    normalize_modalities,
    vicreg_loss,
)


METHODS = ("baseline", "cmpt")
MODALITY_ABBREVIATIONS = {
    "imu": "I",
    "uwb": "U",
    "wifi": "W",
    "tof": "T",
    "mmwave": "M",
}
CATEGORY_NAMES = {
    1: "Single",
    2: "Dual",
    3: "Triple",
    4: "Quad",
    5: "5-Modal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified OctoNet_HAR trainer")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument(
        "--features-path",
        default=str(DEFAULT_FEATURES_PATH),
        help="Path to the pre-extracted OctoNet backbone features.",
    )
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, cuda:0")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint file.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("COMPASS_OUTPUT_DIR", str(PROJECT_ROOT / "output")),
        help="Directory in which timestamped run folders are created.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(log_file: Path | None = None) -> logging.Logger:
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y%m%d %H:%M:%S",
    )
    logger = logging.getLogger("octonet_har")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def load_config(config_path: str | os.PathLike[str]) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if "batch_size" not in config:
        config["batch_size"] = config.get("train_loader", {}).get("batch_size", 32)
    if "training_epoch" not in config:
        config["training_epoch"] = 20
    if "max_epoch" not in config:
        config["max_epoch"] = config["training_epoch"]
    if "num_classes" not in config:
        config["num_classes"] = len(ACTIVITY_LIST)
    if "embed_dim" not in config:
        config["embed_dim"] = 512

    return config


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(method: str, config: Mapping[str, object]) -> nn.Module:
    common_kwargs = {
        "modalities": config.get("modality"),
        "embed_dim": int(config.get("embed_dim", 512)),
        "num_classes": int(config.get("num_classes", len(ACTIVITY_LIST))),
    }

    if method == "baseline":
        return OctoNetConcatBaseline(dropout=float(config.get("baseline_dropout", 0.3)), **common_kwargs)
    if method == "cmpt":
        return OctoNetCMPTNet(
            dropout=float(config.get("proxy_dropout", 0.1)),
            head_dropout=float(config.get("cmpt_head_dropout", 0.3)),
            **common_kwargs,
        )
    raise ValueError(f"Unsupported method: {method}")


def build_dataloaders(
    config: Mapping[str, object],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    features_path: str | os.PathLike[str] | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasets = make_extracted_feature_dataset(config, features_path)
    if not isinstance(datasets, tuple) or len(datasets) != 3:
        raise ValueError("OctoNet_HAR/config.yaml must define a 3-way data_split.")

    trainset, valset, testset = datasets
    if len(trainset) == 0:
        raise RuntimeError("Training split is empty after filtering and splitting.")

    pin_memory = device.type == "cuda"
    train_drop_last = len(trainset) >= batch_size

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=train_drop_last,
        collate_fn=extracted_feature_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    eval_kwargs = {
        "batch_size": int(config.get("val_loader", {}).get("batch_size", batch_size)),
        "shuffle": False,
        "num_workers": int(config.get("val_loader", {}).get("num_workers", num_workers)),
        "drop_last": False,
        "collate_fn": extracted_feature_collate_fn,
        "pin_memory": pin_memory,
        "persistent_workers": int(config.get("val_loader", {}).get("num_workers", num_workers)) > 0,
        "prefetch_factor": 4 if int(config.get("val_loader", {}).get("num_workers", num_workers)) > 0 else None,
    }
    val_loader = DataLoader(valset, **eval_kwargs)
    test_loader = DataLoader(testset, **eval_kwargs)
    return train_loader, val_loader, test_loader


def generate_random_mask(modalities: tuple[str, ...], drop_prob: float) -> dict[str, int] | None:
    mask = {modality: 1 for modality in modalities}
    if random.random() > drop_prob:
        return None

    keep_count = random.randint(1, len(modalities) - 1)
    keep_modalities = set(random.sample(list(modalities), keep_count))
    for modality in modalities:
        if modality not in keep_modalities:
            mask[modality] = 0
    return mask


def _avg_lr(scheduler, optimizer) -> float:
    if hasattr(scheduler, "get_lr"):
        learning_rates = scheduler.get_lr()
        if isinstance(learning_rates, (list, tuple)) and learning_rates:
            return float(sum(learning_rates) / len(learning_rates))
    return float(optimizer.param_groups[0]["lr"])


def move_batch_to_device(batch: Mapping[str, object], device: torch.device) -> tuple[dict[str, object], torch.Tensor]:
    labels = batch["label"].to(device=device, dtype=torch.long)
    features = {
        "globals": {
            modality: tensor.to(device=device, dtype=torch.float32)
            for modality, tensor in batch["features"]["globals"].items()
        },
        "tokens": {
            modality: tensor.to(device=device, dtype=torch.float32)
            for modality, tensor in batch["features"]["tokens"].items()
        },
        "token_lengths": {
            modality: tensor.to(device=device, dtype=torch.long)
            for modality, tensor in batch["features"].get("token_lengths", {}).items()
        },
    }
    return {"features": features}, labels


def _forward_extracted_baseline(
    model: OctoNetConcatBaseline,
    features: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    missing_mask: Mapping[str, int] | None = None,
) -> torch.Tensor:
    global_features = features["globals"]
    reference_tensor = next(iter(global_features.values()))
    batch_size = reference_tensor.size(0)
    device = reference_tensor.device
    dtype = reference_tensor.dtype

    fused_features = []
    for modality in model.modalities:
        feature = global_features.get(modality)
        use_real = feature is not None and (missing_mask is None or missing_mask.get(modality, 1) == 1)
        if use_real:
            fused_features.append(feature)
        else:
            fused_features.append(torch.zeros(batch_size, model.embed_dim, device=device, dtype=dtype))

    return model.classifier(torch.cat(fused_features, dim=-1))


def _run_generator_on_padded_tokens(
    generator: nn.Module,
    tokens: torch.Tensor,
    lengths: torch.Tensor | None,
) -> torch.Tensor:
    if lengths is None:
        return generator(tokens)

    if tokens.size(0) == 0:
        return tokens.new_zeros((0, tokens.size(-1)))

    if int(lengths.min().item()) == int(lengths.max().item()) == int(tokens.size(1)):
        return generator(tokens)

    proxies = []
    feature_dim = int(tokens.size(-1))
    for sample_idx in range(tokens.size(0)):
        valid_length = int(lengths[sample_idx].item())
        if valid_length <= 0:
            proxies.append(tokens.new_zeros(feature_dim))
            continue
        sample_tokens = tokens[sample_idx : sample_idx + 1, :valid_length]
        proxies.append(generator(sample_tokens).squeeze(0))
    return torch.stack(proxies, dim=0)


def _forward_extracted_cmpt(
    model: OctoNetCMPTNet,
    features: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    missing_mask: Mapping[str, int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    global_features = features["globals"]
    token_features = features["tokens"]
    token_lengths = features.get("token_lengths", {})

    reference_tensor = next(iter(global_features.values()))
    batch_size = reference_tensor.size(0)
    device = reference_tensor.device
    dtype = reference_tensor.dtype

    real_globals = {
        modality: global_features[modality]
        for modality in model.modalities
        if modality in global_features
    }
    real_tokens = {modality: token_features[modality] for modality in model.modalities if modality in token_features}

    proxies: dict[str, torch.Tensor] = {}
    for src, tgt in model.proxy_pairs:
        source_is_present = missing_mask is None or missing_mask.get(src, 1) == 1
        if src not in real_tokens or not source_is_present:
            continue
        key = f"{src}_to_{tgt}"
        proxies[key] = _run_generator_on_padded_tokens(
            model.generators[key],
            real_tokens[src],
            token_lengths.get(src),
        )

    fused_features = []
    for target in model.modalities:
        use_real = target in real_globals and (missing_mask is None or missing_mask.get(target, 1) == 1)
        if use_real:
            fused_features.append(real_globals[target])
            continue

        candidate_proxies = [
            proxies[f"{src}_to_{target}"]
            for src in model.modalities
            if src != target and f"{src}_to_{target}" in proxies
        ]
        if candidate_proxies:
            fused_features.append(torch.stack(candidate_proxies, dim=0).mean(dim=0))
        else:
            fused_features.append(torch.zeros(batch_size, model.embed_dim, device=device, dtype=dtype))

    fused = torch.stack(fused_features, dim=0).sum(dim=0)
    logits = model.classifier(fused)
    return logits, proxies, real_globals


def forward_model(
    model: nn.Module,
    batch_inputs: Mapping[str, object],
    *,
    method: str,
    missing_mask: Mapping[str, int] | None = None,
):
    features = batch_inputs["features"]
    if method == "baseline":
        return _forward_extracted_baseline(model, features, missing_mask=missing_mask)
    if method == "cmpt":
        return _forward_extracted_cmpt(model, features, missing_mask=missing_mask)
    raise ValueError(f"Unsupported method: {method}")


def _build_robustness_specs(modalities: Sequence[str]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for subset_size in range(1, len(modalities) + 1):
        category = CATEGORY_NAMES.get(subset_size, f"{subset_size}-Modal")
        for subset in combinations(modalities, subset_size):
            mask = {modality: int(modality in subset) for modality in modalities}
            label = "+".join(MODALITY_ABBREVIATIONS[modality] for modality in subset)
            specs.append(
                {
                    "category": category,
                    "modalities": subset,
                    "display": label,
                    "mask": mask,
                }
            )
    return specs


def _format_robustness_table(rows: Sequence[Mapping[str, object]]) -> str:
    headers = ("Category", "Modalities", "Accuracy", "Loss")
    body = [
        (
            str(row["category"]),
            str(row["modalities"]),
            f"{float(row['accuracy']) * 100.0:.2f}%",
            f"{float(row['loss']):.4f}",
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for values in body:
        for idx, value in enumerate(values):
            widths[idx] = max(widths[idx], len(value))

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header_row = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"

    lines = [border, header_row, border]
    previous_category = None
    for values in body:
        current_category = values[0]
        if previous_category is not None and current_category != previous_category:
            lines.append(border)
        lines.append("| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |")
        previous_category = current_category
    lines.append(border)
    return "\n".join(lines)


def evaluate_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config: Mapping[str, object],
    method: str,
    *,
    output_dir: str | Path | None = None,
    output_filename: str = "robustness.csv",
    title: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    model.eval()
    modalities = normalize_modalities(config.get("modality"))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.1 if method == "cmpt" else 0.0))
    )
    specs = _build_robustness_specs(modalities)

    stats = {
        spec["display"]: {
            "category": spec["category"],
            "modalities": spec["display"],
            "mask": spec["mask"],
            "loss_sum": 0.0,
            "correct": 0,
            "total": 0,
        }
        for spec in specs
    }

    with torch.no_grad():
        iterator = tqdm(test_loader, desc="robustness", leave=False)
        for batch in iterator:
            batch_inputs, labels = move_batch_to_device(batch, device)
            batch_size = int(labels.size(0))

            for spec in specs:
                if method == "cmpt":
                    logits, _, _ = forward_model(
                        model,
                        batch_inputs,
                        method=method,
                        missing_mask=spec["mask"],
                    )
                else:
                    logits = forward_model(
                        model,
                        batch_inputs,
                        method=method,
                        missing_mask=spec["mask"],
                    )

                loss = criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)
                key = spec["display"]
                stats[key]["loss_sum"] += loss.item() * batch_size
                stats[key]["correct"] += int((predictions == labels).sum().item())
                stats[key]["total"] += batch_size

    rows = []
    grouped_results = {category: [] for category in CATEGORY_NAMES.values()}
    for spec in specs:
        key = spec["display"]
        entry = stats[key]
        total = max(int(entry["total"]), 1)
        row = {
            "category": str(entry["category"]),
            "modalities": str(entry["modalities"]),
            "accuracy": float(entry["correct"]) / float(total),
            "loss": float(entry["loss_sum"]) / float(total),
        }
        rows.append(row)
        grouped_results[row["category"]].append(row)

    table = _format_robustness_table(rows)
    heading = title or "ID"
    if logger is not None:
        logger.info("%s robustness evaluation over 31 modality subsets:\n%s", heading, table)
    else:
        print(table)

    csv_root = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "output"
    csv_root.mkdir(parents=True, exist_ok=True)
    csv_path = csv_root / output_filename
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "modalities", "accuracy", "loss"])
        writer.writeheader()
        writer.writerows(rows)

    if logger is not None:
        logger.info("Saved robustness results to %s", csv_path)

    return {
        "rows": rows,
        "by_category": grouped_results,
        "csv_path": str(csv_path),
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    method: str,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = move_batch_to_device(batch, device)
            if method == "cmpt":
                logits, _, _ = forward_model(model, inputs, method=method, missing_mask=None)
            else:
                logits = forward_model(model, inputs, method=method, missing_mask=None)

            loss = criterion(logits, labels)
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    method: str,
    device: torch.device,
    optimizer,
    scheduler,
    criterion: nn.Module,
    config: Mapping[str, object],
) -> dict[str, float]:
    model.train()

    cmpt_modalities = normalize_modalities(config.get("modality"))
    drop_prob = float(config.get("drop_prob", 0.7))
    align_weight = float(config.get("cmpt_loss_weight", 0.2))
    lambda_vicreg = float(config.get("lambda_vicreg", 0.0))
    vicreg_inv = float(config.get("vicreg_inv", 25.0))
    vicreg_var = float(config.get("vicreg_var", 25.0))
    vicreg_cov = float(config.get("vicreg_cov", 1.0))
    total_loss = 0.0
    total_cls = 0.0
    total_align = 0.0
    total_vicreg = 0.0
    total_correct = 0
    total_samples = 0

    iterator = tqdm(dataloader, desc=f"train lr={_avg_lr(scheduler, optimizer):.6f}", leave=False)
    for batch in iterator:
        inputs, labels = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        if method == "cmpt":
            missing_mask = generate_random_mask(cmpt_modalities, drop_prob)
            logits, proxies, real_globals = forward_model(model, inputs, method=method, missing_mask=missing_mask)
            cls_loss = criterion(logits, labels)
            align_loss = alignment_mse_loss(proxies, real_globals)
            vic_loss = vicreg_loss(
                real_globals,
                inv_coeff=vicreg_inv,
                var_coeff=vicreg_var,
                cov_coeff=vicreg_cov,
            )
            loss = cls_loss + align_weight * align_loss + lambda_vicreg * vic_loss
        else:
            logits = forward_model(model, inputs, method=method, missing_mask=None)
            cls_loss = criterion(logits, labels)
            align_loss = torch.zeros((), device=device)
            vic_loss = torch.zeros((), device=device)
            loss = cls_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_cls += cls_loss.item() * labels.size(0)
        total_align += align_loss.item() * labels.size(0)
        total_vicreg += vic_loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        iterator.set_description(
            f"train loss={loss.item():.4f} cls={cls_loss.item():.4f} "
            f"align={align_loss.item():.4f} vic={vic_loss.item():.4f}"
        )

    if total_samples == 0:
        return {"loss": 0.0, "cls_loss": 0.0, "align_loss": 0.0, "vicreg_loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total_samples,
        "cls_loss": total_cls / total_samples,
        "align_loss": total_align / total_samples,
        "vicreg_loss": total_vicreg / total_samples,
        "accuracy": total_correct / total_samples,
    }


def save_checkpoint(
    checkpoint_path: Path,
    *,
    method: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scheduler,
    best_val_acc: float,
    config: Mapping[str, object],
) -> None:
    torch.save(
        {
            "method": method,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "config": dict(config),
        },
        checkpoint_path,
    )


def load_model_state(model: nn.Module, checkpoint: Mapping[str, object]) -> int:
    model_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(model_state, Mapping):
        raise TypeError("Checkpoint does not contain a valid model state dictionary.")

    normalized_state = {
        key.removeprefix("module."): value
        for key, value in model_state.items()
    }
    encoder_keys = [key for key in normalized_state if key.startswith("encoders.")]
    feature_model_state = {
        key: value
        for key, value in normalized_state.items()
        if not key.startswith("encoders.")
    }
    incompatible = model.load_state_dict(feature_model_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the extracted-feature model: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return len(encoder_keys)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training_epoch"] = args.epochs
        config["max_epoch"] = args.epochs
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
        config.setdefault("train_loader", {})["num_workers"] = args.num_workers
        config.setdefault("val_loader", {})["num_workers"] = args.num_workers
    if args.seed is not None:
        config["init_rand_seed"] = args.seed
    config["features_path"] = args.features_path

    device = resolve_device(args.device)
    seed_everything(int(config.get("init_rand_seed", 41)))

    batch_size = int(config.get("batch_size", 32))
    num_workers = int(config.get("num_workers", 4))
    max_epoch = int(config.get("max_epoch", config.get("training_epoch", 20)))
    eval_interval = int(config.get("eval_interval", 1))

    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(run_dir / "train.log")
    logger.info(f"method={args.method} device={device}")
    logger.info(f"extracted_features={args.features_path}")
    logger.info("evaluation_protocol=ID")
    logger.info(f"outputs={run_dir}")

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    train_loader, val_loader, test_loader = build_dataloaders(
        config,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        features_path=args.features_path,
    )

    model = build_model(args.method, config).to(device)
    if args.eval_only and not args.resume:
        raise ValueError("--eval-only requires --resume with a checkpoint path.")

    optimizer = get_optimizer(
        model,
        optimizer=config.get("optimizer", "adamw"),
        lr=float(config.get("learning_rate", 1.0e-4)),
        weight_decay=float(config.get("weight_decay", 5.0e-4)),
    )

    train_iters = max(1, len(train_loader))
    scheduler = get_scheduler(
        config.get("scheduler", "warmuppolylr"),
        optimizer,
        int((max_epoch + 1) * train_iters),
        float(config.get("power", 0.9)),
        train_iters * int(config.get("warmup", 5)),
        float(config.get("warmup_ratio", 0.1)),
    )

    label_smoothing = float(config.get("label_smoothing", 0.1 if args.method == "cmpt" else 0.0))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_checkpoint = run_dir / "best.pt"
    last_checkpoint = run_dir / "last.pt"
    best_val_acc = -1.0
    start_epoch = 0

    if args.resume and not args.eval_only:
        checkpoint = torch.load(args.resume, map_location="cpu")
        ignored_encoder_keys = load_model_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", -1.0))
        logger.info(f"resumed from {args.resume} at epoch {start_epoch}")
        if ignored_encoder_keys:
            logger.info("ignored %d frozen encoder tensors from the legacy checkpoint", ignored_encoder_keys)
    elif args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        ignored_encoder_keys = load_model_state(model, checkpoint)
        logger.info(f"loaded checkpoint for evaluation from {args.resume}")
        if ignored_encoder_keys:
            logger.info("ignored %d frozen encoder tensors from the legacy checkpoint", ignored_encoder_keys)

    if args.eval_only:
        evaluate_robustness(
            model,
            test_loader,
            device,
            config,
            args.method,
            output_dir=run_dir,
            output_filename="robustness.csv",
            title="ID",
            logger=logger,
        )
        return

    for epoch in range(start_epoch, max_epoch):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            method=args.method,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            config=config,
        )

        logger.info(
            f"[epoch {epoch + 1:02d}/{max_epoch:02d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"cls={train_metrics['cls_loss']:.4f} "
            f"align={train_metrics['align_loss']:.4f} "
            f"vicreg={train_metrics['vicreg_loss']:.4f}"
        )

        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == max_epoch:
            val_metrics = evaluate(model, val_loader, method=args.method, device=device, criterion=criterion)
            test_metrics = evaluate(model, test_loader, method=args.method, device=device, criterion=criterion)
            logger.info(
                f"[epoch {epoch + 1:02d}] "
                f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
                f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] >= best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                save_checkpoint(
                    best_checkpoint,
                    method=args.method,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    best_val_acc=best_val_acc,
                    config=config,
                )
                logger.info(f"saved new best checkpoint to {best_checkpoint}")

        save_checkpoint(
            last_checkpoint,
            method=args.method,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_val_acc=best_val_acc,
            config=config,
        )

    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location="cpu")
        load_model_state(model, checkpoint)

    final_val = evaluate(model, val_loader, method=args.method, device=device, criterion=criterion)
    final_test = evaluate(model, test_loader, method=args.method, device=device, criterion=criterion)
    logger.info(
        f"[final] best_val_acc={final_val['accuracy']:.4f} "
        f"best_test_acc={final_test['accuracy']:.4f}"
    )

    evaluate_robustness(
        model,
        test_loader,
        device,
        config,
        args.method,
        output_dir=run_dir,
        output_filename="robustness.csv",
        title="ID",
        logger=logger,
    )


if __name__ == "__main__":
    main()
