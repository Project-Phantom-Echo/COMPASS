from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.extracted_feature_dataset import DEFAULT_FEATURES_PATH
from util import ACTIVITY_LIST, collate_fn, normalize_modalities, resolve_octonet_dataset_path


DEFAULT_DATASET_PATH = str(REPO_ROOT / "data" / "OctoNet" / "dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Extract OctoNet_HAR backbone features")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def load_config(config_path: str | os.PathLike[str]) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["dataset_path"] = resolve_octonet_dataset_path(
        config_path,
        config.get("dataset_path"),
    )
    if "batch_size" not in config:
        config["batch_size"] = 32
    if "num_classes" not in config:
        config["num_classes"] = len(ACTIVITY_LIST)
    if "embed_dim" not in config:
        config["embed_dim"] = 512
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_full_dataset(config: dict):
    from dataset.octonet_dataset import OctoNetHARDataset

    return OctoNetHARDataset(
        dataset_path=str(config.get("dataset_path", DEFAULT_DATASET_PATH)),
        modalities=config.get("modality"),
        user_list=config.get("user_list"),
        activity_list=config.get("activity_list"),
        segmentation_flag=bool(config.get("segmentation_flag", True)),
        min_file_size_bytes=int(config.get("min_file_size_bytes", 100)),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    from Encoders import EncoderToDict

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers

    seed_everything(int(config.get("init_rand_seed", 41)))
    device = resolve_device(args.device)
    modalities = normalize_modalities(config.get("modality"))

    dataset = build_full_dataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 4)),
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 4)) > 0,
        prefetch_factor=4 if int(config.get("num_workers", 4)) > 0 else None,
    )

    encoders = EncoderToDict(modalities, freeze_backbones=True).to(device)
    encoders.eval()

    records: list[dict] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="extract", leave=True):
            inputs = {
                modality: tensor.to(device=device, dtype=torch.float32)
                for modality, tensor in batch["modality_data"].items()
            }
            encoded = encoders(inputs, return_tokens=True, return_global=True)

            batch_size = int(batch["label"].size(0))
            for sample_idx in range(batch_size):
                record = {
                    "label": int(batch["label"][sample_idx].item()),
                    "user_id": str(batch["user_id"][sample_idx]),
                    "recording_id": str(batch["recording_id"][sample_idx]),
                    "globals": {},
                    "tokens": {},
                }
                for modality in modalities:
                    modality_outputs = encoded[modality]
                    record["globals"][modality] = modality_outputs["global"][sample_idx].detach().cpu().float()
                    record["tokens"][modality] = modality_outputs["tokens"][sample_idx].detach().cpu().float()
                records.append(record)

    # Save the extracted tensors together with split metadata.
    label_hash = hash(tuple(r["label"] for r in records))
    feature_payload = {
        "metadata": {
            "num_samples": len(records),
            "modalities": sorted(modalities),
            "user_list": sorted(config.get("user_list", [])),
            "num_classes": config.get("num_classes", 62),
            "seed": config.get("init_rand_seed", 41),
            "label_hash": label_hash,
            "split": "id",
        },
        "records": records,
    }
    output_target = args.output or config.get("features_path") or DEFAULT_FEATURES_PATH
    output_path = Path(output_target).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    else:
        output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feature_payload, output_path)
    print(f"Saved {len(records)} extracted feature records to {output_path}")


if __name__ == "__main__":
    main()
