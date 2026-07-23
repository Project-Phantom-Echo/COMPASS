from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent


def get_octonet_benchmark_root() -> Path:
    configured = os.environ.get("OCTONET_BENCHMARK_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    in_repo = REPO_ROOT / "third_party" / "OctoNet" / "OctonetBenchmark"
    legacy_in_repo = REPO_ROOT / "OctoNet" / "OctonetBenchmark"
    sibling = REPO_ROOT.parent / "OctoNet" / "OctonetBenchmark"
    candidates = (in_repo, legacy_in_repo, sibling)
    return next((path.resolve() for path in candidates if path.exists()), in_repo.resolve())


def resolve_octonet_dataset_path(
    config_path: str | Path,
    configured_path: str | Path | None,
) -> str:
    environment_value = os.environ.get("OCTONET_DATA_ROOT")
    value = environment_value or configured_path
    if value is None:
        return str((REPO_ROOT / "data" / "OctoNet" / "dataset").resolve())

    path = Path(value).expanduser()
    if not path.is_absolute():
        base_dir = REPO_ROOT if environment_value else Path(config_path).expanduser().resolve().parent
        path = base_dir / path
    return str(path.resolve())


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    octonet_name: str
    nodes: tuple[int, ...]
    in_channels: int
    target_shape: tuple[int, ...]
    weight_relpath: str
    path_columns: tuple[str, ...]


# OctoNet raw metadata exposes 64 activities, but the released 62-way
# checkpoints exclude the final two benchmark extras: gym and freestyle.
ACTIVITY_LIST = (
    "sit",
    "walk",
    "bow",
    "sleep",
    "dance",
    "jog",
    "falldown",
    "jump",
    "jumpingjack",
    "squat",
    "lunge",
    "turn",
    "pushup",
    "legraise",
    "airdrum",
    "boxing",
    "shakehead",
    "answerphone",
    "eat",
    "drink",
    "wipeface",
    "pickup",
    "jumprope",
    "moppingfloor",
    "brushhair",
    "bicepcurl",
    "playphone",
    "brushteeth",
    "type",
    "thumbup",
    "thumbdown",
    "makeoksign",
    "makevictorysign",
    "drawcircleclockwise",
    "drawcirclecounterclockwise",
    "stopsign",
    "pullhandin",
    "pushhandaway",
    "handwave",
    "sweep",
    "clap",
    "slide",
    "drawzigzag",
    "dodge",
    "bowling",
    "liftupahand",
    "tap",
    "spreadandpinch",
    "drawtriangle",
    "sneeze",
    "cough",
    "stagger",
    "yawn",
    "blownose",
    "stretchoneself",
    "touchface",
    "handshake",
    "hug",
    "pushsomeone",
    "kicksomeone",
    "punchsomeone",
    "conversation",
)

ACTIVITY_TO_INDEX = {activity: idx for idx, activity in enumerate(ACTIVITY_LIST)}
INDEX_TO_ACTIVITY = {idx: activity for activity, idx in ACTIVITY_TO_INDEX.items()}

DEFAULT_USER_LIST = (1, 2, 3, 4, 5, 6, 7, 111, 102, 104, 101, 108)

MODALITY_SPECS = {
    "imu": ModalitySpec(
        name="imu",
        octonet_name="imu",
        nodes=(1,),
        in_channels=1,
        target_shape=(1, 360, 221),
        weight_relpath="backbones/imu/resnet18.pth",
        path_columns=("imu_data_path",),
    ),
    "uwb": ModalitySpec(
        name="uwb",
        octonet_name="uwb",
        nodes=(1,),
        in_channels=1,
        target_shape=(1, 102, 1535),
        weight_relpath="backbones/uwb/resnet18.pth",
        path_columns=("node_1_uwb_data_path",),
    ),
    "wifi": ModalitySpec(
        name="wifi",
        octonet_name="wifi",
        nodes=(1, 2, 3, 4),
        in_channels=8,
        target_shape=(8, 420, 114),
        weight_relpath="backbones/wifi/resnet18.pth",
        path_columns=(
            "node_1_wifi_data_path",
            "node_2_wifi_data_path",
            "node_3_wifi_data_path",
            "node_4_wifi_data_path",
        ),
    ),
    "tof": ModalitySpec(
        name="tof",
        octonet_name="ToF",
        nodes=(4,),
        in_channels=1080,
        target_shape=(1080, 8, 8),
        weight_relpath="backbones/tof/resnet18.pth",
        path_columns=("node_4_ToF_data_path",),
    ),
    "mmwave": ModalitySpec(
        name="mmwave",
        octonet_name="mmWave",
        nodes=(1, 2, 3, 4, 5),
        in_channels=20,
        target_shape=(20, 54, 150),
        weight_relpath="backbones/mmwave/resnet18.pth",
        path_columns=(
            "node_1_mmWave_data_path",
            "node_2_mmWave_data_path",
            "node_3_mmWave_data_path",
            "node_4_mmWave_data_path",
            "node_5_mmWave_data_path",
        ),
    ),
}

DEFAULT_MODALITIES = ("imu", "uwb", "wifi", "tof", "mmwave")


def normalize_modalities(modalities: Iterable[str] | None) -> tuple[str, ...]:
    if modalities is None:
        return DEFAULT_MODALITIES

    normalized: list[str] = []
    for modality in modalities:
        key = modality.lower()
        if key not in MODALITY_SPECS:
            raise ValueError(f"Unsupported modality: {modality}")
        normalized.append(key)
    return tuple(normalized)


def get_backbone_weight_path(modality: str) -> Path:
    return PROJECT_ROOT / MODALITY_SPECS[modality].weight_relpath


def zero_tensor_for_modality(modality: str) -> torch.Tensor:
    spec = MODALITY_SPECS[modality]
    return torch.zeros(spec.target_shape, dtype=torch.float32)


def collate_fn(batch: Sequence[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    modalities = normalize_modalities(batch[0].get("modality_data", {}).keys())
    user_ids = [sample["user_id"] for sample in batch]
    activities = [sample["activity"] for sample in batch]
    recording_ids = [sample.get("recording_id") for sample in batch]
    labels = torch.tensor(
        [sample.get("label", ACTIVITY_TO_INDEX[sample["activity"]]) for sample in batch],
        dtype=torch.long,
    )

    modality_data: dict[str, torch.Tensor] = {}
    for modality in modalities:
        stacked = []
        for sample in batch:
            tensor = sample["modality_data"].get(modality)
            if tensor is None:
                tensor = zero_tensor_for_modality(modality)
            stacked.append(tensor.float())
        modality_data[modality] = torch.stack(stacked, dim=0)

    return {
        "user_id": user_ids,
        "recording_id": recording_ids,
        "activity": activities,
        "label": labels,
        "labels": labels,
        "modality_data": modality_data,
    }


collate_fn_octonet = collate_fn


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)(logits, labels)


def alignment_mse_loss(
    proxies: Mapping[str, torch.Tensor],
    real_globals: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    reference = next(iter(real_globals.values()), None)
    if reference is None:
        reference = next(iter(proxies.values()), None)
    device = reference.device if reference is not None else torch.device("cpu")

    total = torch.zeros((), device=device)
    count = 0
    for key, proxy_tensor in proxies.items():
        _, _, target_modality = key.partition("_to_")
        if target_modality not in real_globals:
            continue
        total = total + F.mse_loss(proxy_tensor, real_globals[target_modality])
        count += 1

    if count == 0:
        return total
    return total / count


def vicreg_loss(
    feature_dict: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
    *,
    inv_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1.0e-4,
) -> torch.Tensor:
    if isinstance(feature_dict, Mapping):
        features = list(feature_dict.values())
    else:
        features = list(feature_dict)

    if len(features) < 2:
        device = features[0].device if features else torch.device("cpu")
        return torch.zeros((), device=device)

    flattened: list[torch.Tensor] = []
    for tensor in features:
        if tensor.dim() == 3 and tensor.size(1) == 1:
            tensor = tensor.squeeze(1)
        elif tensor.dim() > 2:
            tensor = tensor.reshape(tensor.size(0), -1)
        flattened.append(tensor)

    if any(tensor.size(0) < 2 for tensor in flattened):
        return torch.zeros((), device=flattened[0].device)

    total = torch.zeros((), device=flattened[0].device)
    pair_count = 0
    for left_idx in range(len(flattened)):
        for right_idx in range(left_idx + 1, len(flattened)):
            left = flattened[left_idx]
            right = flattened[right_idx]

            invariance = F.mse_loss(left, right)

            left_std = torch.sqrt(left.var(dim=0, unbiased=False) + eps)
            right_std = torch.sqrt(right.var(dim=0, unbiased=False) + eps)
            variance = (F.relu(1.0 - left_std).mean() + F.relu(1.0 - right_std).mean()) / 2.0

            feature_dim = left.size(1)
            left_centered = left - left.mean(dim=0)
            right_centered = right - right.mean(dim=0)
            left_cov = (left_centered.T @ left_centered) / max(1, left.size(0) - 1)
            right_cov = (right_centered.T @ right_centered) / max(1, right.size(0) - 1)
            off_diagonal = ~torch.eye(feature_dim, dtype=torch.bool, device=left.device)
            covariance = (
                left_cov[off_diagonal].pow(2).sum() / feature_dim
                + right_cov[off_diagonal].pow(2).sum() / feature_dim
            ) / 2.0

            total = total + inv_coeff * invariance + var_coeff * variance + cov_coeff * covariance
            pair_count += 1

    return total / max(pair_count, 1)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.argmax(logits, dim=1)
    return float((predictions == labels).float().mean().item())


__all__ = [
    "ACTIVITY_LIST",
    "ACTIVITY_TO_INDEX",
    "DEFAULT_MODALITIES",
    "DEFAULT_USER_LIST",
    "INDEX_TO_ACTIVITY",
    "MODALITY_SPECS",
    "ModalitySpec",
    "accuracy",
    "alignment_mse_loss",
    "classification_loss",
    "collate_fn",
    "collate_fn_octonet",
    "get_backbone_weight_path",
    "normalize_modalities",
    "vicreg_loss",
    "zero_tensor_for_modality",
]
