from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from util import (
    ACTIVITY_LIST,
    MODALITY_SPECS,
    get_backbone_weight_path,
    get_octonet_benchmark_root,
    normalize_modalities,
)


PROJECT_ROOT = Path(__file__).resolve().parent
OCTONET_BENCHMARK_ROOT = get_octonet_benchmark_root()


def _load_resnet18():
    module_path = OCTONET_BENCHMARK_ROOT / "Models" / "resnet.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find the OctoNet ResNet implementation at {module_path}")

    spec = importlib.util.spec_from_file_location("_octonet_resnet", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the OctoNet ResNet module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resnet18


try:
    resnet18 = _load_resnet18()
except (AttributeError, FileNotFoundError, ImportError) as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Failed to import OctoNet ResNet18. Set OCTONET_BENCHMARK_ROOT to "
        "the official OctonetBenchmark directory; see the data preparation "
        "section in the top-level README."
    ) from exc


def _load_state_dict(weight_path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(weight_path, map_location="cpu")

    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, Mapping):
        raise TypeError(f"Unexpected checkpoint format in {weight_path}")

    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    return state_dict


class OctoNetFeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class OctoNetModalityEncoder(nn.Module):
    def __init__(self, modality: str, *, freeze_backbone: bool = False) -> None:
        super().__init__()
        modality = normalize_modalities([modality])[0]
        self.modality = modality
        self.freeze_backbone = freeze_backbone
        spec = MODALITY_SPECS[modality]

        backbone = resnet18(num_classes=len(ACTIVITY_LIST), in_channels=spec.in_channels)
        state_dict = _load_state_dict(get_backbone_weight_path(modality))
        backbone.load_state_dict(state_dict, strict=True)

        self.extractor = OctoNetFeatureExtractor(backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if freeze_backbone:
            self.requires_grad_(False)
            self.extractor.eval()

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_tokens: bool = True,
        return_global: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not return_tokens and not return_global:
            raise ValueError("At least one of return_tokens or return_global must be True.")

        feature_map = self.extractor(x)
        outputs: dict[str, torch.Tensor] = {}

        if return_tokens:
            outputs["tokens"] = feature_map.flatten(2).transpose(1, 2).contiguous()
        if return_global:
            outputs["global"] = self.pool(feature_map).flatten(1)

        return outputs

    def train(self, mode: bool = True) -> "OctoNetModalityEncoder":
        super().train(mode)
        if self.freeze_backbone:
            self.extractor.eval()
        return self

    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_tokens=False, return_global=True)["global"]

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_tokens=True, return_global=False)["tokens"]


class EncoderToDict(nn.Module):
    def __init__(
        self,
        modalities: Iterable[str] | None = None,
        *,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.modalities = normalize_modalities(modalities)
        self.encoders = nn.ModuleDict(
            {
                modality: OctoNetModalityEncoder(modality, freeze_backbone=freeze_backbones)
                for modality in self.modalities
            }
        )

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        return_tokens: bool = True,
        return_global: bool = True,
    ) -> dict[str, dict[str, torch.Tensor]]:
        outputs: dict[str, dict[str, torch.Tensor]] = {}
        for modality in self.modalities:
            tensor = inputs.get(modality)
            if tensor is None:
                continue
            outputs[modality] = self.encoders[modality](
                tensor,
                return_tokens=return_tokens,
                return_global=return_global,
            )
        return outputs

    def global_dict(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {modality: output["global"] for modality, output in self.forward(inputs, return_tokens=False).items()}

    def token_dict(self, inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {modality: output["tokens"] for modality, output in self.forward(inputs, return_global=False).items()}


def adaptive_avg_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.dim() != 3:
        raise ValueError("Expected token tensor with shape [B, N, C].")
    return F.adaptive_avg_pool1d(tokens.transpose(1, 2), 1).squeeze(-1)


__all__ = [
    "EncoderToDict",
    "OctoNetFeatureExtractor",
    "OctoNetModalityEncoder",
    "adaptive_avg_pool_tokens",
]
