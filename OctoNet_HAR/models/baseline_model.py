from __future__ import annotations

from typing import Iterable, Mapping

import torch
from torch import nn

from util import ACTIVITY_LIST, normalize_modalities


class ConcatClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(x)))


class OctoNetConcatBaseline(nn.Module):
    def __init__(
        self,
        *,
        modalities: Iterable[str] | None = None,
        embed_dim: int = 512,
        num_classes: int = len(ACTIVITY_LIST),
        dropout: float = 0.3,
        load_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.modalities = normalize_modalities(modalities)
        self.embed_dim = embed_dim

        if load_backbones:
            from Encoders import EncoderToDict

            self.encoders = EncoderToDict(self.modalities, freeze_backbones=True)
        self.classifier = ConcatClassificationHead(len(self.modalities) * embed_dim, num_classes, dropout=dropout)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        missing_mask: Mapping[str, int] | None = None,
    ) -> torch.Tensor:
        if not inputs:
            raise ValueError("inputs must contain at least one modality tensor.")
        if not hasattr(self, "encoders"):
            raise RuntimeError("Raw-input forward requires load_backbones=True.")

        encoded = self.encoders(inputs, return_tokens=False, return_global=True)
        reference_tensor = next(iter(inputs.values()))
        batch_size = reference_tensor.size(0)
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        features = []
        for modality in self.modalities:
            use_real = modality in encoded and (missing_mask is None or missing_mask.get(modality, 1) == 1)
            if use_real:
                features.append(encoded[modality]["global"])
            else:
                features.append(torch.zeros(batch_size, self.embed_dim, device=device, dtype=dtype))

        fused = torch.cat(features, dim=-1)
        return self.classifier(fused)


__all__ = ["OctoNetConcatBaseline"]
