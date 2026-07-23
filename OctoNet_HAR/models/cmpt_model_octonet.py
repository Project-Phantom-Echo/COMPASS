from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from shared.utils.proxy_routing import should_generate_proxy
from util import ACTIVITY_LIST, normalize_modalities


class CrossModalProxyGenerator(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.proxy_token = nn.Parameter(torch.randn(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, source_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = source_tokens.size(0)
        proxy = self.proxy_token.expand(batch_size, -1, -1)
        tokens = torch.cat([proxy, source_tokens], dim=1)
        adapted = self.adapter(tokens)
        return adapted[:, 0, :]


class GlobalClassificationHead(nn.Module):
    def __init__(self, embed_dim: int = 512, num_classes: int = 62, dropout: float = 0.3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.norm(x)))


class OctoNetCMPTNet(nn.Module):
    def __init__(
        self,
        *,
        modalities: Iterable[str] | None = None,
        embed_dim: int = 512,
        num_classes: int = len(ACTIVITY_LIST),
        dropout: float = 0.1,
        head_dropout: float = 0.3,
        n_heads: int = 4,
        proxy_pairs: Sequence[tuple[str, str]] | None = None,
        load_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.modalities = normalize_modalities(modalities)
        self.embed_dim = embed_dim

        if load_backbones:
            from Encoders import EncoderToDict

            self.encoders = EncoderToDict(self.modalities, freeze_backbones=True)

        if proxy_pairs is None:
            proxy_pairs = [(src, tgt) for src in self.modalities for tgt in self.modalities if src != tgt]
        self.proxy_pairs = tuple(proxy_pairs)

        self.generators = nn.ModuleDict(
            {
                f"{src}_to_{tgt}": CrossModalProxyGenerator(
                    input_dim=embed_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for src, tgt in self.proxy_pairs
            }
        )
        self.classifier = GlobalClassificationHead(embed_dim, num_classes, dropout=head_dropout)

    def forward(
        self,
        inputs: Mapping[str, torch.Tensor],
        *,
        missing_mask: Mapping[str, int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if not inputs:
            raise ValueError("inputs must contain at least one modality tensor.")
        if not hasattr(self, "encoders"):
            raise RuntimeError("Raw-input forward requires load_backbones=True.")

        encoded = self.encoders(inputs, return_tokens=True, return_global=True)
        if not encoded:
            raise ValueError(f"No encoded modalities were produced for {self.modalities}.")

        reference_tensor = next(iter(inputs.values()))
        batch_size = reference_tensor.size(0)
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        real_tokens = {modality: outputs["tokens"] for modality, outputs in encoded.items()}
        real_globals = {modality: outputs["global"] for modality, outputs in encoded.items()}

        proxies: dict[str, torch.Tensor] = {}
        for src, tgt in self.proxy_pairs:
            source_is_present = missing_mask is None or missing_mask.get(src, 1) == 1
            if src not in real_tokens or not source_is_present:
                continue
            # Inference-only: skip proxies whose target is present -- its real token
            # is used by fusion, so this proxy would be generated then discarded.
            # This makes the active generator count |O|*|M| (zero at full observation),
            # matching the deployed cost. Training keeps ALL proxies because the
            # per-proxy discriminative loss supervises every src->tgt proxy.
            if not should_generate_proxy(
                training=self.training,
                target=tgt,
                real_features=real_globals,
                missing_mask=missing_mask,
            ):
                continue
            key = f"{src}_to_{tgt}"
            proxies[key] = self.generators[key](real_tokens[src])

        fused_features = []
        for target in self.modalities:
            use_real = target in real_globals and (missing_mask is None or missing_mask.get(target, 1) == 1)
            if use_real:
                fused_features.append(real_globals[target])
                continue

            candidate_proxies = [
                proxies[f"{src}_to_{target}"]
                for src in self.modalities
                if src != target and f"{src}_to_{target}" in proxies
            ]
            if candidate_proxies:
                fused_features.append(torch.stack(candidate_proxies, dim=0).mean(dim=0))
            else:
                fused_features.append(torch.zeros(batch_size, self.embed_dim, device=device, dtype=dtype))

        fused = torch.stack(fused_features, dim=0).sum(dim=0)
        logits = self.classifier(fused)
        return logits, proxies, real_globals


__all__ = ["CrossModalProxyGenerator", "OctoNetCMPTNet"]
