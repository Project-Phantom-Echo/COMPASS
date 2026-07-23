import torch
import torch.nn as nn
from typing import Dict, Optional

from Encoders import EncoderToDict
from shared.utils.proxy_routing import should_generate_proxy


class classification_Head(nn.Sequential):
    def __init__(self, emb_size=512, num_classes=27, dropout=0.3): # 增加 dropout 参数
        super(classification_Head, self).__init__()
        self.norm = nn.LayerNorm(emb_size)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_size, num_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class CrossModalProxyGenerator(nn.Module):
    def __init__(self, input_dim=512, n_heads=4, dropout=0.1):
        super().__init__()
        self.proxy_token = nn.Parameter(torch.randn(1, 1, input_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, source_feats: torch.Tensor) -> torch.Tensor:
        b = source_feats.shape[0]
        proxy = self.proxy_token.expand(b, -1, -1)          # [B,1,D]
        tokens = torch.cat([proxy, source_feats], dim=1)    # [B,1+L,D]
        out = self.adapter(tokens)                          # [B,1+L,D]
        return out[:, 0, :].unsqueeze(1)                    # [B,1,D]


class ModalityImputer(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=None, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class MMFi_CMPT_Net(nn.Module):
    def __init__(
        self,
        num_classes=27,
        modalities=None,
        embed_dim=512,
        dropout=0.1,
        n_heads=4,
        missing_fill="proxy",
        impute_hidden_dim=None,
        impute_dropout=0.3,
        generator_mode="pairwise",
        source_priority=None,
        mask_sources_during_training=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # encoder：输出 dict(mod -> [B,32,512])
        self.encoders = EncoderToDict()

        if modalities is None:
            modalities = ["rgb", "depth", "mmwave", "lidar"]
        self.modalities = list(modalities)
        if missing_fill not in {"proxy", "impute"}:
            raise ValueError(f"Unsupported missing_fill: {missing_fill}")
        if generator_mode not in {"pairwise", "single_source"}:
            raise ValueError(f"Unsupported generator_mode: {generator_mode}")
        self.missing_fill = missing_fill
        self.generator_mode = generator_mode
        self.mask_sources_during_training = mask_sources_during_training
        if source_priority is None:
            self.source_priority = list(self.modalities)
        else:
            if isinstance(source_priority, str):
                source_priority = [s.strip() for s in source_priority.split(",") if s.strip()]
            priority = [m for m in source_priority if m in self.modalities]
            priority += [m for m in self.modalities if m not in priority]
            self.source_priority = priority

        if self.missing_fill == "impute":
            self.imputers = nn.ModuleDict({
                modality: ModalityImputer(
                    embed_dim=self.embed_dim,
                    hidden_dim=impute_hidden_dim,
                    dropout=impute_dropout,
                )
                for modality in self.modalities
            })
        else:
            self.generators = nn.ModuleDict()
            for src in self.modalities:
                for tgt in self.modalities:
                    if src == tgt:
                        continue
                    self.generators[f"{src}_to_{tgt}"] = CrossModalProxyGenerator(
                        input_dim=self.embed_dim,
                        n_heads=n_heads,
                        dropout=dropout
                    )

        self.decoder = classification_Head(emb_size=self.embed_dim, num_classes=num_classes)

    def _split_available(self, inputs, missing_mask):
        select_inputs = {k: v for k, v in inputs.items() if k in self.modalities}
        if missing_mask is None:
            return select_inputs, select_inputs
        available_inputs = {
            k: v for k, v in select_inputs.items()
            if missing_mask.get(k, 1) == 1
        }
        return select_inputs, available_inputs

    def _encode_for_fill(self, inputs, missing_mask):
        select_inputs, available_inputs = self._split_available(inputs, missing_mask)
        if len(available_inputs) == 0:
            raise ValueError(f"No available modality in inputs. Need at least one of {self.modalities}")

        if self.training:
            all_feats = self.encoders(select_inputs)
            if self.mask_sources_during_training:
                available_feats = {m: all_feats[m] for m in available_inputs if m in all_feats}
            else:
                available_feats = {m: all_feats[m] for m in self.modalities if m in all_feats}
            real_globals = {m: feat.mean(dim=1, keepdim=True) for m, feat in all_feats.items()}
        else:
            available_feats = self.encoders(available_inputs)
            real_globals = {m: feat.mean(dim=1, keepdim=True) for m, feat in available_feats.items()}

        if len(available_feats) == 0:
            raise ValueError(f"No encoded available modality. Need at least one of {self.modalities}")
        return available_feats, real_globals

    def _forward_impute(self, raw_features_available, real_globals, batch_size, device):
        observed_globals = [
            real_globals[m] for m in self.modalities
            if m in raw_features_available and m in real_globals
        ]
        if observed_globals:
            context = torch.stack(observed_globals, dim=0).mean(dim=0).squeeze(1)
        else:
            context = torch.zeros(batch_size, self.embed_dim, device=device)

        imputed = {}
        final_modality_feats = []
        for tgt in self.modalities:
            if tgt in raw_features_available and tgt in real_globals:
                final_modality_feats.append(real_globals[tgt])
            else:
                pred = self.imputers[tgt](context).unsqueeze(1)
                imputed[tgt] = pred
                final_modality_feats.append(pred)
        return final_modality_feats, imputed

    def forward(self, inputs: Dict[str, torch.Tensor], missing_mask: Optional[Dict[str, int]] = None):
        raw_features, real_globals = self._encode_for_fill(inputs, missing_mask)

        b = next(iter(raw_features.values())).shape[0]
        device = next(iter(raw_features.values())).device

        if self.missing_fill == "impute":
            final_modality_feats, imputed = self._forward_impute(
                raw_features_available=raw_features,
                real_globals=real_globals,
                batch_size=b,
                device=device,
            )
            fused = torch.stack(final_modality_feats, dim=0).sum(dim=0)
            logits = self.decoder(fused)
            return logits, imputed, real_globals

        # proxies: [B,1,512]
        proxies = {}
        for src in self.modalities:
            if src in raw_features:
                for tgt in self.modalities:
                    if src == tgt:
                        continue
                    if not should_generate_proxy(
                        training=self.training,
                        target=tgt,
                        real_features=real_globals,
                        missing_mask=missing_mask,
                    ):
                        continue
                    k = f"{src}_to_{tgt}"
                    proxies[k] = self.generators[k](raw_features[src])

        # build final_modality_feats: list of [B,1,512]
        final_modality_feats = []
        for tgt in self.modalities:
            if missing_mask is not None:
                is_available = (missing_mask.get(tgt, 1) == 1)
            else:
                is_available = (tgt in raw_features)

            if is_available and (tgt in real_globals):
                final_modality_feats.append(real_globals[tgt])
            else:
                if self.generator_mode == "single_source":
                    chosen = next(
                        (src for src in self.source_priority if src != tgt and f"{src}_to_{tgt}" in proxies),
                        None,
                    )
                    if chosen is not None:
                        final_modality_feats.append(proxies[f"{chosen}_to_{tgt}"])
                    else:
                        final_modality_feats.append(torch.zeros(b, 1, self.embed_dim, device=device))
                else:
                    tgt_proxies = []
                    for src in self.modalities:
                        if src == tgt:
                            continue
                        k = f"{src}_to_{tgt}"
                        if k in proxies:
                            tgt_proxies.append(proxies[k])

                    if len(tgt_proxies) > 0:
                        final_modality_feats.append(torch.stack(tgt_proxies, dim=0).mean(dim=0))  # [B,1,512]
                    else:
                        final_modality_feats.append(torch.zeros(b, 1, self.embed_dim, device=device))

        fused = torch.stack(final_modality_feats, dim=0).sum(dim=0)   # [B,1,512]

        # ✅ classification_Head 期待 [B,L,512]，这里 L=1 正好
        logits = self.decoder(fused)  # [B,num_classes]
        return logits, proxies, real_globals
