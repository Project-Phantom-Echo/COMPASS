import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from Encoders import Encoders, Decoder
from shared.utils.proxy_routing import should_generate_proxy


class CrossModalProxyGenerator(nn.Module):
    """
    CMPT 核心生成器 (Transformer 结构)
    """

    def __init__(self, input_dim=512, n_heads=4, dropout=0.1):
        super().__init__()
        self.proxy_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # dropout 参数会在这里传入 TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=n_heads,
            dim_feedforward=input_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, source_feats):
        """
        source_feats: [B, L, D] (必须是这个形状!)
        """
        batch_size = source_feats.shape[0]
        proxy = self.proxy_token.expand(batch_size, -1, -1)
        tokens = torch.cat([proxy, source_feats], dim=1)
        output = self.adapter(tokens)
        return output[:, 0, :].unsqueeze(1)


class LinearProxyGenerator(nn.Module):
    """Anti-cloning baseline: simple linear projection from source global to target slot."""

    def __init__(self, input_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, source_feats):
        """source_feats: [B, L, D]"""
        global_feat = source_feats.mean(dim=1)  # [B, D]
        return self.proj(global_feat).unsqueeze(1)  # [B, 1, D]


class ModalityImputer(nn.Module):
    """Per-target feature imputer: embed_dim -> hidden -> embed_dim."""

    def __init__(self, embed_dim=512, hidden_dim=None, dropout=0.3):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context):
        return self.net(context)


class SharedProxyGenerator(nn.Module):
    """P1-1 scalability: ONE shared generator for all (src->tgt) pairs (O(1)).
    Per-source embedding tags the input modality; per-target query selects the target.
    Returns [B,1,D] (XRF55 convention)."""

    def __init__(self, modalities, input_dim=512, n_heads=4, dropout=0.1):
        super().__init__()
        self.mod_index = {m: i for i, m in enumerate(modalities)}
        n = len(modalities)
        self.source_embed = nn.Parameter(torch.randn(n, 1, input_dim) * 0.02)
        self.target_query = nn.Parameter(torch.randn(n, 1, input_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, dim_feedforward=input_dim * 2,
            dropout=dropout, batch_first=True)
        self.adapter = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, source_feats, src, tgt):
        b = source_feats.shape[0]
        si, ti = self.mod_index[src], self.mod_index[tgt]
        query = self.target_query[ti:ti + 1].expand(b, -1, -1)
        tagged = source_feats + self.source_embed[si:si + 1]
        tokens = torch.cat([query, tagged], dim=1)
        out = self.adapter(tokens)
        return out[:, 0, :].unsqueeze(1)


class TargetSharedProxyGenerator(nn.Module):
    """P1-1 scalability: ONE generator per target modality (O(N)).
    Per-source embedding distinguishes the observed source feeding each target generator."""

    def __init__(self, modalities, input_dim=512, n_heads=4, dropout=0.1):
        super().__init__()
        self.mod_index = {m: i for i, m in enumerate(modalities)}
        n = len(modalities)
        self.source_embed = nn.Parameter(torch.randn(n, 1, input_dim) * 0.02)
        self.generators = nn.ModuleDict({
            m: CrossModalProxyGenerator(input_dim=input_dim, n_heads=n_heads, dropout=dropout)
            for m in modalities
        })

    def forward(self, source_feats, src, tgt):
        si = self.mod_index[src]
        tagged = source_feats + self.source_embed[si:si + 1]
        return self.generators[tgt](tagged)


class XRF55_CMPT_Net(nn.Module):
    def __init__(self, task_encoders: nn.ModuleDict, task_decoder: Tuple,
                 proj_dim=32, embed_dim=512, dropout=0.1,
                 fusion_type="sum", no_proxy=False,
                 proxy_agg="uniform", conf_temperature=1.0,
                 generator_type="transformer", generator_mode="pairwise",
                 source_priority=None, missing_fill="proxy",
                 impute_hidden_dim=None, impute_dropout=0.3):
        super().__init__()

        self.modalities = list(task_encoders.keys())
        self.embed_dim = embed_dim  # 记住 D
        self.fusion_type = fusion_type
        if missing_fill not in {"proxy", "impute"}:
            raise ValueError(f"Unsupported missing_fill: {missing_fill}")
        self.missing_fill = missing_fill
        self.no_proxy = no_proxy if self.missing_fill == "proxy" else False
        self.generator_mode = generator_mode
        # CMPT-style single-source fill uses a fixed source-selection priority (a permutation of modalities).
        # Default = modality order [mmwave, wifi, rfid], which is the descending unimodal-strength order;
        # for a missing target the highest-priority observed source is used.
        if source_priority is None:
            self.source_priority = list(self.modalities)
        else:
            if isinstance(source_priority, str):
                source_priority = [s.strip() for s in source_priority.split(",") if s.strip()]
            sp = [m for m in source_priority if m in self.modalities]
            sp += [m for m in self.modalities if m not in sp]
            self.source_priority = sp
        if proxy_agg not in {"uniform", "confidence"}:
            raise ValueError(f"Unsupported proxy_agg: {proxy_agg}")
        self.proxy_agg = proxy_agg
        self.conf_temperature = conf_temperature

        # 1) Encoders 输出应为 [B, proj_dim, embed_dim]
        self.encoders = Encoders(task_encoders, proj_dim=proj_dim)

        # 2) Missing-slot fillers.
        self.shared_generator = None
        if self.missing_fill == "impute":
            self.imputers = nn.ModuleDict({
                modality: ModalityImputer(
                    embed_dim=embed_dim,
                    hidden_dim=impute_hidden_dim,
                    dropout=impute_dropout,
                )
                for modality in self.modalities
            })
        elif not self.no_proxy:
            # Generators (P1-1: pairwise=N(N-1) 原版; shared=O(1); target_shared=O(N))
            if generator_mode in ("pairwise", "single_source"):
                self.generators = nn.ModuleDict()
                for src in self.modalities:
                    for tgt in self.modalities:
                        if src == tgt:
                            continue
                        if generator_type == "linear":
                            self.generators[f"{src}_to_{tgt}"] = LinearProxyGenerator(
                                input_dim=embed_dim)
                        else:
                            self.generators[f"{src}_to_{tgt}"] = CrossModalProxyGenerator(
                                input_dim=embed_dim, dropout=dropout)
            elif generator_mode == "shared":
                self.generators = nn.ModuleDict()
                self.shared_generator = SharedProxyGenerator(
                    self.modalities, input_dim=embed_dim, dropout=dropout)
            elif generator_mode == "target_shared":
                self.generators = nn.ModuleDict()
                self.shared_generator = TargetSharedProxyGenerator(
                    self.modalities, input_dim=embed_dim, dropout=dropout)
            else:
                raise ValueError(f"Unknown generator_mode: {generator_mode}")

        # 3) Decoder：输入维必须是 embed_dim
        self.decoder = Decoder(task_decoder[0], task_decoder[1])

        # 4) Fusion Norm：维度 embed_dim
        self.fusion_norm = nn.LayerNorm(embed_dim)
        if self.fusion_type == "concat":
            self.fusion_proj = nn.Sequential(
                nn.Linear(len(self.modalities) * embed_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            )
        elif self.fusion_type == "cross_attn":
            self.fusion_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(embed_dim)
        elif self.fusion_type != "sum":
            raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")

    def proxy_generator_fn(self, src, tgt):
        """Return callable(source_feats)->proxy[B,1,D] for (src,tgt), dispatching on mode."""
        if self.generator_mode in ("pairwise", "single_source"):
            return self.generators[f"{src}_to_{tgt}"]
        gen = self.shared_generator
        return lambda feats: gen(feats, src, tgt)

    def _forward_impute(self, raw_features_available, real_globals, batch_size, device):
        observed_globals = [real_globals[mod] for mod in self.modalities if mod in raw_features_available]
        if observed_globals:
            context = torch.stack(observed_globals, dim=0).mean(dim=0).squeeze(1)  # [B,D]
        else:
            context = torch.zeros(batch_size, self.embed_dim, device=device)

        imputed = {}
        final_modality_feats = []
        for tgt in self.modalities:
            if tgt in raw_features_available:
                final_modality_feats.append(real_globals[tgt])
            else:
                prediction = self.imputers[tgt](context).unsqueeze(1)  # [B,1,D]
                imputed[tgt] = prediction
                final_modality_feats.append(prediction)
        return final_modality_feats, imputed

    def forward(self, inputs: Dict, missing_mask: Dict[str, int] = None):

        select_inputs = {k: v for k, v in inputs.items() if k in self.encoders.encoders}

        batch_size = next(iter(select_inputs.values())).shape[0]
        device = next(iter(select_inputs.values())).device

        # 按 missing_mask 区分 available / missing 模态
        if missing_mask is not None:
            available_inputs = {k: v for k, v in select_inputs.items() if missing_mask.get(k, 1) == 1}
        else:
            available_inputs = select_inputs

        if self.training:
            # 训练时：编码所有模态（missing 的 real 特征作为对齐目标）
            raw_features_all = self.encoders(select_inputs)
            raw_features_available = {k: v for k, v in raw_features_all.items() if k in available_inputs}
            real_globals = {mod: feat.mean(dim=1, keepdim=True) for mod, feat in raw_features_all.items()}
        else:
            # 推理时：只编码 available 模态（与 X-Fi 一致，无信息泄漏）
            raw_features_available = self.encoders(available_inputs)
            real_globals = {mod: feat.mean(dim=1, keepdim=True) for mod, feat in raw_features_available.items()}

        if self.missing_fill == "impute":
            final_modality_feats, imputed = self._forward_impute(
                raw_features_available=raw_features_available,
                real_globals=real_globals,
                batch_size=batch_size,
                device=device,
            )
            modality_feats = [feat.squeeze(1) for feat in final_modality_feats]
            if self.fusion_type == "sum":
                fused_embedding = torch.stack(final_modality_feats).sum(dim=0).squeeze(1)  # [B,D]
                fused_embedding = self.fusion_norm(fused_embedding)
            elif self.fusion_type == "concat":
                fused_embedding = self.fusion_proj(torch.cat(modality_feats, dim=-1))
            else:
                kv = torch.stack(modality_feats, dim=1)
                query = self.fusion_query.expand(batch_size, -1, -1)
                attn_output, _ = self.fusion_attn(query, kv, kv)
                fused_embedding = self.fusion_norm(attn_output.squeeze(1))

            logits = self.decoder(fused_embedding)
            return logits, imputed, real_globals

        # proxies: 只从 available 模态的特征生成
        proxies = {}
        if not self.no_proxy:
            for src in self.modalities:
                if src in raw_features_available:
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
                        proxies[k] = self.proxy_generator_fn(src, tgt)(raw_features_available[src])

        # fuse per tgt -> list of [B,1,D]
        final_modality_feats = []
        for tgt in self.modalities:
            is_available = tgt in raw_features_available
            if missing_mask is not None:
                is_available = is_available and missing_mask.get(tgt, 1) == 1

            if is_available:
                final_modality_feats.append(real_globals[tgt])
            else:
                if self.no_proxy:
                    final_modality_feats.append(torch.zeros(batch_size, 1, self.embed_dim, device=device))
                    continue

                if self.generator_mode == "single_source":
                    # CMPT-style: fill the missing slot from ONE observed source (fixed priority order),
                    # instead of averaging over all observed sources.
                    chosen = next((src for src in self.source_priority
                                   if src != tgt and f"{src}_to_{tgt}" in proxies), None)
                    if chosen is not None:
                        final_modality_feats.append(proxies[f"{chosen}_to_{tgt}"])
                    else:
                        final_modality_feats.append(torch.zeros(batch_size, 1, self.embed_dim, device=device))
                    continue

                tgt_proxies = []
                for src in self.modalities:
                    if src == tgt:
                        continue
                    k = f"{src}_to_{tgt}"
                    if k in proxies:
                        tgt_proxies.append(proxies[k])

                if tgt_proxies:
                    if self.proxy_agg == "uniform" or len(tgt_proxies) == 1:
                        final_modality_feats.append(torch.stack(tgt_proxies).mean(dim=0))  # [B,1,D]
                    else:
                        proxy_stack = torch.stack(tgt_proxies, dim=0)  # [S,B,1,D]
                        with torch.no_grad():
                            conf_scores = []
                            for proxy in tgt_proxies:
                                logits = self.decoder(proxy.squeeze(1))
                                conf = F.softmax(logits, dim=-1).max(dim=-1).values
                                conf_scores.append(conf)
                            conf_scores = torch.stack(conf_scores, dim=0)  # [S,B]
                            weights = F.softmax(conf_scores / self.conf_temperature, dim=0)

                        final_modality_feats.append(
                            (proxy_stack * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
                        )
                else:
                    final_modality_feats.append(torch.zeros(batch_size, 1, self.embed_dim, device=device))

        modality_feats = [feat.squeeze(1) for feat in final_modality_feats]
        if self.fusion_type == "sum":
            fused_embedding = torch.stack(final_modality_feats).sum(dim=0).squeeze(1)  # [B,D]
            fused_embedding = self.fusion_norm(fused_embedding)
        elif self.fusion_type == "concat":
            fused_embedding = self.fusion_proj(torch.cat(modality_feats, dim=-1))
        else:
            kv = torch.stack(modality_feats, dim=1)
            query = self.fusion_query.expand(batch_size, -1, -1)
            attn_output, _ = self.fusion_attn(query, kv, kv)
            fused_embedding = self.fusion_norm(attn_output.squeeze(1))

        logits = self.decoder(fused_embedding)

        return logits, proxies, real_globals
