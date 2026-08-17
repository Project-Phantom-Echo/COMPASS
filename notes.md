# COMPASS

## Paper

- **Title:** COMPASS: Complete Multimodal Fusion via Proxy Tokens and Shared Spaces for Ubiquitous Sensing
- **Authors:** Hao Wang, Yanyu Qian, Pengcheng Weng, Zixuan Xia, William Dan, Yangxin Xu, Fei Wang
- **Published in:** arXiv preprint, version 2 (5 June 2026)
- **Venue metrics:** Journal impact factor/quartile not applicable
- **Link:** [arXiv](https://arxiv.org/abs/2604.02056)

## Method

- Maintain a fixed slot per modality: real features for observed sensors and
  generated proxy tokens for missing sensors.
- Project each modality to 32 tokens × 512 dimensions.
- For every directed source→target pair, a one-layer transformer predicts the
  missing slot; average predictions from multiple observed sources.
- Sum real/proxy slots, normalize, and classify.
- Train with synthetic modality masks, task loss, proxy alignment,
  VICReg-style shared-space stabilization, and per-proxy classification.

## Architecture

Read off `XRF55_HAR/models/cmpt_model_xrf55.py`, `XRF55_HAR/Encoders.py`, and
`XRF55_HAR/train.py`. Class: `XRF55_CMPT_Net`, `proj_dim=32`, `embed_dim=512`.

### Per-modality encoding

Each modality has a pretrained ResNet-18 feature extractor loaded from
`backbone_models/{mmWave,WIFI,RFID}/*.pt`, pretrained by X-Fi. They are **fine-tuned end to end** by
default (`freeze_backbone=False`); `freeze_backbone=True` is the lower-capacity
CMPT-style baseline.

The extractors are constructed with `.eval()` and a comment claiming BN is kept
in eval mode for stability, but this is a **no-op**: `train.py` calls
`model.train()` at the top of every epoch, which recurses into all submodules
and puts those BatchNorm layers back into training mode. Training therefore uses
batch statistics and updates running statistics as normal. The default path is
unaffected, but `freeze_backbone=True` only freezes *weights* — BN statistics
still adapt, so that row is not a cleanly frozen backbone.

Backbones emit 512 channels over a modality-specific length (`encode_info`),
which `LinearProjection` maps to a common token grid:

| Modality | Backbone output | Projection | Tokens |
|---|---|---|---|
| mmWave | `(B, 512, 32)` | `Conv1d(512→512, k=1)` → BN → ReLU → `Linear(32→32)` → ReLU | `(B, 32, 512)` |
| Wi-Fi | `(B, 512, 4)` | same, `Linear(4→32)` | `(B, 32, 512)` |
| RFID | `(B, 512, 5)` | same, `Linear(5→32)` | `(B, 32, 512)` |

Note that Wi-Fi and RFID are *upsampled* into 32 token positions from only 4 and
5 backbone positions, so their token axis is a learned expansion, not genuine
temporal resolution.

### The slot that actually gets fused

Each modality's 32 tokens are then **mean-pooled to a single `(B, 1, 512)`
vector** (`feat.mean(dim=1, keepdim=True)`). This is the "slot": fusion, proxy
targets and the VICReg term all operate on this one pooled vector per modality,
not on the 32 tokens. The 32-token sequence survives only as *input* to the
proxy generators.

### Proxy generation

`CrossModalProxyGenerator`, one per directed pair (`pairwise` →
`N(N−1) = 6` modules keyed `{src}_to_{tgt}`):

- a learned `proxy_token` of shape `(1, 1, 512)` is prepended to the source's
  full 32-token sequence, giving 33 tokens
- one `nn.TransformerEncoderLayer`: `d_model=512`, `nhead=4`,
  `dim_feedforward=1024` (`= 2 × embed_dim`), `dropout=0.3` (from
  `cmpt_dropout`, **not** the class default of `0.1`), `batch_first=True`,
  post-norm; `num_layers=1`
- output position 0 is taken as the predicted slot → `(B, 1, 512)`

Scalability and ablation variants: `shared` (one generator for all pairs, O(1),
via per-source tag + per-target query embeddings), `target_shared` (one per
target, O(N)), `generator_type="linear"` (mean-pool → `Linear→ReLU→Linear`, the
anti-cloning control), and `missing_fill="impute"` (per-target MLP
`512→hidden→512` on the mean of observed globals, replacing proxies entirely).

### Slot assembly and fusion

For each modality: observed → its real pooled global; missing → aggregate of the
proxies produced by every observed source. Aggregation is a plain mean
(`proxy_agg="uniform"`), or confidence-weighted — softmax over each proxy's
max decoder softmax, temperature `1.0`, computed under `no_grad`. With no source
available the slot is zeros. `generator_mode="single_source"` instead fills from
one observed source by fixed priority (default order `mmwave, wifi, rfid`,
i.e. descending unimodal strength).

Fusion is `sum` by default: stack the three slots, sum, then `LayerNorm(512)`.
Alternatives are `concat` (`Linear(3×512→512)` + LayerNorm) and `cross_attn`
(learned query + `MultiheadAttention(512, 8 heads)` + LayerNorm).

The head is a single `Linear(512 → 55)` — `Decoder` builds nothing deeper.

### Training / inference asymmetry

In `training` mode **all** modalities are encoded, including masked-out ones,
because their real globals are needed as proxy-alignment targets. At inference
only available modalities are encoded, so no information leaks from a missing
sensor. This is a deliberate asymmetry and matches X-Fi's evaluation.

### Losses

| Term | Weight (config) | Definition |
|---|---|---|
| Task | `1.0` | `CrossEntropyLoss(label_smoothing=0.1)` on fused logits |
| Proxy alignment | `cmpt_loss_weight = 0.2` | MSE between each proxy and the real pooled global of its target, averaged over generated pairs |
| VICReg | `lambda_vicreg = 0.3` | pairwise over **real** modality globals only, with `inv=5.0`, `var=25.0`, `cov=1.0` |
| Proxy classification | `lambda_proxy_cls = 0.5` | cross-entropy on `decoder(proxy)` for each proxy, averaged |
| Reconstruction | `lambda_recon = 0.5` | impute variant only; MSE on masked modalities, replaces the proxy path |

VICReg's three parts: invariance `MSE(zi, zj)`; variance `relu(1 − std)` per
dimension, forcing every dim to keep `std ≥ 1` across the batch; covariance,
the squared off-diagonal feature covariance normalised by `d`. Variance and
covariance are what stop the shared space collapsing to a constant, which
alignment alone would reward. Note `inv` is set to `5.0` against the VICReg
paper default of `25.0`, so anti-collapse pressure outweighs pulling-together
pressure by 5×. A second, unused implementation exists in
`OctoNet_HAR/util.py:280` — it defaults `inv=25.0` and computes variance with
`unbiased=False`, where the XRF55 inline path uses the Bessel-corrected default,
so the two are not numerically identical.

### Modality masking

`generate_random_mask` with `drop_prob=0.7`: about 30% of steps are fully
observed, and the remaining 70% keep `1` or `2` of the 3 modalities, drawn
uniformly, with the kept subset sampled uniformly.

### Optimization

`AdamW`, lr `1e-4`, weight decay `5e-2`, `warmuppolylr` schedule with
`power=0.9`, 5 warm-up epochs at ratio `0.1`, batch size `32`, 100 epochs,
seed `0`, `class_num=55`.

## XRF55 trial split results

**Part 1, all four scenes:** repetitions 1–14 train (`15,400`) and 15–20
  test (`6,600`).
| Modalities | X-Fi | COMPASS |
|---|---:|---:|
| R | 83.9 | 90.5 |
| W | 55.7 | 85.4 |
| RF | 42.5 | 48.4 |
| R + W | 88.2 | 95.1 |
| R + RF | 86.5 | 91.2 |
| W + RF | 58.1 | 86.3 |
| All | 89.8 | 95.1 |

The seven COMPASS values displayed above average to `84.57%` (reported as
`84.6%` in the paper). Separately, across three seeded runs, the paper reports
a seven-scenario average of `84.3 ± 0.15%`, versus 72.1% for X-Fi. The
displayed per-combination values are therefore not the per-cell three-seed
means.

## Notes

- Pairwise proxy generators require `N(N−1)` modules and fully observed
  training samples.
- Shared-space stabilization is the strongest auxiliary loss, so gains are
  not solely from proxy completion.
- COMPASS reports controlled PTA at `73.5 ± 1.6%`; PTA's own 84.83% average
  uses the easier Scene-1-only protocol.
