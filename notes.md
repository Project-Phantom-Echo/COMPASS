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

## XRF55 data

- **Part 1, all four scenes:** repetitions 1–14 train (`15,400`) and 15–20
  test (`6,600`).

## XRF55 results

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
