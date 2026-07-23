"""Routing rules shared by COMPASS inference implementations."""

from __future__ import annotations

from collections.abc import Mapping


def should_generate_proxy(
    *,
    training: bool,
    target: str,
    real_features: Mapping[str, object],
    missing_mask: Mapping[str, int] | None,
) -> bool:
    """Keep all training proxies, but skip unused observed-target proxies at inference."""
    if training:
        return True
    target_is_present = target in real_features and (
        missing_mask is None or missing_mask.get(target, 1) == 1
    )
    return not target_is_present
