from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import Dataset, Subset


def _compute_split_sizes(total: int, split_ratio: Sequence[float]) -> list[int]:
    if len(split_ratio) not in {2, 3}:
        raise ValueError("data_split must contain 2 or 3 ratios.")
    if any(ratio < 0 for ratio in split_ratio):
        raise ValueError("data_split cannot contain negative ratios.")

    ratio_sum = float(sum(split_ratio))
    if ratio_sum <= 0:
        raise ValueError("data_split must sum to a positive value.")

    normalized = [ratio / ratio_sum for ratio in split_ratio]
    raw_sizes = [ratio * total for ratio in normalized]
    split_sizes = [int(size) for size in raw_sizes]

    remainder = total - sum(split_sizes)
    fractional_order = sorted(
        range(len(raw_sizes)),
        key=lambda idx: raw_sizes[idx] - split_sizes[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        split_sizes[idx] += 1
    return split_sizes


def _get_dataset_recording_ids(dataset: Dataset) -> list[str]:
    if isinstance(dataset, Subset):
        parent_ids = _get_dataset_recording_ids(dataset.dataset)
        return [parent_ids[idx] for idx in dataset.indices]

    if hasattr(dataset, "sample_recording_ids"):
        return list(dataset.sample_recording_ids)
    if hasattr(dataset, "recording_ids"):
        return list(dataset.recording_ids)

    recording_ids = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        recording_id = sample.get("recording_id")
        if recording_id is None:
            recording_id = f"sample_{idx}"
        recording_ids.append(str(recording_id))
    return recording_ids


def split_dataset(
    dataset: Dataset,
    split_ratio: Sequence[float],
    *,
    seed: int = 41,
) -> tuple[Subset, ...]:
    total = len(dataset)
    if total == 0:
        return tuple(Subset(dataset, []) for _ in split_ratio)

    split_sizes = _compute_split_sizes(total, split_ratio)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()

    subsets = []
    start = 0
    for size in split_sizes:
        stop = start + size
        subsets.append(Subset(dataset, permutation[start:stop]))
        start = stop
    return tuple(subsets)


def split_dataset_by_recording(
    dataset: Dataset,
    split_ratio: Sequence[float],
    *,
    seed: int = 41,
) -> tuple[Subset, ...]:
    total = len(dataset)
    if total == 0:
        return tuple(Subset(dataset, []) for _ in split_ratio)

    recording_ids = _get_dataset_recording_ids(dataset)
    if len(recording_ids) != total:
        raise RuntimeError(
            "recording_id count does not match dataset length: "
            f"{len(recording_ids)} != {total}"
        )

    grouped_indices: dict[str, list[int]] = {}
    ordered_recordings: list[str] = []
    for idx, recording_id in enumerate(recording_ids):
        if recording_id not in grouped_indices:
            grouped_indices[recording_id] = []
            ordered_recordings.append(recording_id)
        grouped_indices[recording_id].append(idx)

    split_sizes = _compute_split_sizes(len(ordered_recordings), split_ratio)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(ordered_recordings), generator=generator).tolist()

    subsets = []
    start = 0
    for size in split_sizes:
        stop = start + size
        subset_indices: list[int] = []
        for recording_idx in permutation[start:stop]:
            subset_indices.extend(grouped_indices[ordered_recordings[recording_idx]])
        subset_indices.sort()
        subsets.append(Subset(dataset, subset_indices))
        start = stop
    return tuple(subsets)


__all__ = ["split_dataset", "split_dataset_by_recording"]
