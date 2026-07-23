from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset, Subset

from dataset.splits import split_dataset, split_dataset_by_recording
from util import DEFAULT_MODALITIES, normalize_modalities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES_PATH = PROJECT_ROOT / "features" / "features.pt"


def resolve_features_path(feature_path: str | Path) -> Path:
    resolved_path = Path(feature_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return resolved_path.resolve()


def load_extracted_features(feature_path: str | Path) -> list[dict]:
    resolved_path = resolve_features_path(feature_path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Could not find extracted features at {resolved_path}. "
            "Download features.pt or run extract_features.py first."
        )

    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    # Support both the metadata payload and the earlier plain-list format.
    if isinstance(payload, dict) and "records" in payload:
        meta = payload.get("metadata", {})
        print(
            f"Feature metadata: {meta.get('num_samples')} samples, "
            f"modalities={meta.get('modalities')}, "
            f"users={meta.get('user_list')}"
        )
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise TypeError(f"Expected extracted features to be a list or dict, got {type(payload)!r}")
    return records


class ExtractedFeatureDataset(Dataset):
    def __init__(
        self,
        feature_path: str | Path = DEFAULT_FEATURES_PATH,
        *,
        modalities: Sequence[str] | None = None,
    ) -> None:
        self.feature_path = str(resolve_features_path(feature_path))
        self.records = load_extracted_features(self.feature_path)
        self.modalities = normalize_modalities(modalities)
        self.sample_recording_ids = tuple(self._build_recording_ids())

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        globals_dict = {
            modality: record.get("globals", {}).get(modality, torch.zeros(512, dtype=torch.float32)).float()
            for modality in self.modalities
        }
        tokens_dict = {
            modality: record.get("tokens", {}).get(modality, torch.zeros(0, 512, dtype=torch.float32)).float()
            for modality in self.modalities
        }
        label = int(record["label"])

        return {
            "user_id": str(record.get("user_id", "")),
            "recording_id": self.sample_recording_ids[index],
            "label": label,
            "labels": label,
            "globals": globals_dict,
            "tokens": tokens_dict,
        }

    def _build_recording_ids(self) -> list[str]:
        recording_ids = []
        missing_recording_id = False
        for idx, record in enumerate(self.records):
            recording_id = record.get("recording_id")
            if recording_id is None:
                missing_recording_id = True
                recording_id = f"legacy_sample_{idx}"
            recording_ids.append(str(recording_id))

        if missing_recording_id:
            print(
                "Warning: extracted features do not contain recording_id. "
                "Falling back to per-sample ids; rerun extract_features.py "
                "to enable exact recording-level splits."
            )
        return recording_ids


def _pad_tokens(token_tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not token_tensors:
        raise ValueError("Cannot pad an empty token tensor sequence.")

    max_tokens = max(int(tensor.size(0)) for tensor in token_tensors)
    feature_dim = int(token_tensors[0].size(-1)) if token_tensors[0].ndim == 2 else 512

    padded = torch.zeros(len(token_tensors), max_tokens, feature_dim, dtype=torch.float32)
    lengths = torch.zeros(len(token_tensors), dtype=torch.long)
    for idx, tensor in enumerate(token_tensors):
        tensor = tensor.float()
        current_length = int(tensor.size(0))
        lengths[idx] = current_length
        if current_length > 0:
            padded[idx, :current_length] = tensor
    return padded, lengths


def extracted_feature_collate_fn(batch: Sequence[dict]) -> dict:
    if not batch:
        raise ValueError("Cannot collate an empty extracted-feature batch.")

    user_ids = [sample.get("user_id", "") for sample in batch]
    recording_ids = [sample.get("recording_id") for sample in batch]
    labels = torch.tensor([int(sample["label"]) for sample in batch], dtype=torch.long)
    modalities = normalize_modalities(batch[0].get("globals", {}).keys() or DEFAULT_MODALITIES)

    global_features: dict[str, torch.Tensor] = {}
    token_features: dict[str, torch.Tensor] = {}
    token_lengths: dict[str, torch.Tensor] = {}
    for modality in modalities:
        global_features[modality] = torch.stack(
            [sample["globals"][modality].float() for sample in batch],
            dim=0,
        )
        padded_tokens, lengths = _pad_tokens([sample["tokens"][modality] for sample in batch])
        token_features[modality] = padded_tokens
        token_lengths[modality] = lengths

    return {
        "user_id": user_ids,
        "recording_id": recording_ids,
        "label": labels,
        "labels": labels,
        "features": {
            "globals": global_features,
            "tokens": token_features,
            "token_lengths": token_lengths,
        },
    }


def make_extracted_feature_dataset(
    config: Mapping[str, object],
    feature_path: str | Path | None = None,
) -> Dataset | tuple[Subset, ...]:
    dataset = ExtractedFeatureDataset(
        feature_path=feature_path or config.get("features_path", DEFAULT_FEATURES_PATH),
        modalities=config.get("modality"),
    )

    split_ratio = config.get("data_split")
    if split_ratio is None:
        return dataset

    splitter = split_dataset_by_recording if bool(config.get("recording_level_split", True)) else split_dataset
    return splitter(dataset, split_ratio=split_ratio, seed=int(config.get("init_rand_seed", 41)))


__all__ = [
    "DEFAULT_FEATURES_PATH",
    "ExtractedFeatureDataset",
    "extracted_feature_collate_fn",
    "load_extracted_features",
    "make_extracted_feature_dataset",
    "resolve_features_path",
]
