from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util import (  # noqa: E402
    ACTIVITY_LIST,
    ACTIVITY_TO_INDEX,
    DEFAULT_MODALITIES,
    DEFAULT_USER_LIST,
    MODALITY_SPECS,
    get_octonet_benchmark_root,
    normalize_modalities,
    zero_tensor_for_modality,
)
from dataset.splits import split_dataset, split_dataset_by_recording  # noqa: E402


DEFAULT_DATASET_PATH = str(PROJECT_ROOT.parent / "data" / "OctoNet" / "dataset")
OCTONET_BENCHMARK_ROOT = get_octonet_benchmark_root()

if str(OCTONET_BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(OCTONET_BENCHMARK_ROOT))

try:
    from octonet.Octonet import OctonetDataset, custom_collate
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "Failed to import OctoNet benchmark utilities. Set "
        "OCTONET_BENCHMARK_ROOT to the official OctonetBenchmark directory; "
        "see the data preparation section in the top-level README."
    ) from exc


def _resolve_data_path(dataset_root: Path, relative_or_absolute_path: str) -> str:
    candidate = Path(relative_or_absolute_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((dataset_root / candidate).resolve())


def _is_valid_file_or_dir(path: str, min_file_size_bytes: int = 100) -> bool:
    candidate = Path(path)
    if not candidate.exists():
        return False

    if candidate.is_file():
        return candidate.stat().st_size >= min_file_size_bytes

    if candidate.is_dir():
        total_size = 0
        for sub_path in candidate.rglob("*"):
            if sub_path.is_file():
                total_size += sub_path.stat().st_size
                if total_size >= min_file_size_bytes:
                    return True
        return total_size >= min_file_size_bytes

    return False


def _required_path_columns(modalities: Sequence[str]) -> tuple[str, ...]:
    columns: list[str] = []
    for modality in modalities:
        columns.extend(MODALITY_SPECS[modality].path_columns)
    return tuple(columns)


def build_filtered_metadata(
    *,
    dataset_path: str | Path,
    modalities: Iterable[str] | None = None,
    user_list: Sequence[int] | None = None,
    activity_list: Sequence[str] | None = None,
    min_file_size_bytes: int = 100,
) -> pd.DataFrame:
    dataset_root = Path(dataset_path).expanduser().resolve()
    metadata_path = dataset_root / "cut_manual.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Could not find OctoNet metadata at {metadata_path}")

    normalized_modalities = normalize_modalities(modalities)
    selected_users = tuple(user_list) if user_list is not None else DEFAULT_USER_LIST
    selected_activities = tuple(activity_list) if activity_list is not None else ACTIVITY_LIST

    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df[metadata_df["user_id"].isin(selected_users)]
    metadata_df = metadata_df[metadata_df["activity"].isin(selected_activities)]

    required_columns = ("user_id", "activity", "cut_timestamps", "recording_time") + _required_path_columns(
        normalized_modalities
    )
    missing_columns = [column for column in required_columns if column not in metadata_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in cut_manual.csv: {missing_columns}")

    valid_rows = []
    for _, row in metadata_df.iterrows():
        row_data = {
            "user_id": int(row["user_id"]),
            "activity": row["activity"],
            "cut_timestamps": row["cut_timestamps"],
            "recording_time": row["recording_time"],
        }

        row_valid = True
        for column in _required_path_columns(normalized_modalities):
            value = row[column]
            if pd.isna(value):
                row_valid = False
                break

            resolved_path = _resolve_data_path(dataset_root, str(value))
            if not _is_valid_file_or_dir(resolved_path, min_file_size_bytes=min_file_size_bytes):
                row_valid = False
                break

            row_data[column] = resolved_path

        if row_valid:
            valid_rows.append(row_data)

    output_columns = list(required_columns)
    if not valid_rows:
        return pd.DataFrame(columns=output_columns)

    filtered_df = pd.DataFrame(valid_rows)
    return filtered_df.loc[:, output_columns].reset_index(drop=True)


class OctoNetHARDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_path: str = DEFAULT_DATASET_PATH,
        modalities: Iterable[str] | None = None,
        user_list: Sequence[int] | None = None,
        activity_list: Sequence[str] | None = None,
        segmentation_flag: bool = True,
        min_file_size_bytes: int = 100,
    ) -> None:
        self.modalities = normalize_modalities(modalities)
        self.dataset_path = str(Path(dataset_path).expanduser().resolve())
        self.user_list = tuple(user_list) if user_list is not None else DEFAULT_USER_LIST
        self.activity_list = tuple(activity_list) if activity_list is not None else ACTIVITY_LIST
        self.segmentation_flag = bool(segmentation_flag)

        self.metadata_df = build_filtered_metadata(
            dataset_path=self.dataset_path,
            modalities=self.modalities,
            user_list=self.user_list,
            activity_list=self.activity_list,
            min_file_size_bytes=min_file_size_bytes,
        )
        self.raw_dataset = OctonetDataset(self.metadata_df, segmentation_flag=self.segmentation_flag)
        self.recording_ids = tuple(self._build_recording_ids())
        self.sample_recording_ids = tuple(self._build_sample_recording_ids())

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, index: int) -> dict:
        raw_sample = self.raw_dataset[index]
        collated_sample = custom_collate([raw_sample])

        modality_data = {}
        for modality in self.modalities:
            spec = MODALITY_SPECS[modality]
            modality_tensor = collated_sample["modality_data"].get(spec.octonet_name)
            modality_data[modality] = self._convert_modality_tensor(modality, modality_tensor)

        activity = raw_sample["activity"]
        return {
            "user_id": raw_sample["user_id"],
            "activity": activity,
            "label": ACTIVITY_TO_INDEX[activity],
            "recording_id": self.sample_recording_ids[index],
            "modality_data": modality_data,
        }

    def _build_recording_ids(self) -> list[str]:
        recording_ids: list[str] = []
        for row_idx, row in self.metadata_df.reset_index(drop=True).iterrows():
            recording_time = row.get("recording_time", "")
            cut_timestamps = row.get("cut_timestamps", "")
            recording_ids.append(
                f"row{row_idx}|user{int(row['user_id'])}|{row['activity']}|{recording_time}|{cut_timestamps}"
            )
        return recording_ids

    def _build_sample_recording_ids(self) -> list[str]:
        if not self.segmentation_flag:
            return list(self.recording_ids)

        sample_recording_ids: list[str] = []
        for row_idx, seg_info in enumerate(self.raw_dataset.segments_per_row):
            num_segments = int(seg_info.get("num_segments", 0))
            if num_segments <= 0:
                num_segments = 1
            sample_recording_ids.extend([self.recording_ids[row_idx]] * num_segments)

        if len(sample_recording_ids) != len(self.raw_dataset):
            raise RuntimeError(
                "Failed to align recording ids with segmented dataset length: "
                f"{len(sample_recording_ids)} != {len(self.raw_dataset)}"
            )
        return sample_recording_ids

    @staticmethod
    def _pad_or_crop(tensor: torch.Tensor, target_shape: Sequence[int]) -> torch.Tensor:
        target = tuple(int(size) for size in target_shape)
        output = torch.zeros(target, dtype=tensor.dtype)
        common_shape = tuple(min(current, target_dim) for current, target_dim in zip(tensor.shape, target))
        src_slices = tuple(slice(0, size) for size in common_shape)
        output[src_slices] = tensor[src_slices]
        return output

    @staticmethod
    def _sanitize_mmwave(sample_tensor: torch.Tensor) -> torch.Tensor:
        sample_tensor = sample_tensor.clone().float()
        reasonable_mask = torch.abs(sample_tensor) < 1.0e10
        sample_tensor = sample_tensor * reasonable_mask

        coordinate_ranges = [(-5.0, 5.0), (0.0, 3.0), (-5.0, 5.0), (-2.0, 2.0)]
        for dim, (low, high) in enumerate(coordinate_ranges):
            if dim >= sample_tensor.size(-1):
                break
            sample_tensor[..., dim].clamp_(min=low, max=high)
        return sample_tensor

    def _convert_modality_tensor(self, modality: str, collated_tensor: torch.Tensor | None) -> torch.Tensor:
        if collated_tensor is None:
            return zero_tensor_for_modality(modality)

        sample_tensor = collated_tensor.squeeze(0)

        if modality == "imu":
            sample_tensor = sample_tensor.reshape(sample_tensor.size(0), sample_tensor.size(1), -1)
        elif modality == "uwb":
            sample_tensor = sample_tensor.float()
        elif modality == "wifi":
            sample_tensor = torch.abs(sample_tensor).permute(0, 2, 1, 3).contiguous()
            sample_tensor = sample_tensor.reshape(-1, sample_tensor.size(-2), sample_tensor.size(-1))
        elif modality == "tof":
            sample_tensor = sample_tensor.permute(0, 1, 4, 2, 3).contiguous()
            sample_tensor = sample_tensor.reshape(-1, sample_tensor.size(-2), sample_tensor.size(-1))
        elif modality == "mmwave":
            sample_tensor = self._sanitize_mmwave(sample_tensor)
            sample_tensor = sample_tensor.permute(0, 3, 1, 2).contiguous()
            sample_tensor = sample_tensor.reshape(-1, sample_tensor.size(-2), sample_tensor.size(-1))
        else:  # pragma: no cover - guarded by normalize_modalities
            raise ValueError(f"Unsupported modality: {modality}")

        sample_tensor = sample_tensor.float()
        return self._pad_or_crop(sample_tensor, MODALITY_SPECS[modality].target_shape)


def make_dataset(config: Mapping[str, object]) -> Dataset | tuple[Subset, ...]:
    dataset = OctoNetHARDataset(
        dataset_path=str(config.get("dataset_path", DEFAULT_DATASET_PATH)),
        modalities=config.get("modality"),
        user_list=config.get("user_list"),
        activity_list=config.get("activity_list"),
        segmentation_flag=bool(config.get("segmentation_flag", True)),
        min_file_size_bytes=int(config.get("min_file_size_bytes", 100)),
    )

    split_ratio = config.get("data_split")
    if split_ratio is None:
        return dataset

    splitter = split_dataset_by_recording if bool(config.get("recording_level_split", True)) else split_dataset
    return splitter(dataset, split_ratio=split_ratio, seed=int(config.get("init_rand_seed", 41)))


__all__ = [
    "DEFAULT_DATASET_PATH",
    "OctoNetHARDataset",
    "build_filtered_metadata",
    "make_dataset",
    "split_dataset",
    "split_dataset_by_recording",
]
