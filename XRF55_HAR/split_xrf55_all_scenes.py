#!/usr/bin/env python3
"""Prepare the XRF55 Part 1 train/test split expected by COMPASS.

Expected source layout:
    {src}/Scene{N}/Scene{N}/{RFID,WiFi,mmWave}/*.npy

Generated layout:
    {dst}/{train_data,test_data}/{modality}/Scene{N}/Scene{N}/*.npy

Files are named ``{person_id}_{action_id}_{repetition_id}.npy``. Following
the official XRF55 protocol, repetitions 1--14 are used for training and
repetitions 15--20 are used for testing. The generated files are symbolic
links to the extracted Part 1 data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


SCENES = ("Scene1", "Scene2", "Scene3", "Scene4")
MODALITIES = ("RFID", "WiFi", "mmWave")
TRAIN_REPETITIONS = range(1, 15)
TEST_REPETITIONS = range(15, 21)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "data" / "XRF55"


def parse_repetition(filename: str) -> int | None:
    """Return the repetition ID from an XRF55 filename."""
    parts = Path(filename).stem.split("_")
    if len(parts) != 3:
        return None
    try:
        return int(parts[2])
    except ValueError:
        return None


def split_name(repetition: int) -> str | None:
    if repetition in TRAIN_REPETITIONS:
        return "train_data"
    if repetition in TEST_REPETITIONS:
        return "test_data"
    return None


def create_link(source: Path, destination: Path, *, force: bool) -> str:
    """Create one absolute symbolic link and return its status."""
    if destination.is_symlink():
        if destination.resolve(strict=False) == source:
            return "existing"
        if not force:
            raise FileExistsError(
                f"{destination} already points elsewhere; pass --force to replace it"
            )
        destination.unlink()
    elif destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; pass --force to replace it"
            )
        destination.unlink()

    destination.symlink_to(source)
    return "created"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the official XRF55 Part 1 repetition split."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Extracted Part 1 root containing Scene1 through Scene4.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output dataset root (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the source and report counts without creating links.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files or links at destination paths.",
    )
    args = parser.parse_args()

    source_root = args.src.expanduser().resolve()
    output_root = args.dst.expanduser().resolve()
    if not source_root.is_dir():
        parser.error(f"source directory does not exist: {source_root}")

    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    created = 0
    existing = 0
    invalid = []

    for scene in SCENES:
        reference_names: set[str] | None = None

        for modality in MODALITIES:
            source_dir = source_root / scene / scene / modality
            if not source_dir.is_dir():
                parser.error(f"expected source directory is missing: {source_dir}")

            names = {path.name for path in source_dir.glob("*.npy")}
            if reference_names is None:
                reference_names = names
            elif names != reference_names:
                missing = sorted(reference_names - names)
                extra = sorted(names - reference_names)
                parser.error(
                    f"modalities are not aligned in {scene}/{modality}: "
                    f"{len(missing)} missing and {len(extra)} extra files"
                )

            for source_path in sorted(source_dir.glob("*.npy")):
                repetition = parse_repetition(source_path.name)
                split = split_name(repetition) if repetition is not None else None
                if split is None:
                    invalid.append(str(source_path))
                    continue

                stats[scene][modality][split] += 1
                if args.dry_run:
                    continue

                destination_dir = output_root / split / modality / scene / scene
                destination_dir.mkdir(parents=True, exist_ok=True)
                status = create_link(
                    source_path.resolve(),
                    destination_dir / source_path.name,
                    force=args.force,
                )
                created += status == "created"
                existing += status == "existing"

    if invalid:
        examples = "\n".join(f"  {path}" for path in invalid[:5])
        parser.error(
            f"{len(invalid)} files have invalid names or repetition IDs; examples:\n"
            f"{examples}"
        )

    print(f"Source: {source_root}")
    print(f"Destination: {output_root}")
    print("Split: repetitions 1-14 train, 15-20 test")
    print()
    print(f"{'Scene':<10} {'Modality':<10} {'Train':>8} {'Test':>8}")
    print("-" * 40)
    for scene in SCENES:
        for modality in MODALITIES:
            counts = stats[scene][modality]
            print(
                f"{scene:<10} {modality:<10} "
                f"{counts['train_data']:>8} {counts['test_data']:>8}"
            )

    train_samples = sum(
        stats[scene][MODALITIES[0]]["train_data"] for scene in SCENES
    )
    test_samples = sum(
        stats[scene][MODALITIES[0]]["test_data"] for scene in SCENES
    )
    print("-" * 40)
    print(f"Samples: {train_samples:,} train, {test_samples:,} test")
    if args.dry_run:
        print("Dry run complete; no links were created.")
    else:
        print(f"Links: {created:,} created, {existing:,} already correct.")


if __name__ == "__main__":
    main()
