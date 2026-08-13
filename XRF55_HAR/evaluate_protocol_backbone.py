"""Evaluate one fixed-final split-specific X-Fi ResNet-18 checkpoint once."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset.xrf55_dataset import XRF55_Protocol_Modality_Dataset


MODEL_PATHS = {
    "mmwave": Path("mmWave/mmwave_ResNet18.pt"),
    "wifi": Path("WIFI/wifi_ResNet18.pt"),
    "rfid": Path("RFID/RFID_ResNet18.pt"),
}

EXPECTED = {
    "subject_split_21_9": (23100, {"subjects_22_30": 9900}),
    "scene_split": (
        23100,
        {"Scene2": 3300, "Scene3": 3300, "Scene4": 3300},
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--backbone-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--modality", choices=tuple(MODEL_PATHS), required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dataset_specs(protocol: str) -> tuple[dict, dict[str, dict]]:
    if protocol == "subject_split_21_9":
        train = {
            "scenes": ["Scene1"],
            "subjects": range(1, 22),
            "split_name": "subjects_01_21",
        }
        tests = {
            "subjects_22_30": {
                "scenes": ["Scene1"],
                "subjects": range(22, 31),
                "split_name": "subjects_22_30",
            }
        }
        return train, tests
    if protocol == "scene_split":
        train = {
            "scenes": ["Scene1"],
            "repetitions": range(1, 15),
            "split_name": "Scene1_trials_01_14",
        }
        tests = {
            scene: {"scenes": [scene], "split_name": scene}
            for scene in ("Scene2", "Scene3", "Scene4")
        }
        return train, tests
    raise ValueError(protocol)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, int]:
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    count = 0
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        logits = model(inputs)
        loss_sum += criterion(logits, labels).item() * labels.numel()
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.numel()
    return loss_sum / count, correct / count, count


def main() -> None:
    args = parse_args()
    relative_model_path = MODEL_PATHS[args.modality]
    run_root = args.backbone_root / args.protocol / f"seed{args.seed}"
    model_path = run_root / relative_model_path
    metadata_path = model_path.with_suffix(".json")
    output_path = model_path.with_suffix(".evaluation.json")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")

    metadata = json.loads(metadata_path.read_text())
    expected_metadata = {
        "protocol": args.protocol,
        "modality": args.modality,
        "seed": args.seed,
        "epochs": 100,
        "test_evaluation": "disabled",
        "checkpoint_selection": "fixed final epoch; test evaluation disabled",
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{metadata_path}: {key} expected {expected!r}, "
                f"found {metadata.get(key)!r}"
            )

    train_spec, test_specs = dataset_specs(args.protocol)
    train_data = XRF55_Protocol_Modality_Dataset(
        raw_root=args.raw_root,
        modality=args.modality,
        **train_spec,
    )
    test_data = {
        name: XRF55_Protocol_Modality_Dataset(
            raw_root=args.raw_root,
            modality=args.modality,
            **specification,
        )
        for name, specification in test_specs.items()
    }
    overlaps = {
        name: len(train_data.identities & dataset.identities)
        for name, dataset in test_data.items()
    }
    expected_train, expected_tests = EXPECTED[args.protocol]
    observed_tests = {name: len(dataset) for name, dataset in test_data.items()}
    if len(train_data) != expected_train or observed_tests != expected_tests:
        raise RuntimeError(
            f"Unexpected sample counts: train={len(train_data)}, test={observed_tests}"
        )
    if any(overlaps.values()):
        raise RuntimeError(f"Train/test overlap: {overlaps}")

    device = torch.device(args.device)
    model = torch.load(model_path, map_location="cpu", weights_only=False).to(device)
    started = time.monotonic()
    results = {}
    weighted_correct = 0.0
    evaluated_samples = 0
    for name, dataset in test_data.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.workers > 0,
            prefetch_factor=2 if args.workers > 0 else None,
        )
        loss, accuracy, count = evaluate(model, loader, device)
        results[name] = {
            "loss": loss,
            "accuracy": accuracy,
            "accuracy_percent": 100 * accuracy,
            "samples": count,
        }
        weighted_correct += accuracy * count
        evaluated_samples += count
        print(
            f"test={name} accuracy={100 * accuracy:.6f}% samples={count}",
            flush=True,
        )

    aggregate = weighted_correct / evaluated_samples
    report = {
        "status": "one-time evaluation of fixed-final split-specific X-Fi backbone",
        "source_metadata": str(metadata_path),
        "model_path": str(model_path),
        "protocol": args.protocol,
        "modality": args.modality,
        "seed": args.seed,
        "fixed_epoch": metadata["epochs"],
        "checkpoint_selection": "fixed final epoch; test evaluated once",
        "train_samples": len(train_data),
        "test_samples": observed_tests,
        "train_test_overlap": overlaps,
        "evaluation": results,
        "aggregate_test_accuracy": aggregate,
        "aggregate_test_accuracy_percent": 100 * aggregate,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"aggregate_accuracy={100 * aggregate:.6f}% saved={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
