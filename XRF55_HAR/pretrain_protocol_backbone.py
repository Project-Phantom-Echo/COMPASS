"""Pretrain one X-Fi ResNet-18 using only a protocol's training samples."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from backbone_models.RFID.ResNet import resnet18 as rfid_resnet18
from backbone_models.WIFI.ResNet import resnet18 as wifi_resnet18
from backbone_models.mmWave.ResNet import resnet18 as mmwave_resnet18
from dataset.xrf55_dataset import XRF55_Protocol_Modality_Dataset


MODEL_SPECS = {
    'mmwave': {
        'constructor': mmwave_resnet18,
        'path': Path('mmWave/mmwave_ResNet18.pt'),
        'batch_size': 16,
        'learning_rate': 1e-4,
        'milestones': [],
    },
    'wifi': {
        'constructor': wifi_resnet18,
        'path': Path('WIFI/wifi_ResNet18.pt'),
        'batch_size': 32,
        'learning_rate': 1e-3,
        'milestones': [],
    },
    'rfid': {
        'constructor': rfid_resnet18,
        'path': Path('RFID/RFID_ResNet18.pt'),
        'batch_size': 16,
        'learning_rate': 1e-3,
        'milestones': [20, 40],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument(
        '--protocol',
        choices=('subject_split_21_9', 'scene_split'),
        required=True,
    )
    parser.add_argument('--modality', choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def training_spec(protocol):
    if protocol == 'subject_split_21_9':
        return {
            'raw_root': None,
            'scenes': ['Scene1'],
            'subjects': range(1, 22),
            'split_name': 'subjects_01_21',
        }
    if protocol == 'scene_split':
        return {
            'raw_root': None,
            'scenes': ['Scene1'],
            'repetitions': range(1, 15),
            'split_name': 'Scene1_trials_01_14',
        }
    raise ValueError(protocol)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    specification = training_spec(args.protocol)
    specification['raw_root'] = args.raw_root
    dataset = XRF55_Protocol_Modality_Dataset(
        modality=args.modality,
        **specification,
    )
    if len(dataset) != 23100:
        raise RuntimeError(f'Expected 23,100 training samples, found {len(dataset)}')

    settings = MODEL_SPECS[args.modality]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=settings['batch_size'],
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    model = settings['constructor']().to(device)
    # PyTorch AdamW's default weight decay (0.01) matches the recovered X-Fi
    # preparation notebooks, which specify only the learning rate.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings['learning_rate'],
        weight_decay=0.01,
    )
    scheduler = None
    if settings['milestones']:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=settings['milestones'],
            gamma=0.1,
        )
    criterion = nn.CrossEntropyLoss()
    started = time.monotonic()
    final_loss = None
    final_accuracy = None

    print(
        f"protocol={args.protocol} modality={args.modality} seed={args.seed} "
        f"train={len(dataset)} test_evaluation=disabled",
        flush=True,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        epoch_lr = optimizer.param_groups[0]['lr']
        for batch_index, (inputs, labels) in enumerate(loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_count = labels.numel()
            loss_sum += loss.item() * batch_count
            correct += (logits.argmax(1) == labels).sum().item()
            count += batch_count
        if scheduler is not None:
            scheduler.step()
        final_loss = loss_sum / count
        final_accuracy = correct / count
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} lr={epoch_lr:.8g} "
                f"train_loss={final_loss:.5f} train_accuracy={final_accuracy:.4f}",
                flush=True,
            )

    destination = args.output_root / args.protocol / f'seed{args.seed}' / settings['path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu()
    torch.save(model, destination)
    torch.save(model.state_dict(), destination.with_suffix('.state_dict.pt'))
    metadata = {
        'status': 'split-clean X-Fi backbone reproduction for COMPASS',
        'protocol': args.protocol,
        'modality': args.modality,
        'train_samples': len(dataset),
        'test_evaluation': 'disabled',
        'epochs': args.epochs,
        'batch_size': settings['batch_size'],
        'optimizer': 'AdamW',
        'learning_rate': settings['learning_rate'],
        'weight_decay': 0.01,
        'scheduler': 'MultiStepLR' if scheduler is not None else 'constant',
        'milestones': settings['milestones'],
        'gamma': 0.1 if scheduler is not None else None,
        'seed': args.seed,
        'checkpoint_selection': 'fixed final epoch; test evaluation disabled',
        'final_train_loss': final_loss,
        'final_train_accuracy': final_accuracy,
        'max_train_batches': args.max_train_batches,
        'elapsed_seconds': time.monotonic() - started,
        'model_path': str(destination),
    }
    destination.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'saved={destination}', flush=True)


if __name__ == '__main__':
    main()
