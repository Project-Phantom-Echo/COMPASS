"""Train COMPASS on XRF55.

Backbones are fine-tuned by default. Pass ``freeze_backbone=True`` to keep
their parameters frozen.

Usage:
    python train.py with task_finetune_xrf55 [key=value ...]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backbone_models'))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from torch.optim import AdamW
import gc
import json
import pprint
from datetime import datetime

from shared.utils.utils import fix_seeds, setup_cudnn, get_logger
from shared.utils.metric import Metrics
from config import parse_config
from shared.utils.optimizers import get_optimizer
from shared.utils.schedulers import get_scheduler
from misc import collate_fn_padd

# --- 3. 引入自定义模块 ---
from models.cmpt_model_xrf55 import (XRF55_CMPT_Net)
from dataset.xrf55_dataset import XRF55_Dataset, make_protocol_datasets
from Encoders import Encoder, Decoder
from Extractor import mmwave_feature_extractor, wifi_feature_extractor, rfid_feature_extractor


PROJECT_DIR = Path(__file__).resolve().parent


def print_trainable_parameters(model, logger=None):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    msg = f"可训练参数: {trainable_params} || 总参数: {all_param} || 可训练占比 (%): {100 * trainable_params / all_param:.2f}"

    if logger:
        logger.info(msg)
    else:
        print(msg)


def validate_protocol_backbones(backbone_dir, protocol, seed, logger):
    """Reject split-specific backbones that did not pass the train-only check."""
    if not backbone_dir:
        return
    base_path = Path(backbone_dir).expanduser().resolve()
    metadata_paths = (
        base_path / 'mmWave' / 'mmwave_ResNet18.json',
        base_path / 'WIFI' / 'wifi_ResNet18.json',
        base_path / 'RFID' / 'RFID_ResNet18.json',
    )
    for metadata_path in metadata_paths:
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Missing backbone metadata: {metadata_path}')
        metadata = json.loads(metadata_path.read_text())
        checks = {
            'protocol': metadata.get('protocol') == protocol,
            'seed': metadata.get('seed') == seed,
            'train_samples': metadata.get('train_samples') == 23100,
            'test_evaluation': metadata.get('test_evaluation') == 'disabled',
            'fixed_final': metadata.get('checkpoint_selection')
            == 'fixed final epoch; test evaluation disabled',
            'train_accuracy': metadata.get('final_train_accuracy', 0.0) >= 0.90,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(f'Backbone validation failed for {metadata_path}: {failed}')
        logger.info(
            f"Validated split-clean backbone: {metadata_path.name}, "
            f"train_accuracy={metadata['final_train_accuracy']:.4f}"
        )


def load_custom_encoders(device, logger=None, freeze_backbone=False, backbone_dir=None):
    """Load the pretrained modality encoders."""
    backbone_mode = "冻结" if freeze_backbone else "解冻"
    if logger:
        logger.info(f"正在加载预训练 Backbone ({backbone_mode}模式)...")
    else:
        print(f"正在加载预训练 Backbone ({backbone_mode}模式)...")

    base_path = Path(backbone_dir).expanduser().resolve() if backbone_dir else PROJECT_DIR / "backbone_models"
    if logger:
        logger.info(f"Backbone directory: {base_path}")

    try:
        mmwave_model = torch.load(base_path / "mmWave" / "mmwave_ResNet18.pt", map_location="cpu")
        wifi_model = torch.load(base_path / "WIFI" / "wifi_ResNet18.pt", map_location="cpu")
        rfid_model = torch.load(base_path / "RFID" / "RFID_ResNet18.pt", map_location="cpu")
    except FileNotFoundError as e:
        err_msg = f"错误: 找不到模型文件。请检查 {base_path} 下的文件结构。"
        if logger:
            logger.error(err_msg)
        else:
            print(err_msg)
        raise e

    mmwave_extractor = mmwave_feature_extractor(mmwave_model).eval() # 注意：BN层依然保持eval模式通常更稳定，若需训练BN可改为.train()
    wifi_extractor = wifi_feature_extractor(wifi_model).eval()
    rfid_extractor = rfid_feature_extractor(rfid_model).eval()

    # Unfreeze backbones for end-to-end fine-tuning (default), unless freeze_backbone is set
    # (frozen = CMPT-style PEFT-spirit / lower-capacity baseline row).
    requires_grad = not freeze_backbone
    for model in [mmwave_extractor, wifi_extractor, rfid_extractor]:
        for param in model.parameters():
            param.requires_grad = requires_grad

    encode_info = [(512, 512, 32), (512, 512, 4), (512, 512, 5)]
    task_encoders = nn.ModuleDict()
    task_encoders['mmwave'] = Encoder(0, mmwave_extractor, encode_info[0])
    task_encoders['wifi'] = Encoder(1, wifi_extractor, encode_info[1])
    task_encoders['rfid'] = Encoder(2, rfid_extractor, encode_info[2])

    return task_encoders


def generate_random_mask(modalities=['mmwave', 'wifi', 'rfid'], drop_prob=0.5):
    mask = {m: 1 for m in modalities}
    if random.random() > drop_prob:
        return None
    num_keep = random.randint(1, len(modalities) - 1)
    keep_mods = random.sample(modalities, num_keep)
    for m in modalities:
        if m not in keep_mods:
            mask[m] = 0
    return mask


def evaluate_xrf55(model, dataloader, device, loss_fn):
    model.eval()
    metric = Metrics()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
            mmwave_data, wifi_data, rfid_data, labels = batch
            inputs = {
                'mmwave': mmwave_data.to(device),
                'wifi': wifi_data.to(device),
                'rfid': rfid_data.to(device)
            }
            labels = labels.to(device).long()

            # 验证时通常假设全模态可见
            logits, _, _ = model(inputs, missing_mask=None)

            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            metric.update(predicted, labels)

    scores = metric.compute_score(prefix='test_')
    scores['test_loss'] = total_loss / len(dataloader)
    return scores


def main(_config):
    # --- 1. 基础设置与时间戳生成 ---
    fix_seeds(_config["seed"])
    setup_cudnn()

    # 【核心修改】自动添加时间戳，防止覆盖
    base_exp_name = _config["wandb_exp_name"]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 生成如 20251216_153000
    wandb_exp_name = f"{base_exp_name}_{timestamp}"

    # 创建带时间戳的保存目录
    save_dir = Path(_config['save_dir'], wandb_exp_name)
    os.makedirs(save_dir, exist_ok=True)

    # 获取 Logger 对象
    logger = get_logger(save_dir / 'train.log')

    logger.info(f"实验名称已更新为: {wandb_exp_name}")
    logger.info(f"结果将保存在: {save_dir}")
    logger.info("实验配置:")
    # logger.info(pprint.pformat(_config))

    if len(_config['gpu_ids']) > 0 and torch.cuda.is_available():
        # 如果你在命令行传了 gpu_ids="[3]"，这里就会变成 "cuda:3"
        main_gpu_id = _config['gpu_ids'][0]
        device = torch.device(f"cuda:{main_gpu_id}")
        logger.info(f"正在使用指定 GPU: cuda:{main_gpu_id}")
    else:
        # 兜底逻辑
        device = torch.device(_config['device'])
        logger.info(f"正在使用默认设备: {device}")

    # 数据集路径兜底
    data_root = (
        os.environ.get("XRF55_DATA_ROOT")
        or _config.get("data_dir")
        or str(PROJECT_DIR.parent / "data" / "XRF55")
    )
    data_root_path = Path(data_root).expanduser()
    if not data_root_path.is_absolute():
        data_root_path = PROJECT_DIR.parent / data_root_path
    data_root = str(data_root_path.resolve())

    if not os.path.exists(data_root):
        logger.error(f"找不到数据集路径: {data_root}")
        raise FileNotFoundError(f"找不到数据集路径: {data_root}")

    # 2. 数据集初始化
    logger.info("正在初始化数据集...")
    protocol = _config.get('protocol', 'trial_split')
    scene = _config.get('scene', 'all')
    if protocol == 'trial_split':
        trainset = XRF55_Dataset(root_dir=data_root, split='train', scene=scene)
        testsets = {
            'trial_split': XRF55_Dataset(root_dir=data_root, split='test', scene=scene)
        }
    else:
        raw_data_root = _config.get('raw_data_dir') or os.environ.get('XRF55_RAW_ROOT')
        if not raw_data_root:
            raise ValueError(
                f'protocol={protocol} requires raw_data_dir or XRF55_RAW_ROOT'
            )
        raw_data_root = str(Path(raw_data_root).expanduser().resolve())
        logger.info(f'Raw XRF55 root: {raw_data_root}')
        trainset, testsets = make_protocol_datasets(raw_data_root, protocol)
    logger.info(
        f"protocol={protocol}: {len(trainset)} train / "
        + ", ".join(f"{name}={len(dataset)} test" for name, dataset in testsets.items())
    )

    # As released, checkpoint selection runs on the test set. val_ratio>0 carves a
    # held-out slice off the training split and selects on that instead, so the
    # test set is only read for the final report.
    val_ratio = float(_config.get('val_ratio', 0.0))
    final_epoch_only = bool(_config.get('final_epoch_only', False))
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if final_epoch_only and val_ratio > 0:
        raise ValueError('final_epoch_only=True cannot be combined with val_ratio>0')
    holdout = None
    if val_ratio > 0:
        num_val = int(len(trainset) * val_ratio)
        split_generator = torch.Generator().manual_seed(_config["seed"])
        trainset, holdout = random_split(
            trainset, [len(trainset) - num_val, num_val], generator=split_generator)
        logger.info(f"val_ratio={val_ratio}: {len(trainset)} train / {len(holdout)} val "
                    f"(selection), {sum(len(dataset) for dataset in testsets.values())} "
                    "test (final report only)")
    else:
        if final_epoch_only:
            logger.info('Fixed final-epoch checkpoint; test data is read only after training')
        elif len(testsets) != 1:
            raise ValueError(
                'Multiple test sets require final_epoch_only=True or a train holdout'
            )
        else:
            logger.info("val_ratio=0: checkpoint selected on the test set (released behaviour)")

    # 3. 模型初始化
    proj_dim = 32
    embed_dim = 512
    freeze_backbone = bool(_config.get('freeze_backbone', False))
    validate_protocol_backbones(
        _config.get('backbone_dir'), protocol, _config['seed'], logger
    )
    task_encoders = load_custom_encoders(
        device,
        logger=logger,
        freeze_backbone=freeze_backbone,
        backbone_dir=_config.get('backbone_dir'),
    )
    task_decoder_config = ([embed_dim, _config['class_num']], 'classification')
    cmpt_dropout = _config.get('cmpt_dropout', 0.1)
    fusion_type = _config.get('fusion_type', 'sum')
    proxy_agg = _config.get('proxy_agg', 'uniform')
    conf_temperature = float(_config.get('conf_temperature', 1.0))
    generator_type = _config.get('generator_type', 'transformer')
    generator_mode = _config.get('generator_mode', 'pairwise')
    source_priority = _config.get('source_priority', None)
    missing_fill = _config.get('missing_fill', 'proxy')
    impute_hidden_dim = _config.get('impute_hidden_dim', None)
    impute_dropout = float(_config.get('impute_dropout', 0.3))
    no_proxy = bool(_config.get('no_proxy', False))
    effective_no_proxy = no_proxy if missing_fill == 'proxy' else False
    model = XRF55_CMPT_Net(
        task_encoders=task_encoders,
        task_decoder=task_decoder_config,
        proj_dim=proj_dim,
        embed_dim=embed_dim,
        dropout=cmpt_dropout,
        fusion_type=fusion_type,
        proxy_agg=proxy_agg,
        conf_temperature=conf_temperature,
        generator_type=generator_type,
        generator_mode=generator_mode,
        source_priority=source_priority,
        no_proxy=effective_no_proxy,
        missing_fill=missing_fill,
        impute_hidden_dim=impute_hidden_dim,
        impute_dropout=impute_dropout
    )

    print_trainable_parameters(model, logger=logger)

    if len(_config['gpu_ids']) > 1:
        model = torch.nn.DataParallel(model, device_ids=_config['gpu_ids'])
    model = model.to(device)

    # 4. 优化器 & 调度器
    optimizer = get_optimizer(model, _config['optim_type'], _config['learning_rate'], _config['weight_decay'])
    iters_per_epoch = len(trainset) // _config['batch_size']
    scheduler = get_scheduler(_config['scheduler'], optimizer, int((_config['max_epoch'] + 1) * iters_per_epoch),
                              _config['power'], iters_per_epoch * _config['warmup'], _config['warmup_ratio'])

    # 5. DataLoader
    trainloader = DataLoader(
        trainset,
        batch_size=_config['batch_size'],
        num_workers=_config['num_workers'],
        drop_last=True,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True if _config['num_workers'] > 0 else False,
        prefetch_factor=4 if _config['num_workers'] > 0 else None,
        collate_fn=collate_fn_padd
    )

    testloaders = {
        name: DataLoader(
            testset,
            batch_size=_config['batch_size'],
            num_workers=_config['num_workers'],
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if _config['num_workers'] > 0 else False,
            prefetch_factor=4 if _config['num_workers'] > 0 else None,
            collate_fn=collate_fn_padd,
        )
        for name, testset in testsets.items()
    }

    # Loader driving checkpoint selection: the held-out train slice when
    # val_ratio>0, otherwise the test set (as released).
    if holdout is not None:
        selloader = DataLoader(
            holdout,
            batch_size=_config['batch_size'],
            num_workers=_config['num_workers'],
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if _config['num_workers'] > 0 else False,
            prefetch_factor=4 if _config['num_workers'] > 0 else None,
            collate_fn=collate_fn_padd
        )
    else:
        selloader = None if final_epoch_only else next(iter(testloaders.values()))

    # 6. 损失函数
    task_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    align_loss_fn = nn.MSELoss()
    metric = Metrics()

    best_performance = 0.0
    best_epoch = 0

    # 验证间隔
    eval_interval = 5
    lambda_align = float(_config.get('cmpt_loss_weight', 0.2))
    lambda_vicreg = float(_config.get('lambda_vicreg', 0.0))
    vicreg_inv = float(_config.get('vicreg_inv', 25.0))
    vicreg_var = float(_config.get('vicreg_var', 25.0))
    vicreg_cov = float(_config.get('vicreg_cov', 1.0))
    lambda_proxy_cls = float(_config.get('lambda_proxy_cls', 0.0))
    lambda_recon = float(_config.get('lambda_recon', 0.5))
    use_impute = missing_fill == 'impute'
    if effective_no_proxy:
        lambda_vicreg = 0.0
    if use_impute:
        lambda_vicreg = 0.0
        lambda_proxy_cls = 0.0
        logger.info(f"Impute missing-fill enabled: lambda_recon={lambda_recon}, hidden_dim={impute_hidden_dim}, dropout={impute_dropout}")
        if no_proxy:
            logger.info("missing_fill=impute ignores no_proxy=True")
    elif lambda_vicreg > 0:
        logger.info(
            f"VICReg enabled: lambda={lambda_vicreg}, inv={vicreg_inv}, var={vicreg_var}, cov={vicreg_cov}")
    if not use_impute and lambda_proxy_cls > 0:
        logger.info(f"ProxyCls enabled: lambda={lambda_proxy_cls}")

    # --- 训练循环 ---
    logger.info(f"开始训练 (FP32模式)，共 {_config['max_epoch']} 个 Epoch...")
    simulate_missing = _config.get('simulate_missing', True)
    drop_prob = float(_config.get("drop_prob", 0.7))
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
    logger.info(f"鲁棒性训练模式 (Simulate Missing): {simulate_missing}")
    logger.info(f"随机模态缺失概率: {drop_prob}")
    logger.info(f"验证频率: 每 {eval_interval} 轮验证一次")
    if use_impute:
        logger.info(f"fusion_type={fusion_type} | missing_fill=impute | lambda_recon={lambda_recon}")
    else:
        logger.info(f"fusion_type={fusion_type} | missing_fill=proxy | no_proxy={effective_no_proxy} | cmpt_loss_weight={lambda_align}")

    for epoch in range(_config['max_epoch']):
        model.train()
        total_loss_meter = 0.0
        task_loss_meter = 0.0
        align_loss_meter = 0.0
        proxy_cls_loss_meter = 0.0
        recon_loss_meter = 0.0

        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch + 1}] LR: {lr:.6f}")

        for iter, batch in pbar:
            mmwave_data, wifi_data, rfid_data, labels = batch
            inputs = {
                'mmwave': mmwave_data.to(device),
                'wifi': wifi_data.to(device),
                'rfid': rfid_data.to(device)
            }
            labels = labels.to(device).long()

            current_mask = None
            if simulate_missing:
                current_mask = generate_random_mask(drop_prob=drop_prob)

            optimizer.zero_grad(set_to_none=True)

            logits, fill_outputs, real_globals = model(inputs, missing_mask=current_mask)

            t_loss = task_loss_fn(logits, labels)

            a_loss = torch.tensor(0.0, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            vicreg_loss = torch.tensor(0.0, device=device)
            proxy_cls_loss = torch.tensor(0.0, device=device)

            if use_impute:
                recon_terms = []
                for modality, imputed_feat in fill_outputs.items():
                    is_missing = current_mask is not None and current_mask.get(modality, 1) == 0
                    if is_missing and modality in real_globals:
                        recon_terms.append(align_loss_fn(imputed_feat, real_globals[modality]))
                if recon_terms:
                    recon_loss = torch.stack(recon_terms).mean()
                loss = t_loss + lambda_recon * recon_loss
            else:
                proxies = fill_outputs

                # 2. 对齐损失 (引入归一化)
                count = 0
                for key, proxy_feat in proxies.items():
                    src, _, tgt = key.partition('_to_')
                    if tgt in real_globals:
                        a_loss += align_loss_fn(proxy_feat, real_globals[tgt])
                        count += 1

                if count > 0:
                    a_loss = a_loss / count

                # VICReg cross-modal alignment on real modality globals.
                if lambda_vicreg > 0 and len(real_globals) >= 2:
                    mods = list(real_globals.keys())
                    pair_count = 0
                    for i_m in range(len(mods)):
                        for j_m in range(i_m + 1, len(mods)):
                            zi = real_globals[mods[i_m]].squeeze(1)
                            zj = real_globals[mods[j_m]].squeeze(1)
                            inv_loss = F.mse_loss(zi, zj)
                            std_zi = torch.sqrt(zi.var(dim=0) + 1e-4)
                            std_zj = torch.sqrt(zj.var(dim=0) + 1e-4)
                            var_loss = (F.relu(1 - std_zi).mean() + F.relu(1 - std_zj).mean()) / 2
                            d = zi.size(1)
                            zi_c = zi - zi.mean(0)
                            zj_c = zj - zj.mean(0)
                            cov_zi = (zi_c.T @ zi_c) / (zi.size(0) - 1)
                            cov_zj = (zj_c.T @ zj_c) / (zj.size(0) - 1)
                            off_diag_mask = ~torch.eye(d, dtype=torch.bool, device=device)
                            cov_loss = (
                                cov_zi[off_diag_mask].pow(2).sum() / d +
                                cov_zj[off_diag_mask].pow(2).sum() / d
                            ) / 2
                            vicreg_loss += (
                                vicreg_inv * inv_loss +
                                vicreg_var * var_loss +
                                vicreg_cov * cov_loss
                            )
                            pair_count += 1
                    if pair_count > 0:
                        vicreg_loss = vicreg_loss / pair_count
                loss = t_loss + lambda_align * a_loss
                if lambda_vicreg > 0:
                    loss = loss + lambda_vicreg * vicreg_loss
                if lambda_proxy_cls > 0 and proxies:
                    raw_model = model.module if isinstance(model, nn.DataParallel) else model
                    proxy_cls_count = 0
                    for _, proxy_feat in proxies.items():
                        proxy_logit = raw_model.decoder(proxy_feat.squeeze(1))
                        proxy_cls_loss += task_loss_fn(proxy_logit, labels)
                        proxy_cls_count += 1
                    if proxy_cls_count > 0:
                        proxy_cls_loss = proxy_cls_loss / proxy_cls_count
                        loss = loss + lambda_proxy_cls * proxy_cls_loss
            _, predicted = torch.max(logits, 1)
            metric.update(predicted, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss_meter += loss.item()
            task_loss_meter += t_loss.item()
            align_loss_meter += a_loss.item()
            proxy_cls_loss_meter += proxy_cls_loss.item()
            recon_loss_meter += recon_loss.item()
            if use_impute:
                desc = f"Epoch: [{epoch + 1}] Loss: {loss.item():.4f} (Task: {t_loss.item():.4f} Recon: {recon_loss.item():.4f}"
            else:
                desc = f"Epoch: [{epoch + 1}] Loss: {loss.item():.4f} (Task: {t_loss.item():.4f} Align: {a_loss.item():.4f}"
                if lambda_proxy_cls > 0:
                    desc += f" ProxyCls: {proxy_cls_loss.item():.4f}"
                if lambda_vicreg > 0:
                    desc += f" VICReg: {vicreg_loss.item():.4f}"
            desc += ")"
            pbar.set_description(desc)

        # --- Epoch 结束 ---
        avg_train_loss = total_loss_meter / (iter + 1)
        train_scores = metric.compute_score(prefix='train_')
        train_scores.update({
            'train_loss': avg_train_loss,
            'epoch': epoch
        })
        if use_impute:
            avg_recon_loss = recon_loss_meter / (iter + 1)
            train_scores['train_recon_loss'] = avg_recon_loss
        metric.reset()

        # --- 验证阶段 ---
        if (not final_epoch_only) and (
            (epoch + 1) % eval_interval == 0 or (epoch + 1) == _config['max_epoch']
        ):
            logger.info(f"\n[Epoch {epoch + 1}] 正在进行验证...")
            val_scores = evaluate_xrf55(model, selloader, device, task_loss_fn)

            results_str = pprint.pformat({**train_scores, **val_scores})
            logger.info(f"Epoch {epoch + 1} Results:\n{results_str}")

            current_acc = val_scores['test_accuracy']
            if best_performance < current_acc:
                best_performance = current_acc
                best_epoch = epoch + 1
                ckpt_path = save_dir / f"{wandb_exp_name}_best.pth"
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f">>> 新最佳 Accuracy: {best_performance:.4f} @ Epoch {best_epoch}")
        else:
            # 即使跳过验证，也记录一下 Training Loss
            logger.info(f"[Epoch {epoch + 1}] 训练 Loss: {avg_train_loss:.4f} (跳过验证)")

    logger.info("训练结束")

    if final_epoch_only:
        best_epoch = _config['max_epoch']
        ckpt_path = save_dir / f"{wandb_exp_name}_best.pth"
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f'固定最终模型已保存: Epoch {best_epoch} -> {ckpt_path}')

    # --- 训练后鲁棒性评估 ---
    logger.info("开始 7 场景鲁棒性评估...")

    # 重新加载最佳模型（训练结束时 model 可能不在最优状态）
    best_ckpt = save_dir / f"{wandb_exp_name}_best.pth"
    if best_ckpt.exists():
        best_state = torch.load(best_ckpt, map_location=device)
        clean_state = {k.replace('module.', ''): v for k, v in best_state.items()}
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model.load_state_dict(clean_state, strict=True)

        from eval_all import evaluate_robustness, print_results
        saved_results = {
            'protocol': protocol,
            'seed': _config['seed'],
            'checkpoint_epoch': best_epoch,
            'final_epoch_only': final_epoch_only,
            'test_sets': {},
        }
        for test_name, testloader in testloaders.items():
            print(f'\nTest set: {test_name}')
            results = evaluate_robustness(raw_model, testloader, device)
            print_results(results)
            saved_results['test_sets'][test_name] = {
                scenario: {
                    'accuracy': stats['correct'] / stats['total'] * 100,
                    'loss': stats['loss'] / stats['total'],
                    'samples': stats['total'],
                }
                for scenario, stats in results.items()
            }
        results_path = save_dir / 'robustness_results.json'
        results_path.write_text(json.dumps(saved_results, indent=2) + '\n')
        logger.info(f'Robustness results saved: {results_path}')
    else:
        logger.warning(f"未找到最佳模型: {best_ckpt}，跳过评估")


if __name__ == "__main__":
    main(parse_config())
