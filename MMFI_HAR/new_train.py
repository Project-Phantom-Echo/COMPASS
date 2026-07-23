"""
MMFI_HAR Single-GPU Training Script

This script provides single-machine training for the MMFi CMPT model,
aligned with the XRF55 training style.

Usage:
    python new_train.py --config config.yaml
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), 'backbones'))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import re
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from datetime import datetime


PROJECT_DIR = Path(__file__).resolve().parent

try:
    from dataset.mmfi_dataset import make_dataset
    from util import collate_fn_padd
    from shared.utils.utils import fix_seeds, get_logger
    from shared.utils.schedulers import get_scheduler
    from shared.utils.optimizers import get_optimizer
    from models.cmpt_model_mmfi import MMFi_CMPT_Net
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

FINAL_EVAL_COMBOS = [
    ("RGB Only",              True,  False, False, False),
    ("Depth Only",            False, True,  False, False),
    ("Lidar Only",            False, False, True,  False),
    ("MMWave Only",           False, False, False, True),
    ("RGB + Depth",           True,  True,  False, False),
    ("RGB + Lidar",           True,  False, True,  False),
    ("RGB + MMWave",          True,  False, False, True),
    ("Depth + Lidar",         False, True,  True,  False),
    ("Depth + MMWave",        False, True,  False, True),
    ("Lidar + MMWave",        False, False, True,  True),
    ("RGB + Depth + Lidar",   True,  True,  True,  False),
    ("RGB + Depth + MMWave",  True,  True,  False, True),
    ("RGB + Lidar + MMWave",  True,  False, True,  True),
    ("Depth + Lidar + MMWave",False, True,  True,  True),
    ("All (R+D+L+M)",         True,  True,  True,  True),
]

FINAL_EVAL_RE = re.compile(
    r"INFO:\s*-\s*(?P<name>"
    + "|".join(re.escape(name) for name, *_ in sorted(FINAL_EVAL_COMBOS, key=lambda item: len(item[0]), reverse=True))
    + r")\s*\|\s*(?P<loss>[0-9]+(?:\.[0-9]+)?)\s*\|\s*(?P<acc>[0-9]+(?:\.[0-9]+)?)"
)


def read_final_eval_results(log_path):
    results = {}
    log_path = Path(log_path)
    if not log_path.exists():
        return results
    for line in log_path.read_text(errors="replace").replace("\r", "\n").splitlines():
        match = FINAL_EVAL_RE.search(line)
        if match:
            results[match.group("name")] = (
                float(match.group("loss")),
                float(match.group("acc")),
            )
    return results


# ==========================================
# 训练辅助函数
# ==========================================
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


def generate_random_mask(modalities=('rgb', 'depth', 'mmwave', 'lidar'), drop_prob=0.7):
    """
    ✅ 对齐 XRF55: drop_prob 默认 0.7
    返回 None 表示不缺失；否则返回 dict: {mod: 0/1}
    """
    mask = {m: 1 for m in modalities}
    if random.random() > drop_prob:
        return None
    num_keep = random.randint(1, len(modalities) - 1)
    keep_mods = random.sample(list(modalities), num_keep)
    for m in modalities:
        if m not in keep_mods:
            mask[m] = 0
    return mask


def _avg_lr(scheduler, optimizer):
    """
    ✅ 对齐 XRF55: 他们用 scheduler.get_lr() 再平均
    这里做兼容：有 get_lr 就用，否则用 optimizer 的 lr
    """
    if hasattr(scheduler, "get_lr"):
        lrs = scheduler.get_lr()
        if isinstance(lrs, (list, tuple)) and len(lrs) > 0:
            return float(sum(lrs) / len(lrs))
    return float(optimizer.param_groups[0]["lr"])


@torch.no_grad()
def evaluate_mmfi(model, dataloader, device, loss_fn, is_main: bool):
    """
    ✅ 不对 depth 做 resize
    ✅ DataParallel/单机：统计本地 loss/acc（按样本数加权）
    """
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    iterator = tqdm(dataloader, desc="验证中", leave=False) if is_main else dataloader

    for batch in iterator:
        rgb, depth, mmwave, lidar, labels, _ = batch

        inputs = {}
        if rgb is not None and rgb.dim() > 1:
            inputs['rgb'] = rgb.to(device).float()
        if depth is not None:
            inputs['depth'] = depth.to(device).float()
        if mmwave is not None:
            inputs['mmwave'] = mmwave.to(device).float()
        if lidar is not None:
            inputs['lidar'] = lidar.to(device).float()

        labels = labels.to(device).long()

        logits, _, _ = model(inputs, missing_mask=None)
        loss = loss_fn(logits, labels)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)

    avg_loss = loss_sum / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0
    return {"test_loss": avg_loss, "test_accuracy": avg_acc}


# ==========================================
# 主训练逻辑（单机多卡 DataParallel）
# ==========================================
def main(args):
    # -------------------------
    # 0) 读取 Config
    # -------------------------
    if not os.path.exists(args.config):
        print(f"❌ 错误: 找不到配置文件 {args.config}")
        sys.exit(1)

    with open(args.config, 'r') as f:
        _config = yaml.safe_load(f)

    # -------- Config Normalization --------
    if 'max_epoch' not in _config:
        _config['max_epoch'] = _config.get('training_epoch', 50)

    if 'batch_size' not in _config:
        if 'train_loader' in _config and 'batch_size' in _config['train_loader']:
            _config['batch_size'] = _config['train_loader']['batch_size']
        else:
            _config['batch_size'] = 16

    if args.seed is not None:
        _config['seed'] = int(args.seed)

    method = args.method or _config.get("method", "compass")
    if method not in {"compass", "impute", "cmptstyle"}:
        raise ValueError(f"Unsupported --method {method}")
    _config["method"] = method
    if method == "impute":
        _config["missing_fill"] = "impute"
        _config["generator_mode"] = "pairwise"
        _config["lambda_vicreg"] = 0.0
        _config["lambda_proxy_cls"] = 0.0
        _config["mask_sources_during_training"] = True
    elif method == "cmptstyle":
        _config["missing_fill"] = "proxy"
        _config["generator_mode"] = "single_source"
        _config["lambda_vicreg"] = 0.0
        _config["lambda_proxy_cls"] = 0.0
        _config["mask_sources_during_training"] = True
        _config.setdefault("source_priority", "mmwave,rgb,depth,lidar")
    # --------------------------------------

    # -------------------------
    # 1) Device / GPUs
    # -------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：当前环境没有 CUDA。")

    gpus = args.gpus if args.gpus is not None else ",".join(str(i) for i in range(torch.cuda.device_count()))
    gpu_ids = [int(x) for x in gpus.split(",") if x.strip() != ""]
    if len(gpu_ids) == 0:
        raise ValueError("--gpus 解析为空，请传如 '2,3'")

    torch.cuda.set_device(gpu_ids[0])
    device = torch.device("cuda", gpu_ids[0])
    is_main = True

    # -------------------------
    # 2) 基础设置
    # -------------------------
    fix_seeds(_config.get("seed", 2023))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    # -------------------------
    # 3) 日志与保存路径
    # -------------------------
    if args.resume:
        save_dir = Path(args.resume)
    else:
        exp_prefix = "MMFi"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_folder_name = f"{exp_prefix}_{timestamp}_{method}_AlignedXRF55_DP_{gpus.replace(',', '-')}"
        base_output_dir = Path(os.environ.get("COMPASS_OUTPUT_DIR", "./output/MMFi"))
        save_dir = base_output_dir / exp_folder_name

    os.makedirs(save_dir, exist_ok=True)
    previous_final_eval = read_final_eval_results(save_dir / 'train.log') if args.resume else {}
    logger = get_logger(save_dir / 'train.log')
    logger.info(f"实验结果将保存至: {save_dir}")
    logger.info(f"使用设备: {device} | DataParallel GPUs={gpu_ids}")
    if previous_final_eval:
        logger.info(f"检测到已完成 final eval: {len(previous_final_eval)}/{len(FINAL_EVAL_COMBOS)} combos")

    # -------------------------
    # 4) 数据集
    # -------------------------
    environment_data_root = os.environ.get("MMFI_DATA_ROOT")
    data_root = environment_data_root or _config.get("data_root", "../data/MMFi")
    data_root_path = Path(data_root).expanduser()
    if not data_root_path.is_absolute():
        base_dir = PROJECT_DIR.parent if environment_data_root else Path(args.config).resolve().parent
        data_root_path = (base_dir / data_root_path).resolve()
    data_root = str(data_root_path)
    logger.info(f"正在初始化数据集 (Root: {data_root})...")

    _config['modality'] = ['rgb', 'depth', 'lidar', 'mmwave']
    trainset, valset = make_dataset(data_root, _config)

    logger.info("=" * 30)
    logger.info(f"📊 数据集原始统计:")
    logger.info(f"   - 训练集样本总数 (len(trainset)): {len(trainset)}")
    logger.info(f"   - 验证集样本总数 (len(valset)): {len(valset)}")
    logger.info("=" * 30)

    # -------------------------
    # 4.1) tiny_run（保留）
    # -------------------------
    tiny_run = bool(_config.get("tiny_run", False)) or bool(args.tiny_run)
    tiny_ratio = float(_config.get("tiny_ratio", 0.2))
    if args.tiny_ratio is not None:
        tiny_ratio = float(args.tiny_ratio)

    if tiny_run:
        rng = np.random.RandomState(_config.get("seed", 2023))

        def shrink(ds, ratio):
            n = len(ds)
            k = max(1, int(n * ratio))
            idx = rng.permutation(n)[:k].tolist()
            return Subset(ds, idx)

        trainset = shrink(trainset, tiny_ratio)
        valset = shrink(valset, tiny_ratio)
        logger.info(f"🧪 tiny_run ON: train={len(trainset)} | val={len(valset)} | ratio={tiny_ratio}")
    else:
        logger.info("✅ tiny_run OFF: 使用完整训练/验证集")

    # -------------------------
    # 5) DataLoader（✅ 对齐 XRF55：drop_last=True）
    # -------------------------
    batch_size = int(_config['batch_size'])
    num_workers = int(_config.get('num_workers', 4))

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        drop_last=True,            # ✅ 对齐 XRF55
        collate_fn=collate_fn_padd,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    valloader = DataLoader(
        valset,
        batch_size=int(_config.get('val_loader', {}).get('batch_size', batch_size)),
        shuffle=False,
        sampler=None,
        num_workers=int(_config.get('val_loader', {}).get('num_workers', num_workers)),
        collate_fn=collate_fn_padd,
        pin_memory=True,
        persistent_workers=True if int(_config.get('val_loader', {}).get('num_workers', num_workers)) > 0 else False,
        prefetch_factor=4 if int(_config.get('val_loader', {}).get('num_workers', num_workers)) > 0 else None,
    )

    logger.info(f"✅ Dataloader 就绪: batch={batch_size} | num_workers={num_workers} | drop_last_train=True")
    logger.info("⚠️ 当前版本：Depth 不做任何 resize/插值")

    # -------------------------
    # 6) 模型 + DataParallel
    # -------------------------
    feature_dim = 512
    class_num = 27
    dropout = float(_config.get('cmpt_dropout', 0.1))
    missing_fill = _config.get("missing_fill", "proxy")
    generator_mode = _config.get("generator_mode", "pairwise")
    source_priority = _config.get("source_priority", None)
    mask_sources_during_training = bool(_config.get("mask_sources_during_training", False))
    impute_hidden_dim = _config.get("impute_hidden_dim", None)
    impute_dropout = float(_config.get("impute_dropout", 0.3))

    logger.info(
        "正在构建 MMFi_CMPT_Net: "
        f"method={method} missing_fill={missing_fill} generator_mode={generator_mode} "
        f"mask_sources_during_training={mask_sources_during_training}"
    )
    model = MMFi_CMPT_Net(
        embed_dim=feature_dim,
        num_classes=class_num,
        dropout=dropout,
        missing_fill=missing_fill,
        impute_hidden_dim=impute_hidden_dim,
        impute_dropout=impute_dropout,
        generator_mode=generator_mode,
        source_priority=source_priority,
        mask_sources_during_training=mask_sources_during_training,
    ).to(device)

    for p in model.parameters():
        p.requires_grad = True

    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        logger.info(f"✅ 已启用 DataParallel: device_ids={gpu_ids}, output_device={gpu_ids[0]}")
    else:
        logger.info("✅ 单卡运行（未启用 DataParallel）")

    print_trainable_parameters(model, logger=logger)

    base_lr = float(_config['learning_rate'])
    weight_decay = float(_config.get('weight_decay', 1e-4))
    optim_type = _config.get('optim_type', _config.get('optimizer', 'adamw'))  # 兼容两种 key

    optimizer = get_optimizer(model, optim_type, base_lr, weight_decay)
    logger.info(f"Optimizer: {optim_type} | lr={base_lr:.6f} | weight_decay={weight_decay}")

    # -------------------------
    # 8) Scheduler（✅ 对齐 XRF55：iters_per_epoch = len(trainset)//batch_size）
    # -------------------------
    max_epoch = int(_config['max_epoch'])
    max_iters = int(_config.get("max_iters_per_epoch", 0))
    iters_per_epoch = max(1, len(trainset) // batch_size)
    if max_iters > 0:
        iters_per_epoch = min(iters_per_epoch, max_iters)

    scheduler = get_scheduler(
        _config.get('scheduler', 'warmuppolylr'),
        optimizer,
        int((max_epoch + 1) * iters_per_epoch),
        _config.get('power', 0.9),
        iters_per_epoch * _config.get('warmup', 5),
        _config.get('warmup_ratio', 0.1)
    )

    # -------------------------
    # 9) Loss & Train loop（✅ meter/log 对齐 XRF55）
    # -------------------------
    task_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    align_loss_fn = nn.MSELoss()

    best_performance = -1.0
    best_epoch = 0
    start_epoch = 0

    if args.resume:
        ckpt_path = Path(args.resume) / "checkpoint.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            raw_model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            best_performance = ckpt['best_performance']
            best_epoch = ckpt['best_epoch']
            logger.info(f"Resumed from epoch {ckpt['epoch'] + 1}, best_acc={best_performance:.4f}")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, training from scratch")

    eval_interval = int(_config.get('eval_interval', 10))      # ✅ 对齐 XRF55 默认 10
    simulate_missing = bool(_config.get('simulate_missing', True))
    cmpt_loss_weight = float(_config.get('cmpt_loss_weight', 0.2))
    drop_prob = float(_config.get('drop_prob', 0.7))           # ✅ 对齐 XRF55 默认 0.7
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}")
    lambda_recon = float(_config.get('lambda_recon', 0.5))
    use_impute = missing_fill == "impute"
    lambda_vicreg = float(_config.get('lambda_vicreg', 0.0))
    vicreg_inv = float(_config.get('vicreg_inv', 25.0))
    vicreg_var = float(_config.get('vicreg_var', 25.0))
    vicreg_cov = float(_config.get('vicreg_cov', 1.0))
    if lambda_vicreg > 0:
        logger.info(f"VICReg enabled: lambda={lambda_vicreg}, inv={vicreg_inv}, var={vicreg_var}, cov={vicreg_cov}")
    lambda_proxy_cls = float(_config.get('lambda_proxy_cls', 0.0))
    if lambda_proxy_cls > 0:
        logger.info(f'ProxyCls enabled: lambda={lambda_proxy_cls}')

    logger.info(f"开始训练 (FP32模式)，共 {max_epoch} 个 Epoch...")
    logger.info(f"鲁棒性训练模式 (Simulate Missing): {simulate_missing}")
    logger.info(f"验证频率: 每 {eval_interval} 轮验证一次")
    logger.info(f"drop_prob={drop_prob} | cmpt_loss_weight={cmpt_loss_weight} | lambda_recon={lambda_recon}")

    for epoch in range(start_epoch, max_epoch):
        model.train()

        # ✅ 对齐 XRF55：loss meter 按 iter 累加 item()
        total_loss_meter = 0.0
        task_loss_meter = 0.0
        align_loss_meter = 0.0
        recon_loss_meter = 0.0
        proxy_cls_loss_meter = 0.0

        correct = 0
        total = 0

        lr_now = _avg_lr(scheduler, optimizer)
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch + 1}] LR: {lr_now:.6f}")

        for it, batch in pbar:
            rgb, depth, mmwave, lidar, labels, _ = batch

            inputs = {}
            if rgb is not None and rgb.dim() > 1:
                inputs['rgb'] = rgb.to(device).float()
            if depth is not None:
                inputs['depth'] = depth.to(device).float()
            if mmwave is not None:
                inputs['mmwave'] = mmwave.to(device).float()
            if lidar is not None:
                inputs['lidar'] = lidar.to(device).float()

            labels = labels.to(device).long()

            current_mask = None
            if simulate_missing:
                current_mask = generate_random_mask(('rgb', 'depth', 'mmwave', 'lidar'), drop_prob=drop_prob)

            optimizer.zero_grad(set_to_none=True)  # ✅ 对齐 XRF55

            logits, fill_outputs, real_globals = model(inputs, missing_mask=current_mask)

            t_loss = task_loss_fn(logits, labels)

            a_loss = torch.tensor(0.0, device=device)
            recon_loss = torch.tensor(0.0, device=device)
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
                count = 0
                for key, proxy_feat in fill_outputs.items():
                    src, _, tgt = key.partition('_to_')
                    if tgt in real_globals:
                        a_loss += align_loss_fn(proxy_feat, real_globals[tgt])
                        count += 1
                if count > 0:
                    a_loss = a_loss / count

                loss = t_loss + cmpt_loss_weight * a_loss

            # VICReg cross-modal alignment on real modality globals
            vicreg_loss = torch.tensor(0.0, device=device)
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
                        vicreg_loss += vicreg_inv * inv_loss + vicreg_var * var_loss + vicreg_cov * cov_loss
                        pair_count += 1
                if pair_count > 0:
                    vicreg_loss = vicreg_loss / pair_count
            if lambda_vicreg > 0:
                loss = loss + lambda_vicreg * vicreg_loss

            proxy_cls_loss = torch.tensor(0.0, device=device)
            if lambda_proxy_cls > 0 and fill_outputs:
                raw_model = model.module if isinstance(model, nn.DataParallel) else model
                proxy_cls_count = 0
                for _, proxy_feat in fill_outputs.items():
                    proxy_logit = raw_model.decoder(proxy_feat)  # decoder expects [B,1,D]
                    proxy_cls_loss += task_loss_fn(proxy_logit, labels)
                    proxy_cls_count += 1
                if proxy_cls_count > 0:
                    proxy_cls_loss = proxy_cls_loss / proxy_cls_count
            if lambda_proxy_cls > 0:
                loss = loss + lambda_proxy_cls * proxy_cls_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss_meter += loss.item()
            task_loss_meter += t_loss.item()
            align_loss_meter += a_loss.item()
            recon_loss_meter += recon_loss.item()
            proxy_cls_loss_meter += proxy_cls_loss.item()

            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            desc = (
                f"Epoch: [{epoch + 1}] "
                f"Loss: {loss.item():.4f} (Task: {t_loss.item():.4f} Align: {a_loss.item():.4f}"
            )
            if use_impute:
                desc += f" Recon: {recon_loss.item():.4f}"
            if lambda_proxy_cls > 0:
                desc += f" ProxyCls: {proxy_cls_loss.item():.4f}"
            if lambda_vicreg > 0:
                desc += f" VICReg: {vicreg_loss.item():.4f}"
            desc += f") LR: {optimizer.param_groups[0]['lr']:.6f}"
            pbar.set_description(desc)

            # ✅ 严格对齐 XRF55：只跑 iters_per_epoch 次
            if (it + 1) >= iters_per_epoch:
                break

        avg_train_loss = total_loss_meter / max(1, iters_per_epoch)
        avg_task_loss = task_loss_meter / max(1, iters_per_epoch)
        avg_align_loss = align_loss_meter / max(1, iters_per_epoch)
        avg_recon_loss = recon_loss_meter / max(1, iters_per_epoch)
        avg_proxy_cls_loss = proxy_cls_loss_meter / max(1, iters_per_epoch)
        train_acc = correct / total if total > 0 else 0.0

        logger.info(
            f"[Epoch {epoch + 1}] "
            f"Train Loss: {avg_train_loss:.4f} | Task: {avg_task_loss:.4f} | Align: {avg_align_loss:.4f} | "
            f"Recon: {avg_recon_loss:.4f} | "
            f"ProxyCls: {avg_proxy_cls_loss:.4f} | "
            f"Train Acc: {train_acc:.4f}"
        )

        to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        # ---- 验证（✅ 对齐 XRF55：每 eval_interval 轮 or 最后一轮）
        if (epoch + 1) % eval_interval == 0 or (epoch + 1) == max_epoch:
            logger.info(f"正在验证 (Epoch {epoch + 1})...")
            val_scores = evaluate_mmfi(model, valloader, device, task_loss_fn, is_main=is_main)

            logger.info(
                f"Validation Results: Loss={val_scores['test_loss']:.4f}, Acc={val_scores['test_accuracy']:.4f}"
            )

            if val_scores['test_accuracy'] > best_performance:
                best_performance = val_scores['test_accuracy']
                best_epoch = epoch + 1

                torch.save(to_save, save_dir / "best_model.pth")

                logger.info(f"🌟 新最佳模型! Acc: {best_performance:.4f} @ Epoch {best_epoch}")
        else:
            logger.info(f"[Epoch {epoch + 1}] 训练 Loss: {avg_train_loss:.4f} (跳过验证)")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_performance': best_performance,
            'best_epoch': best_epoch,
        }
        torch.save(checkpoint, save_dir / "checkpoint.pth")

    logger.info(f"训练结束. Best Acc: {best_performance:.4f} @ Epoch {best_epoch}")
    logger.info(f"Best ckpt: {save_dir / 'best_model.pth'}")

    # --- 训练后 15 场景鲁棒性评估 ---
    best_ckpt = save_dir / "best_model.pth"
    if best_ckpt.exists():
        logger.info("开始 15 场景鲁棒性评估...")
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        best_state = torch.load(best_ckpt, map_location=device)
        clean_state = {k.replace('module.', ''): v for k, v in best_state.items()}
        raw_model.load_state_dict(clean_state, strict=True)

        from eval_all import eval_one_combo

        eval_loss_fn = nn.CrossEntropyLoss()
        logger.info(f"{'Combo':<25} | {'Loss':<10} | {'Acc':<10}")
        logger.info("-" * 50)

        completed_final_eval = dict(previous_final_eval)
        completed_final_eval.update(read_final_eval_results(save_dir / 'train.log'))
        for name, r_on, d_on, l_on, m_on in FINAL_EVAL_COMBOS:
            if name in completed_final_eval:
                loss, acc = completed_final_eval[name]
                logger.info(f"{name:<25} | {loss:.4f}     | {acc:.4f}     | cached")
                continue

            loss, acc = eval_one_combo(raw_model, valloader, device, eval_loss_fn, name, r_on, d_on, l_on, m_on)
            completed_final_eval[name] = (loss, acc)
            logger.info(f"{name:<25} | {loss:.4f}     | {acc:.4f}")
    else:
        logger.warning(f"未找到最佳模型: {best_ckpt}，跳过评估")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=str(PROJECT_DIR / "config.yaml"),
        help='Path to config file',
    )
    parser.add_argument('--gpus', type=str, default=None, help="使用哪些GPU，例如 '2,3'。默认 2,3")
    parser.add_argument('--method', type=str, default=None, choices=['compass', 'impute', 'cmptstyle'],
                        help='Training method: compass, impute, or cmptstyle')
    parser.add_argument('--seed', type=int, default=None, help='Override seed from config')
    parser.add_argument('--resume', type=str, default=None, help='Resume from a directory containing checkpoint.pth')
    parser.add_argument('--tiny_run', action='store_true', help='只用部分数据跑通流程（默认关闭）')
    parser.add_argument('--tiny_ratio', type=float, default=None, help='tiny_run 的数据比例，例如 0.2 表示 20%%')
    args = parser.parse_args()
    main(args)
