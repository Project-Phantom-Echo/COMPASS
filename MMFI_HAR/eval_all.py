import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from dataset.mmfi_dataset import make_dataset
from util import collate_fn_padd
from shared.utils.utils import fix_seeds

# 你的模型
from models.cmpt_model_mmfi import MMFi_CMPT_Net

MODEL_MODALITIES = ["rgb", "depth", "mmwave", "lidar"]
PROJECT_DIR = Path(__file__).resolve().parent


@torch.no_grad()
def eval_one_combo(model, loader, device, loss_fn, combo_name, use_rgb, use_depth, use_lidar, use_mmwave):
    """
    针对 MMFi_CMPT_Net 的特定验证逻辑：
    1. 严格检查数据完整性：只有当 batch 中包含所有"要求的模态"时，才进行验证。
    2. 显式构建 missing_mask：强制告诉模型哪些模态是"存在"的(1)，哪些是"缺失"的(0)。
    3. 模型会自动根据 mask=0 的部分，用 mask=1 的部分生成的 Proxy 进行填补。
    """
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc=f"Eval {combo_name}", leave=False):
        rgb, depth, mmwave, lidar, labels, _ = batch

        labels = labels.to(device).long()

        # --- 1. 数据完整性检查 (Data Integrity Check) ---
        # 如果当前组合要求用某模态，但这个 batch 的数据里没有有效数据，
        # 那就没法测这个组合，必须跳过这个 batch。
        if use_rgb and (rgb is None or rgb.dim() <= 1):
            continue
        if use_depth and depth is None:
            continue
        if use_lidar and lidar is None:
            continue
        if use_mmwave and mmwave is None:
            continue

        # --- 2. 构建 missing_mask (Explicit Mask Construction) ---
        # 这里的逻辑是：只要你的组合配置里要求用(use_xxx=True)，mask就设为1。
        # 不要求的设为0。这会触发模型内部的 Proxy 生成机制来填补0的部分。
        current_mask = {
            "rgb": 1 if use_rgb else 0,
            "depth": 1 if use_depth else 0,
            "lidar": 1 if use_lidar else 0,
            "mmwave": 1 if use_mmwave else 0
        }

        # --- 3. 构建 inputs ---
        # 只把 mask=1 (即当前组合要求) 的数据喂进去
        inputs = {}
        if use_rgb:
            inputs["rgb"] = rgb.to(device).float()
        if use_depth:
            inputs["depth"] = depth.to(device).float()
        if use_lidar:
            inputs["lidar"] = lidar.to(device).float()
        if use_mmwave:
            inputs["mmwave"] = mmwave.to(device).float()

        # 双重保险：如果 inputs 空了(理论上上面check过不会空)，跳过
        if len(inputs) == 0:
            continue

        # --- 4. Forward ---
        # 传入 missing_mask，模型会根据它决定是用 Real Feature 还是 Proxy
        logits, _, _ = model(inputs, missing_mask=current_mask)

        loss = loss_fn(logits, labels)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == labels).sum().item()
        bs = labels.size(0)
        total += bs
        loss_sum += loss.item() * bs

    avg_loss = loss_sum / total if total > 0 else 0.0
    avg_acc = correct / total if total > 0 else 0.0

    return avg_loss, avg_acc


def main():
    parser = argparse.ArgumentParser("MMFi CMPT test (15 combos with RGB)")
    parser.add_argument("--config", type=str, default=str(PROJECT_DIR / "config.yaml"))
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.environ.get(
            "MMFI_DATA_ROOT",
            str(PROJECT_DIR.parent / "data" / "MMFi"),
        ),
        help="MMFi dataset root"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(PROJECT_DIR.parent / "outputs" / "MMFi" / "best_model.pth"),
        help="path to best_model.pth"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    dataset_path = Path(args.dataset).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_DIR.parent / dataset_path
    args.dataset = str(dataset_path.resolve())

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- load config ----------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    fix_seeds(cfg.get("seed", cfg.get("init_rand_seed", 2023)))

    cfg["modality"] = ["rgb", "depth", "lidar", "mmwave"]

    # ---------- dataset ----------
    trainset, valset = make_dataset(args.dataset, cfg)

    evalset = valset

    from torch.utils.data import DataLoader
    batch_size = int(cfg.get("val_loader", {}).get("batch_size", cfg.get("batch_size", 32)))
    num_workers = int(cfg.get("val_loader", {}).get("num_workers", cfg.get("num_workers", 4)))

    loader = DataLoader(
        evalset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_padd,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    print(f"Eval set size={len(evalset)}, batch_size={batch_size}, num_workers={num_workers}")

    class_num = 27
    dropout = float(cfg.get("cmpt_dropout", 0.1))

    print(">>> before build model")
    model = MMFi_CMPT_Net(
        num_classes=class_num,
        modalities=MODEL_MODALITIES,
        embed_dim=512,
        dropout=dropout,
        n_heads=4
    ).to(device)

    print(">>> before torch.load")
    state = torch.load(args.ckpt, map_location="cpu")

    # 兼容：有些保存出来是 {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # 兼容：DDP 保存出来可能带 "module."
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[len("module."):]] = v
            else:
                new_state[k] = v
        state = new_state

    print(">>> before load_state_dict")
    model.load_state_dict(state, strict=True)
    print(">>> after load_state_dict")
    print("Loaded ckpt:", args.ckpt)

    loss_fn = nn.CrossEntropyLoss()

    # ---------- 15 combos (with RGB) ----------
    combos = [
        # 单模态
        ("RGB Only", True, False, False, False),
        ("Depth Only", False, True, False, False),
        ("Lidar Only", False, False, True, False),
        ("MMWave Only", False, False, False, True),

        # 双模态
        ("RGB + Depth", True, True, False, False),
        ("RGB + Lidar", True, False, True, False),
        ("RGB + MMWave", True, False, False, True),
        ("Depth + Lidar", False, True, True, False),
        ("Depth + MMWave", False, True, False, True),
        ("Lidar + MMWave", False, False, True, True),

        # 三模态
        ("RGB + Depth + Lidar", True, True, True, False),
        ("RGB + Depth + MMWave", True, True, False, True),
        ("RGB + Lidar + MMWave", True, False, True, True),
        ("Depth + Lidar + MMWave", False, True, True, True),

        # 全模态
        ("All (R+D+L+M)", True, True, True, True),
    ]

    print("\n===== 15 Combinations Evaluation (Missing Modalities are Imputed by Proxies) =====")
    print(f"{'Combo Name':<20} | {'Loss':<10} | {'Acc':<10}")
    print("-" * 46)

    for name, r_on, d_on, l_on, m_on in combos:
        loss, acc = eval_one_combo(model, loader, device, loss_fn, name, r_on, d_on, l_on, m_on)
        print(f"{name:<20} | {loss:.4f}     | {acc:.4f}")


if __name__ == "__main__":
    main()

