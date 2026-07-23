import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import pprint
import time
from pathlib import Path

# --- 1. 路径修复 ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'backbone_models'))

# --- 2. 引入自定义模块 ---
from dataset.xrf55_dataset import XRF55_Dataset
from models.cmpt_model_xrf55 import XRF55_CMPT_Net
from Encoders import Encoder
from Extractor import mmwave_feature_extractor, wifi_feature_extractor, rfid_feature_extractor
from misc import collate_fn_padd


PROJECT_DIR = Path(__file__).resolve().parent

# 定义 7 种测试场景
SCENARIOS = {
    # --- 单模态 ---
    'Only mmWave': {'mmwave': 1, 'wifi': 0, 'rfid': 0},
    'Only WiFi': {'mmwave': 0, 'wifi': 1, 'rfid': 0},
    'Only RFID': {'mmwave': 0, 'wifi': 0, 'rfid': 1},

    # --- 双模态 ---
    'mmWave + WiFi': {'mmwave': 1, 'wifi': 1, 'rfid': 0},
    'mmWave + RFID': {'mmwave': 1, 'wifi': 0, 'rfid': 1},
    'WiFi + RFID': {'mmwave': 0, 'wifi': 1, 'rfid': 1},

    # --- 全模态 ---
    'All Modalities': {'mmwave': 1, 'wifi': 1, 'rfid': 1}
}


def load_custom_encoders(device):
    print("正在构建 Backbone...")
    base_path = PROJECT_DIR / "backbone_models"

    # 加载空壳或预训练权重来初始化结构
    try:
        mmwave_model = torch.load(base_path / "mmWave" / "mmwave_ResNet18.pt", map_location="cpu")
        wifi_model = torch.load(base_path / "WIFI" / "wifi_ResNet18.pt", map_location="cpu")
        rfid_model = torch.load(base_path / "RFID" / "RFID_ResNet18.pt", map_location="cpu")
    except FileNotFoundError:
        print("错误：找不到 Backbone 权重文件，请检查路径。")
        raise

    mmwave_extractor = mmwave_feature_extractor(mmwave_model).eval()
    wifi_extractor = wifi_feature_extractor(wifi_model).eval()
    rfid_extractor = rfid_feature_extractor(rfid_model).eval()

    encode_info = [(512, 512, 32), (512, 512, 4), (512, 512, 5)]
    task_encoders = nn.ModuleDict()
    task_encoders['mmwave'] = Encoder(0, mmwave_extractor, encode_info[0])
    task_encoders['wifi'] = Encoder(1, wifi_extractor, encode_info[1])
    task_encoders['rfid'] = Encoder(2, rfid_extractor, encode_info[2])

    return task_encoders


def evaluate_robustness(model, dataloader, device):
    model.eval()

    # 初始化统计器：每种场景都有独立的统计
    results = {name: {'correct': 0, 'total': 0, 'loss': 0.0} for name in SCENARIOS.keys()}
    loss_fn = nn.CrossEntropyLoss()

    print(f"开始鲁棒性测试，共 {len(dataloader)} 个 Batch...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            mmwave_data, wifi_data, rfid_data, labels = batch

            inputs = {
                'mmwave': mmwave_data.to(device),
                'wifi': wifi_data.to(device),
                'rfid': rfid_data.to(device)
            }
            labels = labels.to(device).long()
            batch_size = labels.size(0)

            # --- 核心循环：针对同一个 Batch，测试 7 种 Mask ---
            for name, mask in SCENARIOS.items():
                # 1. 传入当前场景的 mask
                logits, _, _ = model(inputs, missing_mask=mask)

                # 2. 计算 Loss 和 准确率
                loss = loss_fn(logits, labels)
                _, predicted = torch.max(logits, 1)
                correct = (predicted == labels).sum().item()

                # 3. 记录结果
                results[name]['loss'] += loss.item() * batch_size
                results[name]['correct'] += correct
                results[name]['total'] += batch_size

    return results


def print_results(results):
    print("\n" + "=" * 65)
    print(f"{'Scenario':<20} | {'Accuracy':<10} | {'Avg Loss':<10}")
    print("-" * 65)

    # 按场景顺序打印
    # 先单模态，再双模态，再全模态
    ordered_keys = [
        'Only mmWave', 'Only WiFi', 'Only RFID',
        'mmWave + WiFi', 'mmWave + RFID', 'WiFi + RFID',
        'All Modalities'
    ]

    for name in ordered_keys:
        stats = results[name]
        acc = stats['correct'] / stats['total'] * 100
        avg_loss = stats['loss'] / stats['total']
        print(f"{name:<20} | {acc:.2f}%     | {avg_loss:.4f}")

    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser()
    # 默认使用刚才训练好的最佳权重
    default_ckpt = str(PROJECT_DIR.parent / "outputs" / "XRF55" / "best_model.pth")

    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.environ.get(
            "XRF55_DATA_ROOT",
            str(PROJECT_DIR.parent / "data" / "XRF55"),
        ),
    )
    parser.add_argument('--checkpoint', type=str, default=default_ckpt)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--class_num', type=int, default=55)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scene', type=str, default='all')
    parser.add_argument('--fusion_type', type=str, default='sum', choices=['sum', 'concat', 'cross_attn'])
    parser.add_argument('--proxy_agg', type=str, default='uniform', choices=['uniform', 'confidence'])
    parser.add_argument('--conf_temperature', type=float, default=1.0)
    parser.add_argument('--no_proxy', action='store_true')
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR.parent / data_dir
    args.data_dir = str(data_dir.resolve())

    device = torch.device(args.device)

    # 1. 模型初始化
    proj_dim = 32
    embed_dim = 512

    task_encoders = load_custom_encoders(device)

    # task_decoder 你这里按你训练代码的套路写：([embed_dim, class_num], 'classification')
    task_decoder_config = ([embed_dim, args.class_num], 'classification')

    cmpt_dropout = 0.1
    model = XRF55_CMPT_Net(
        task_encoders=task_encoders,
        task_decoder=task_decoder_config,
        proj_dim=proj_dim,
        embed_dim=embed_dim,
        dropout=cmpt_dropout,
        fusion_type=args.fusion_type,
        proxy_agg=args.proxy_agg,
        conf_temperature=args.conf_temperature,
        no_proxy=args.no_proxy
    )

    # 2. 加载权重
    print(f"正在加载权重: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError("权重文件不存在！")

    state_dict = torch.load(args.checkpoint, map_location=device)
    # 只剥离 DataParallel 的前导 'module.'。
    new_state_dict = {(k[len('module.'):] if k.startswith('module.') else k): v
                      for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)

    # 3. 加载测试集
    testset = XRF55_Dataset(root_dir=args.data_dir, split='test', scene=args.scene)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                            collate_fn=collate_fn_padd)

    # 4. 运行 7 种情况测试
    start_time = time.time()
    results = evaluate_robustness(model, testloader, device)
    end_time = time.time()

    # 5. 打印报表
    print_results(results)
    print(f"Total Evaluation Time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
