#!/bin/bash
#SBATCH --job-name=compass_xrf55
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
# Measured: ~45s/train epoch + 14s val every 5 epochs on one H100 => ~80 min for 100 epochs.
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G

set -e

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "COMPASS / XRF55 training"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
# Works whether submitted from the repo root or from inside XRF55_HAR.
if [ "$(basename "$REPO_DIR")" = "XRF55_HAR" ]; then
    REPO_DIR="$(dirname "$REPO_DIR")"
fi
cd "$REPO_DIR/XRF55_HAR"

# Fail fast with a clear message instead of crashing mid-training.
for f in \
    backbone_models/mmWave/mmwave_ResNet18.pt \
    backbone_models/WIFI/wifi_ResNet18.pt \
    backbone_models/RFID/RFID_ResNet18.pt \
; do
    if [ ! -f "$f" ]; then
        echo "ERROR: expected pretrained backbone missing: $f"
        exit 1
    fi
done

if [ ! -d "$REPO_DIR/data/XRF55/train_data" ]; then
    echo "ERROR: XRF55 split not found at $REPO_DIR/data/XRF55."
    echo "Run: python3 XRF55_HAR/split_xrf55_all_scenes.py \\"
    echo "        --src /mnt/weka/rmkrtchyan/ws/data/XRF55/part1 --dst data/XRF55"
    exit 1
fi

# Paper's reproduce command (README "Reproducing the main results"):
#   train.py with task_finetune_xrf55 seed=0 lambda_proxy_cls=0.5 drop_prob=0.7
# Everything else comes from XRF55_HAR/config.py BASE_CONFIG — do not add overrides
# unless you intend to deviate from the released setting.
SEED="${SEED:-0}"
LAMBDA_PROXY_CLS="${LAMBDA_PROXY_CLS:-0.5}"
DROP_PROB="${DROP_PROB:-0.7}"
MAX_EPOCH="${MAX_EPOCH:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"
# 0.0 = released behaviour (select the checkpoint on the test set, as the code ships).
# >0 = hold out that fraction of the 15,400 train samples for checkpoint selection
# and touch the test set only for the final report.
VAL_RATIO="${VAL_RATIO:-0.0}"

# save_dir is derived from wandb_exp_name + a second-resolution timestamp, so two
# jobs launched in the same second collide on one directory and clobber each
# other's checkpoint. Including the job id and seed keeps runs disjoint.
EXP_NAME="XRF55_s${SEED}_${SLURM_JOB_ID:-local}"

echo "SEED=$SEED LAMBDA_PROXY_CLS=$LAMBDA_PROXY_CLS DROP_PROB=$DROP_PROB MAX_EPOCH=$MAX_EPOCH VAL_RATIO=$VAL_RATIO"
echo "EXP_NAME=$EXP_NAME"

uv run python train.py with task_finetune_xrf55 \
    seed="$SEED" \
    lambda_proxy_cls="$LAMBDA_PROXY_CLS" \
    drop_prob="$DROP_PROB" \
    max_epoch="$MAX_EPOCH" \
    num_workers="$NUM_WORKERS" \
    val_ratio="$VAL_RATIO" \
    wandb_exp_name="$EXP_NAME"

echo "Done!"
