#!/bin/bash
#SBATCH --job-name=compass_gen
#SBATCH --output=XRF55_HAR/logs/slurm/%x_%A_%a.out
#SBATCH --error=XRF55_HAR/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G

set -e

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$REPO_DIR/XRF55_HAR"
cd "$SCRIPT_DIR"
mkdir -p logs/slurm

PROTOCOLS=(subject_split_21_9 scene_split)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PROTOCOL="${PROTOCOLS[$INDEX]}"
SEED="${SEED:-3407}"
RAW_ROOT="${XRF55_RAW_ROOT:-/mnt/weka/rmkrtchyan/ws/data/XRF55}"
OUTPUT_ROOT="${COMPASS_OUTPUT_DIR:-$REPO_DIR/outputs/XRF55_generalization}"
BACKBONE_ROOT="${COMPASS_PROTOCOL_BACKBONE_ROOT:-$REPO_DIR/outputs/XRF55_protocol_backbones}"
BACKBONE_DIR="$BACKBONE_ROOT/$PROTOCOL/seed${SEED}"
EXP_NAME="COMPASS_${PROTOCOL}_s${SEED}_${SLURM_ARRAY_JOB_ID:-local}_${INDEX}"

for path in \
    "$RAW_ROOT/part1/Scene1/Scene1" \
    "$RAW_ROOT/part2/Scene1_part2" \
    "$BACKBONE_DIR/mmWave/mmwave_ResNet18.pt" \
    "$BACKBONE_DIR/WIFI/wifi_ResNet18.pt" \
    "$BACKBONE_DIR/RFID/RFID_ResNet18.pt"
do
    if [ ! -e "$path" ]; then
        echo "ERROR: required path missing: $path"
        exit 1
    fi
done

echo "COMPASS XRF55 protocol=$PROTOCOL seed=$SEED"
echo "Raw data: $RAW_ROOT"
echo "Split-clean backbones: $BACKBONE_DIR"
echo "Outputs: $OUTPUT_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv

XRF55_RAW_ROOT="$RAW_ROOT" COMPASS_OUTPUT_DIR="$OUTPUT_ROOT" \
COMPASS_BACKBONE_DIR="$BACKBONE_DIR" \
uv run python train.py with task_finetune_xrf55 \
    protocol="$PROTOCOL" \
    seed="$SEED" \
    final_epoch_only=true \
    val_ratio=0.0 \
    lambda_proxy_cls=0.5 \
    drop_prob=0.7 \
    max_epoch=100 \
    num_workers=16 \
    backbone_dir="$BACKBONE_DIR" \
    wandb_exp_name="$EXP_NAME"
