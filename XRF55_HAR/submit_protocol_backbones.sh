#!/bin/bash
#SBATCH --job-name=xfi_split_bb
#SBATCH --output=XRF55_HAR/logs/slurm/%x_%A_%a.out
#SBATCH --error=XRF55_HAR/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-5
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
MODALITIES=(mmwave wifi rfid)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PROTOCOL_INDEX=$((INDEX / 3))
MODALITY_INDEX=$((INDEX % 3))
PROTOCOL="${PROTOCOLS[$PROTOCOL_INDEX]}"
MODALITY="${MODALITIES[$MODALITY_INDEX]}"
SEED="${SEED:-3407}"
EPOCHS="${EPOCHS:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
RAW_ROOT="${XRF55_RAW_ROOT:-/mnt/weka/rmkrtchyan/ws/data/XRF55}"
OUTPUT_ROOT="${COMPASS_PROTOCOL_BACKBONE_ROOT:-$REPO_DIR/outputs/XRF55_protocol_backbones}"

EXTRA_ARGS=()
if [ -n "$MAX_TRAIN_BATCHES" ]; then
    EXTRA_ARGS+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi

echo "X-Fi backbone reproduction protocol=$PROTOCOL modality=$MODALITY seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv

uv run python pretrain_protocol_backbone.py \
    --raw-root "$RAW_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --protocol "$PROTOCOL" \
    --modality "$MODALITY" \
    --epochs "$EPOCHS" \
    --workers 16 \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"
