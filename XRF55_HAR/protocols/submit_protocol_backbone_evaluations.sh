#!/bin/bash
#SBATCH --job-name=xfi_bb_eval
#SBATCH --output=XRF55_HAR/logs/slurm/%x_%A_%a.out
#SBATCH --error=XRF55_HAR/logs/slurm/%x_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=48G

set -e

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$REPO_DIR/XRF55_HAR"
cd "$SCRIPT_DIR"
mkdir -p logs/slurm

PROTOCOLS=(subject_split_21_9 subject_split_21_9 subject_split_21_9 scene_split scene_split scene_split)
MODALITIES=(mmwave wifi rfid mmwave wifi rfid)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
PROTOCOL="${PROTOCOLS[$INDEX]}"
MODALITY="${MODALITIES[$INDEX]}"
RAW_ROOT="${XRF55_RAW_ROOT:-/mnt/weka/rmkrtchyan/ws/data/XRF55}"
BACKBONE_ROOT="${COMPASS_PROTOCOL_BACKBONE_ROOT:-$REPO_DIR/outputs/XRF55_protocol_backbones}"

uv run python -m protocols.evaluate_protocol_backbone \
    --raw-root "$RAW_ROOT" \
    --backbone-root "$BACKBONE_ROOT" \
    --protocol "$PROTOCOL" \
    --modality "$MODALITY" \
    --seed 3407 \
    --workers 16
