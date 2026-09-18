#!/bin/bash
#SBATCH --export=ALL
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=research
#SBATCH --mem=64G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1
exec ../.venv/bin/python -u proposed_grid.py --group "${COMPASS_GRID_GROUP:?}" --stage "$1" --task "${SLURM_ARRAY_TASK_ID:?}"
