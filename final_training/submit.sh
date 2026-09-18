#!/bin/bash
#SBATCH --job-name=compass-final-rotations
#SBATCH --partition=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --export=ALL
set -euo pipefail
export OMP_NUM_THREADS=1
exec "$2" -u "$1/launcher.py" --campaign "$1" --task "${SLURM_ARRAY_TASK_ID:?}"
