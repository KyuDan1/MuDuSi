#!/usr/bin/env bash
#SBATCH --job-name=duplex-pair
#SBATCH --qos=own
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=runs/pair-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
source .venv/bin/activate
exec python -m workers.run_pair "$@"
