#!/usr/bin/env bash
#SBATCH --job-name=arena-film-edit
#SBATCH --qos=own
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=runs/film-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python -m workers.edit_film "$@"
