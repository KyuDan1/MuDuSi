#!/usr/bin/env bash
#SBATCH --job-name=mudusi-matrix
#SBATCH --qos=extra
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=512G
#SBATCH --time=00:35:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=runs/matrix-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
if [[ -n "${MUDUSI_ENV_FILE:-}" ]]; then
  source "$MUDUSI_ENV_FILE"
fi
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export TORCHINDUCTOR_COMPILE_THREADS=2
source .venv/bin/activate
exec python -m workers.run_matrix "$@"
