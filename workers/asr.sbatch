#!/usr/bin/env bash
#SBATCH --job-name=arena-asr
#SBATCH --qos=own
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=runs/asr-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec "${ASR_PYTHON:?Set ASR_PYTHON to an environment with openai-whisper}" -m workers.transcribe_recording "$@"
