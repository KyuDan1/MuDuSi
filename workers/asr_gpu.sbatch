#!/usr/bin/env bash
#SBATCH --job-name=arena-asr-gpu
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=runs/asr-gpu-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec "${ASR_PYTHON:?Set ASR_PYTHON to a Whisper environment}" -m workers.transcribe_recording --model small --device cuda "$@"
