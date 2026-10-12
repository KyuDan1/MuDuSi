#!/usr/bin/env bash
#SBATCH --job-name=arena-matrix-video
#SBATCH --qos=own
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --container=python:3.11
#SBATCH --output=runs/render-matrix-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export LD_LIBRARY_PATH="$PWD/work/browser-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
exec "${ARENA_NODE:?Set ARENA_NODE to Node.js 20+}" workers/render_matrix.cjs "$@"
