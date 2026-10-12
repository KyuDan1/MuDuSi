#!/usr/bin/env bash
#SBATCH --job-name=arena-video-package
#SBATCH --qos=own
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=runs/package-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
exec .venv/bin/python -m workers.package_showcase "$@"
