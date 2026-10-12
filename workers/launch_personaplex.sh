#!/usr/bin/env bash
set -euo pipefail
gpu_index="${1:-1}"
worker_port="${2:-8999}"
worker_python="${PERSONAPLEX_PYTHON:-.venvs/personaplex/bin/python}"
if [[ ! -x "$worker_python" ]]; then
  echo "Set PERSONAPLEX_PYTHON to the PersonaPlex virtualenv's python; see docs/GPU_SERVER.md" >&2
  exit 1
fi
# The wrapper enforces --host even on upstream revisions that ignore it.
exec env CUDA_VISIBLE_DEVICES="$gpu_index" "$worker_python" -m workers.native_server \
  --host 127.0.0.1 --port "$worker_port" --static none \
  --hf-repo "${PERSONAPLEX_HF_REPO:-nvidia/personaplex-7b-v1}"
