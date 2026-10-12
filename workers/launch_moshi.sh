#!/usr/bin/env bash
set -euo pipefail
gpu_index="${1:-0}"
worker_port="${2:-8998}"
worker_python="${MOSHI_PYTHON:-.venvs/moshi/bin/python}"
if [[ ! -x "$worker_python" ]]; then
  echo "Set MOSHI_PYTHON to the Moshi virtualenv's python; see docs/GPU_SERVER.md" >&2
  exit 1
fi
exec env CUDA_VISIBLE_DEVICES="$gpu_index" "$worker_python" -m moshi.server \
  --host 127.0.0.1 --port "$worker_port" --static none \
  --hf-repo "${MOSHI_HF_REPO:-kyutai/moshiko-pytorch-bf16}"
