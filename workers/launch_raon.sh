#!/usr/bin/env bash
set -euo pipefail
worker_python="${RAON_PYTHON:-.venvs/raon/bin/python}"
base_prefix=$("$worker_python" -c 'import sys; print(sys.base_prefix)')
worker_site=$("$worker_python" -c 'import site; print(site.getsitepackages()[0])')
export CUDA_HOME="$base_prefix"
export LD_LIBRARY_PATH="$worker_site/nvidia/cu13/lib:$base_prefix/lib:${LD_LIBRARY_PATH:-}"
# SGLang's JIT assumes CUDA libraries live in lib64; Conda stores them in lib.
export LIBRARY_PATH="$base_prefix/lib:${LIBRARY_PATH:-}"
export SGLANG_ENABLE_JIT_DEEPGEMM=0
exec env CUDA_VISIBLE_DEVICES="${1:-0}" "$worker_python" -m workers.raon_server \
  --port "${2:-8998}" --source "${3:?Raon source checkout}" --model "${4:?SGLang bundle}"
