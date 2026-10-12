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
worker_gpu="${1:-0}"
worker_port="${2:-8998}"
worker_source="${3:?Raon source checkout}"
worker_model="${4:?SGLang bundle}"
shift 4
exec env CUDA_VISIBLE_DEVICES="$worker_gpu" "$worker_python" -m workers.raon_server \
  --port "$worker_port" --source "$worker_source" --model "$worker_model" "$@"
