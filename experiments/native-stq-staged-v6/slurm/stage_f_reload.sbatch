#!/usr/bin/env bash
# Reload gate for the formal merged Stage-F checkpoint; no qrels or prediction.
#SBATCH --job-name=stq-native-staged-F-reload-v6
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/home/user46/logs/latent-register/%x-%j.out
#SBATCH --error=/mnt/home/user46/logs/latent-register/%x-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${NATIVE_ROOT:?}"
: "${CHECKPOINT_ROOT:?}"; : "${OUTPUT:?}"; : "${PYTHON_BIN:?}"
test -f "$CHECKPOINT_ROOT/COMPLETE"
mkdir -p "$(dirname "$OUTPUT")"
export PYTHONPATH="$PROJECT_ROOT/src:$NATIVE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# DeepSpeed's ZeRO-3 materializer probes CUDA_HOME during import. Recreate
# the job-local CUDA shim used by the training stages before importing it.
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp}"
CUDA_NVCC_ROOT="${CUDA_HOME:-/mnt/home/user46/miniforge3/pkgs/cuda-nvcc-12.4.131-0}"
if test ! -f "$CUDA_NVCC_ROOT/include/cuda_runtime.h"; then
  CUDA_CUDART_DEV_ROOT="${CUDA_CUDART_DEV_ROOT:-/mnt/home/user46/miniforge3/pkgs/cuda-cudart-dev-12.4.127-0}"
  CUDA_CCCL_ROOT="${CUDA_CCCL_ROOT:-/mnt/home/user46/miniforge3/pkgs/cuda-cccl-12.4.127-0}"
  test -x "$CUDA_NVCC_ROOT/bin/nvcc"
  test -f "$CUDA_CUDART_DEV_ROOT/include/cuda_runtime.h"
  test -f "$CUDA_CCCL_ROOT/include/nv/target"
  CUDA_HOME="$LOCAL_SCRATCH/cuda-home-latebound-${SLURM_JOB_ID}"
  mkdir -p "$CUDA_HOME/include"
  ln -sfn "$CUDA_NVCC_ROOT/bin" "$CUDA_HOME/bin"
  for header in "$CUDA_CUDART_DEV_ROOT/include"/*; do
    ln -sfn "$header" "$CUDA_HOME/include/$(basename "$header")"
  done
  for header_dir in nv cub thrust cuda; do
    ln -sfn "$CUDA_CCCL_ROOT/include/$header_dir" "$CUDA_HOME/include/$header_dir"
  done
  test -d "$CUDA_NVCC_ROOT/lib" && ln -sfn "$CUDA_NVCC_ROOT/lib" "$CUDA_HOME/lib"
fi
export CUDA_HOME
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
export PATH="$PYTHON_BIN_DIR:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LOCAL_SCRATCH/torch-extensions-latebound-reload-${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOCAL_SCRATCH/triton-latebound-reload-${SLURM_JOB_ID}}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/reload_checkpoint.py" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --model-path "$MODEL_ROOT" --native-root "$NATIVE_ROOT" \
  --phase full --reserved-pool-size 999 --device cuda:0 --output "$OUTPUT"
"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json, sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
assert p.get("passed") is True
assert p.get("phase") == "full"
assert p.get("lora_parameter_names_after_merge") == 0
assert p.get("full_backbone_trainable") is True
assert p.get("registration_optimizer_steps") == 0
PY
printf '%s\n' '{"kind":"stq_native_staged_formal_stage_f_reload","passed":true,"qrels_read":false}' > "${OUTPUT%.json}.event.json"
printf '%s\n' "$(sha256sum "$OUTPUT" | awk '{print $1}')" > "${OUTPUT%.json}.COMPLETE"
