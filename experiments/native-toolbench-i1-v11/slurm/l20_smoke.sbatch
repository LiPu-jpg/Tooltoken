#!/usr/bin/env bash
#SBATCH --job-name=lb-i1-smoke
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/user46/logs/latent-register/%x-%j.out
#SBATCH --error=/mnt/home/user46/logs/latent-register/%x-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${MANIFEST:?}"; : "${NATIVE_ROOT:?}"; : "${PREFLIGHT_ROOT:?}"; : "${OUTPUT_ROOT:?}"; : "${PYTHON_BIN:?}"
test -f "$PREFLIGHT_ROOT/COMPLETE"; test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$NATIVE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_P2P_DISABLE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp}"
CUDA_NVCC_ROOT="${CUDA_HOME:-/mnt/home/user46/miniforge3/pkgs/cuda-nvcc-12.4.131-0}"
if test ! -f "$CUDA_NVCC_ROOT/include/cuda_runtime.h"; then
  CUDA_CUDART_DEV_ROOT="${CUDA_CUDART_DEV_ROOT:-/mnt/home/user46/miniforge3/pkgs/cuda-cudart-dev-12.4.127-0}"
  CUDA_CCCL_ROOT="${CUDA_CCCL_ROOT:-/mnt/home/user46/miniforge3/pkgs/cuda-cccl-12.4.127-0}"
  test -x "$CUDA_NVCC_ROOT/bin/nvcc"
  test -f "$CUDA_CUDART_DEV_ROOT/include/cuda_runtime.h"
  test -d "$CUDA_CCCL_ROOT/include/nv"
  CUDA_HOME="$LOCAL_SCRATCH/cuda-home-lb-i1-${SLURM_JOB_ID}"
  mkdir -p "$CUDA_HOME/include"
  ln -sfn "$CUDA_NVCC_ROOT/bin" "$CUDA_HOME/bin"
  for header in "$CUDA_CUDART_DEV_ROOT/include"/*; do
    ln -sfn "$header" "$CUDA_HOME/include/$(basename "$header")"
  done
  for header_dir in nv cub thrust cuda; do
    ln -sfn "$CUDA_CCCL_ROOT/include/$header_dir" "$CUDA_HOME/include/$header_dir"
  done
fi
export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LOCAL_SCRATCH/torch-extensions-lb-i1-${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-lb-i1-${SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
command -v nvcc

# Verify the ordinary PyTorch AdamW path before launching distributed training.
"$PYTHON_BIN" - <<'PY'
import torch

parameter = torch.nn.Parameter(torch.zeros(1, device="cuda"))
optimizer = torch.optim.AdamW([parameter], lr=1e-3)
parameter.grad = torch.ones_like(parameter)
optimizer.step()
print("adamw_single_process_probe=PASS", flush=True)
PY

STARTED=$(date +%s)
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$PROJECT_ROOT/scripts/train.py" \
  --manifest "$MANIFEST" --model-path "$MODEL_ROOT" --output-dir "$OUTPUT_ROOT/checkpoint" \
  --native-root "$NATIVE_ROOT" --steps 2 --epochs 1 --batch-size 2 --gradient-accumulation 1 \
  --max-length 128 --max-document-length 64 --reserved-pool-size 1000 --seed "${TRAIN_SEED:-17}" \
  --deepspeed-config "$PROJECT_ROOT/slurm/ds_z3_i1_smoke.json" --expected-world-size 4
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/reload_checkpoint.py" \
  --checkpoint-root "$OUTPUT_ROOT/checkpoint" --model-path "$MODEL_ROOT" \
  --native-root "$PROJECT_ROOT" --device cuda:0 --output "$OUTPUT_ROOT/reload_audit.json"
FINISHED=$(date +%s)
printf '%s\n' "{\"kind\":\"native_i1_smoke\",\"steps\":2,\"wall_seconds\":$((FINISHED-STARTED)),\"passed\":true}" > "$OUTPUT_ROOT/smoke_summary.json"
sha256sum "$OUTPUT_ROOT/checkpoint/training_metadata.json" "$OUTPUT_ROOT/checkpoint/reserved_ids.json" "$OUTPUT_ROOT/reload_audit.json" "$OUTPUT_ROOT/smoke_summary.json" > "$OUTPUT_ROOT/SHA256SUMS"
printf '%s\n' "$(sha256sum "$OUTPUT_ROOT/SHA256SUMS" | awk '{print $1}')" > "$OUTPUT_ROOT/COMPLETE"
