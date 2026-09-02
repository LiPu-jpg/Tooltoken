#!/usr/bin/env bash
# Real two-step distributed Stage-F smoke after merge/remove of LoRA.
# It is intentionally a gate, not a paper result.
#SBATCH --job-name=stq-native-staged-F-smoke
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
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${MODEL_ROOT:?set MODEL_ROOT}"
: "${MANIFEST:?set MANIFEST}"
: "${NATIVE_ROOT:?set NATIVE_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${PYTHON_BIN:?set PYTHON_BIN}"
: "${WARMUP_ROOT:?set WARMUP_ROOT}"
test -f "$MANIFEST"
test -f "$PROJECT_ROOT/ds_z3_micro1_config.json"
test -f "$WARMUP_ROOT/COMPLETE"
test ! -e "$OUTPUT_ROOT"

mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$NATIVE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp}"
CUDA_NVCC_ROOT="${CUDA_HOME:-/mnt/home/user46/miniforge3/pkgs/cuda-nvcc-12.4.131-0}"
# The cluster packages nvcc and the CUDA runtime headers separately.  PyTorch's
# extension builder requires both under one CUDA_HOME, so create a job-local
# shim without modifying the shared environment.
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
  if test -d "$CUDA_NVCC_ROOT/lib"; then
    ln -sfn "$CUDA_NVCC_ROOT/lib" "$CUDA_HOME/lib"
  fi
fi
export CUDA_HOME
PYTHON_BIN_DIR="$(dirname "$PYTHON_BIN")"
export PATH="$PYTHON_BIN_DIR:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$LOCAL_SCRATCH/torch-extensions-latebound-sequence-sft-${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$LOCAL_SCRATCH/triton-latebound-sequence-sft-${SLURM_JOB_ID}}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
command -v ninja
command -v nvcc

# The training config uses ordinary AdamW and CPU-offloaded ZeRO-3.  Keep the probe on the same optimizer
# path; importing DeepSpeed FusedAdam would introduce an unrelated JIT CUDA
# extension dependency before the actual sequence-SFT process starts.
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
  --phase full \
  --manifest "$MANIFEST" \
  --model-path "$MODEL_ROOT" \
  --output-dir "$OUTPUT_ROOT/checkpoint" --warmup-dir "$WARMUP_ROOT" \
  --native-root "$NATIVE_ROOT" \
  --steps 5 \
  --epochs 1 \
  --batch-size 2 \
  --gradient-accumulation 1 \
  --max-length 128 \
  --max-document-length 64 \
  --reserved-pool-size 999 \
  --seed "${TRAIN_SEED:-17}" \
  --deepspeed-config "$PROJECT_ROOT/ds_z3_micro1_config.json" \
  --expected-world-size 4

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/reload_checkpoint.py" \
  --checkpoint-root "$OUTPUT_ROOT/checkpoint" \
  --model-path "$MODEL_ROOT" \
  --native-root "$NATIVE_ROOT" --phase full --reserved-pool-size 999 \
  --device cuda:0 \
  --output "$OUTPUT_ROOT/reload_audit.json"

FINISHED=$(date +%s)
printf '%s\n' "{\"kind\":\"native_sequence_sft_smoke\",\"steps\":5,\"wall_seconds\":$((FINISHED-STARTED)),\"passed\":true}" > "$OUTPUT_ROOT/smoke_summary.json"
sha256sum "$OUTPUT_ROOT/checkpoint/training_metadata.json" \
  "$OUTPUT_ROOT/checkpoint/reserved_ids.json" \
  "$OUTPUT_ROOT/reload_audit.json" \
  "$OUTPUT_ROOT/smoke_summary.json" > "$OUTPUT_ROOT/SHA256SUMS"
printf '%s\n' "$(sha256sum "$OUTPUT_ROOT/SHA256SUMS" | awk '{print $1}')" > "$OUTPUT_ROOT/COMPLETE"
