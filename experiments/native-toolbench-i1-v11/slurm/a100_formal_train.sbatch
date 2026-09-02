#!/usr/bin/env bash
# Native v11 four-A100 formal full-parameter sequence SFT.
# This job is a smoke only and is never a source of benchmark scores.
#SBATCH --job-name=lb-i1-v11-a100-4gpu-formal
#SBATCH --partition=q_amd_gpu_nvidia_A100_chenkh
#SBATCH --nodelist=gpu059
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/online1/ycsc_chenkh/hitici_11/logs/latent-register/%x-%j.out
#SBATCH --error=/online1/ycsc_chenkh/hitici_11/logs/latent-register/%x-%j.err

set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${MODEL_ROOT:?}"; : "${MANIFEST:?}"
: "${NATIVE_ROOT:?}"; : "${OUTPUT_ROOT:?}"; : "${PYTHON_BIN:?}"
test -f "$MANIFEST"
test -f "$PROJECT_ROOT/slurm/ds_z3_a100_config.json"
test ! -e "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$NATIVE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
COMPAT_INCLUDE="$PROJECT_ROOT/compat"
test -f "$COMPAT_INCLUDE/stdatomic.h"
export C_INCLUDE_PATH="$COMPAT_INCLUDE${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$COMPAT_INCLUDE${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SLURM_TMPDIR:-/tmp}/torch-extensions-lb-i1-v11-4gpu-${SLURM_JOB_ID}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/triton-lb-i1-v11-4gpu-${SLURM_JOB_ID}}"
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
STARTED=$(date +%s)
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
  "$PROJECT_ROOT/scripts/train.py" \
  --manifest "$MANIFEST" --model-path "$MODEL_ROOT" \
  --output-dir "$OUTPUT_ROOT/checkpoint" --native-root "$NATIVE_ROOT" \
  --steps 0 --epochs 1 --batch-size 1 --gradient-accumulation 1 \
  --max-length 128 --max-document-length 64 --reserved-pool-size 1000 \
  --seed "${TRAIN_SEED:-17}" --gradient-checkpointing \
  --deepspeed-config "$PROJECT_ROOT/slurm/ds_z3_a100_config.json" \
  --expected-world-size 4
"$PYTHON_BIN" "$PROJECT_ROOT/scripts/reload_checkpoint.py" \
  --checkpoint-root "$OUTPUT_ROOT/checkpoint" --model-path "$MODEL_ROOT" \
  --native-root "$PROJECT_ROOT" --device cuda:0 \
  --output "$OUTPUT_ROOT/reload_audit.json"
FINISHED=$(date +%s)
printf '%s\n' "{\"kind\":\"native_i1_v11_a100_4gpu_formal_training\",\"wall_seconds\":$((FINISHED-STARTED)),\"passed\":true,\"registration_optimizer_steps\":0}" > "$OUTPUT_ROOT/formal_summary.json"
sha256sum "$OUTPUT_ROOT/checkpoint/training_metadata.json" "$OUTPUT_ROOT/checkpoint/reserved_ids.json" "$OUTPUT_ROOT/reload_audit.json" "$OUTPUT_ROOT/formal_summary.json" > "$OUTPUT_ROOT/SHA256SUMS"
printf '%s\n' "$(sha256sum "$OUTPUT_ROOT/SHA256SUMS" | awk '{print $1}')" > "$OUTPUT_ROOT/COMPLETE"
