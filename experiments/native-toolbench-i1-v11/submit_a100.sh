#!/usr/bin/env bash
set -euo pipefail

# Deliberate gate: the caller must opt in after a fresh scheduler audit.
test "${ALLOW_A100_SUBMIT:-0}" = 1 || {
  echo "refusing A100 submission without ALLOW_A100_SUBMIT=1 after a fresh resource audit" >&2
  exit 2
}

project=${PROJECT_ROOT:?set remote v7 deployment root}
manifest=${MANIFEST:?set fixed API-disjoint manifest}
native=${NATIVE_ROOT:?set frozen Native source root}
model=${MODEL_ROOT:?set Qwen3-8B model root}
run=${RUN_ROOT:?set a new v7 smoke run root}
python=${PYTHON_BIN:?set cluster Python}
sbatch=${SBATCH_BIN:-/opt/hpc/slurm/21.08.6/bin/sbatch}

test -x "$python"
test -x "$sbatch"
test -f "$manifest"
test -d "$native"
test -d "$model"
test ! -e "$run"
mkdir -p "$(dirname "$run")"
common="ALL,PROJECT_ROOT=$project,MANIFEST=$manifest,NATIVE_ROOT=$native,MODEL_ROOT=$model,PYTHON_BIN=$python"
job=$($sbatch --parsable --nodelist=gpu059 \
  --export="$common,OUTPUT_ROOT=$run" \
  "$project/slurm/a100_throughput.sbatch")
printf 'throughput=%s\nnode_request=gpu059\ngpu_request=4\nrun=%s\n' "$job" "$run"
