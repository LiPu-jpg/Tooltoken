#!/usr/bin/env bash
set -euo pipefail

project=${PROJECT_ROOT:-/mnt/home/user46/deployments/latebound-toolbench-i1-unseen-api-v4-20260831}
manifest=${MANIFEST:-$project/artifacts/manifest-api-disjoint-1k.json}
eval_root=${EVAL_ROOT:-$project/artifacts/i1-public}
native=${NATIVE_ROOT:-/mnt/home/user46/external/latent-register-frozen-20260823}
model=${MODEL_ROOT:-/mnt/home/user46/windy/models/Qwen3-8B}
run=${RUN_ROOT:-/mnt/home/user46/runs/native-i1-api-disjoint-v4-seed17}
python=${PYTHON_BIN:-/mnt/home/user46/.venvs/toolgen-l20/bin/python}
sbatch=${SBATCH_BIN:-/mnt/data/hpc/slurm/23.02/bin/sbatch}

test -x "$python"; test -x "$sbatch"; test -d "$model"; test -d "$native"
test -f "$manifest"; test -f "$eval_root/evaluation_manifest.json"; test ! -e "$run"
mkdir -p /mnt/home/user46/logs/latent-register
common="ALL,PROJECT_ROOT=$project,MANIFEST=$manifest,EVAL_ROOT=$eval_root,NATIVE_ROOT=$native,PYTHON_BIN=$python"
preflight=$($sbatch --parsable \
  --export="$common,OUTPUT_ROOT=$run/preflight" \
  "$project/slurm/l20_preflight.sbatch")
smoke=$($sbatch --parsable --dependency="afterok:$preflight" \
  --export="$common,MODEL_ROOT=$model,PREFLIGHT_ROOT=$run/preflight,OUTPUT_ROOT=$run/smoke" \
  "$project/slurm/l20_smoke.sbatch")
printf 'preflight=%s\nsmoke=%s\n' "$preflight" "$smoke"
