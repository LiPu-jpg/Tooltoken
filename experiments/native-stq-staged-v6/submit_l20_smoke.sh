#!/usr/bin/env bash
set -euo pipefail

SBATCH=${SBATCH:-/mnt/data/hpc/slurm/23.02/bin/sbatch}
ROOT=${PROJECT_ROOT:-/mnt/home/user46/deployments/latebound-stq-native-staged-v5-20260902}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/mnt/home/user46/inputs/simpletoolquestions-unseen-manifest-v1}
MODEL_ROOT=${MODEL_ROOT:-/mnt/home/user46/windy/models/Qwen3-8B}
NATIVE_ROOT=${NATIVE_ROOT:-/mnt/home/user46/latent-register}
PYTHON_BIN=${PYTHON_BIN:-/mnt/home/user46/.venvs/toolgen-l20/bin/python}
RUN_ROOT=${RUN_ROOT:-/mnt/home/user46/runs/stq-native-staged-v5-smoke-20260902}
MANIFEST=${MANIFEST:-$ROOT/manifest.json}

test -x "$SBATCH"; test -f "$ROOT/slurm/preflight.sbatch"; test -f "$ROOT/slurm/stage_l_smoke.sbatch"
test -f "$MANIFEST"; test -d "$ARTIFACT_ROOT"; test -d "$MODEL_ROOT"; test -d "$NATIVE_ROOT"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"
preflight=$($SBATCH --parsable --export="ALL,PROJECT_ROOT=$ROOT,MANIFEST=$MANIFEST,ARTIFACT_ROOT=$ARTIFACT_ROOT,PYTHON_BIN=$PYTHON_BIN,OUTPUT_ROOT=$RUN_ROOT/preflight" "$ROOT/slurm/preflight.sbatch")
stage_l=$($SBATCH --parsable --dependency="afterok:$preflight" --export="ALL,PROJECT_ROOT=$ROOT,MODEL_ROOT=$MODEL_ROOT,MANIFEST=$MANIFEST,NATIVE_ROOT=$NATIVE_ROOT,PYTHON_BIN=$PYTHON_BIN,OUTPUT_ROOT=$RUN_ROOT/stage-l-smoke,TRAIN_SEED=${TRAIN_SEED:-17}" "$ROOT/slurm/stage_l_smoke.sbatch")
python3 - "$RUN_ROOT/submitted.json" "$preflight" "$stage_l" "$ROOT" "$RUN_ROOT" "$MANIFEST" "$ARTIFACT_ROOT" <<'PY'
import json, sys
from pathlib import Path
payload = {
    "kind": "stq_native_staged_v5_smoke_submitted",
    "deployment": sys.argv[4], "run_root": sys.argv[5], "manifest": sys.argv[6],
    "artifact_root": sys.argv[7], "phase_l_smoke": sys.argv[3],
    "preflight": sys.argv[2], "phase": "lora", "gpus": 4,
    "qrels_read": False, "dependency": f"afterok:{sys.argv[2]}",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
