#!/usr/bin/env bash
set -euo pipefail

SBATCH=${SBATCH:-/mnt/data/hpc/slurm/23.02/bin/sbatch}
ROOT=${PROJECT_ROOT:-/mnt/home/user46/deployments/latebound-stq-native-staged-v6-20260902}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-/mnt/home/user46/inputs/simpletoolquestions-unseen-manifest-v1}
MODEL_ROOT=${MODEL_ROOT:-/mnt/home/user46/windy/models/Qwen3-8B}
NATIVE_ROOT=${NATIVE_ROOT:-/mnt/home/user46/latent-register}
PYTHON_BIN=${PYTHON_BIN:-/mnt/home/user46/.venvs/toolgen-l20/bin/python}
RUN_ROOT=${RUN_ROOT:-/mnt/home/user46/runs/stq-native-staged-v6-recovery-20260902}
MANIFEST=${MANIFEST:-$ROOT/manifest.json}
SOURCE_STAGE_L=${SOURCE_STAGE_L:-/mnt/home/user46/runs/stq-native-staged-v5-formal-20260902/stage-l-formal/checkpoint}

test -x "$SBATCH"
test -f "$ROOT/slurm/preflight.sbatch"
test -f "$ROOT/slurm/stage_l_reload.sbatch"
test -f "$ROOT/slurm/stage_f_formal.sbatch"
test -f "$ROOT/slurm/stage_f_reload.sbatch"
test -f "$MANIFEST"
test -d "$ARTIFACT_ROOT"; test -d "$MODEL_ROOT"; test -d "$NATIVE_ROOT"
test -f "$SOURCE_STAGE_L/COMPLETE"
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT"

preflight=$($SBATCH --parsable --export="ALL,PROJECT_ROOT=$ROOT,MANIFEST=$MANIFEST,ARTIFACT_ROOT=$ARTIFACT_ROOT,PYTHON_BIN=$PYTHON_BIN,OUTPUT_ROOT=$RUN_ROOT/preflight" "$ROOT/slurm/preflight.sbatch")
stage_l="reused:$SOURCE_STAGE_L"
stage_l_reload=$($SBATCH --parsable --dependency="afterok:$preflight" --export="ALL,PROJECT_ROOT=$ROOT,MODEL_ROOT=$MODEL_ROOT,NATIVE_ROOT=$NATIVE_ROOT,PYTHON_BIN=$PYTHON_BIN,CHECKPOINT_ROOT=$SOURCE_STAGE_L,OUTPUT=$RUN_ROOT/stage-l-reload/reload_audit.json" "$ROOT/slurm/stage_l_reload.sbatch")
stage_f=$($SBATCH --parsable --dependency="afterok:$stage_l_reload" --export="ALL,PROJECT_ROOT=$ROOT,MODEL_ROOT=$MODEL_ROOT,MANIFEST=$MANIFEST,NATIVE_ROOT=$NATIVE_ROOT,PYTHON_BIN=$PYTHON_BIN,OUTPUT_ROOT=$RUN_ROOT/stage-f-formal,WARMUP_ROOT=$SOURCE_STAGE_L,TRAIN_SEED=${TRAIN_SEED:-17}" "$ROOT/slurm/stage_f_formal.sbatch")
stage_f_reload=$($SBATCH --parsable --dependency="afterok:$stage_f" --export="ALL,PROJECT_ROOT=$ROOT,MODEL_ROOT=$MODEL_ROOT,NATIVE_ROOT=$NATIVE_ROOT,PYTHON_BIN=$PYTHON_BIN,CHECKPOINT_ROOT=$RUN_ROOT/stage-f-formal/checkpoint,OUTPUT=$RUN_ROOT/stage-f-reload/reload_audit.json" "$ROOT/slurm/stage_f_reload.sbatch")

"$PYTHON_BIN" - "$RUN_ROOT/submitted.json" "$ROOT" "$RUN_ROOT" "$MANIFEST" "$ARTIFACT_ROOT" "$preflight" "$stage_l" "$stage_l_reload" "$stage_f" "$stage_f_reload" "$SOURCE_STAGE_L" <<'PY'
import json, sys
from pathlib import Path

out, root, run_root, manifest, artifact, preflight, stage_l, stage_l_reload, stage_f, stage_f_reload, source_stage_l = sys.argv[1:]
payload = {
    "kind": "stq_native_staged_v6_recovery_submitted",
    "deployment": root,
    "run_root": run_root,
    "manifest": manifest,
    "artifact_root": artifact,
    "preflight": preflight,
    "stage_l_formal": stage_l,
    "stage_l_source_checkpoint": source_stage_l,
    "stage_l_reload": stage_l_reload,
    "stage_f_formal": stage_f,
    "stage_f_reload": stage_f_reload,
    "gpus_per_training_stage": 4,
    "qrels_read": False,
    "dependencies": [
        f"afterok:{preflight}",
        f"afterok:{stage_l_reload}",
        f"afterok:{stage_f}",
    ],
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
PY
