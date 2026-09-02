#!/usr/bin/env bash
# Immutable input/source preflight; qrels are never read.
#SBATCH --job-name=stq-native-staged-preflight
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/user46/logs/latent-register/%x-%j.out
#SBATCH --error=/mnt/home/user46/logs/latent-register/%x-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${MANIFEST:?}"; : "${ARTIFACT_ROOT:?}"; : "${PYTHON_BIN:?}"; : "${OUTPUT_ROOT:?}"
test -f "$PROJECT_ROOT/scripts/train.py"; test -f "$MANIFEST"; test ! -e "$OUTPUT_ROOT"
"$PYTHON_BIN" - "$MANIFEST" "$ARTIFACT_ROOT" <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
assert m["kind"] == "native_latebound_standard_sequence_sft_manifest"
assert (m["candidate_count"], m["example_count"]) == (999, 10483)
assert m["seen_unseen_exact_identity_overlap"] == 0
root = Path(sys.argv[2])
assert sum(1 for _ in (root / "unseen_model" / "candidates.jsonl").open()) == 837
assert sum(1 for _ in (root / "unseen_model" / "queries.jsonl").open()) == 1066
print("stq_manifest_gate=PASS", flush=True)
PY
(cd "$PROJECT_ROOT" && sha256sum -c SOURCE.sha256)
mkdir -p "$OUTPUT_ROOT"
cp "$PROJECT_ROOT/SOURCE.sha256" "$OUTPUT_ROOT/SOURCE.sha256"
printf '%s\n' '{"kind":"stq_native_staged_preflight","passed":true,"qrels_read":false}' > "$OUTPUT_ROOT/event.json"
sha256sum "$OUTPUT_ROOT/event.json" > "$OUTPUT_ROOT/COMPLETE"
