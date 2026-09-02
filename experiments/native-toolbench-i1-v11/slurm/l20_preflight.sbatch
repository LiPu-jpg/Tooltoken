#!/usr/bin/env bash
#SBATCH --job-name=lb-i1-preflight
#SBATCH --partition=q_intel_share_L20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/user46/logs/latent-register/%x-%j.out
#SBATCH --error=/mnt/home/user46/logs/latent-register/%x-%j.err
set -euo pipefail
: "${PROJECT_ROOT:?}"; : "${MANIFEST:?}"; : "${EVAL_ROOT:?}"; : "${NATIVE_ROOT:?}"; : "${OUTPUT_ROOT:?}"; : "${PYTHON_BIN:?}"
test -f "$PROJECT_ROOT/SOURCE.sha256"
(cd "$PROJECT_ROOT" && sha256sum -c SOURCE.sha256 >/dev/null)
test -f "$MANIFEST"; test -f "$EVAL_ROOT/evaluation_manifest.json"
test -f "$EVAL_ROOT/registry.jsonl"; test -f "$EVAL_ROOT/queries.jsonl"
"$PYTHON_BIN" - "$MANIFEST" "$EVAL_ROOT/evaluation_manifest.json" <<'PY'
import json, sys
manifest=json.load(open(sys.argv[1]))
eval_manifest=json.load(open(sys.argv[2]))
assert manifest["kind"] == "native_latebound_standard_sequence_sft_manifest"
assert manifest["split_kind"] == "api_identity_disjoint_from_official_g1_test"
assert manifest["train_api_count"] == 1000
assert manifest["train_i1_action_intersection_count"] == 0
assert eval_manifest["candidate_documents"] == 10439
assert eval_manifest["registration_optimizer_steps"] == 0
print("i1-preflight=PASS")
PY
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' '{"kind":"native_i1_preflight","passed":true,"qrels_read":false}' > "$OUTPUT_ROOT/preflight.json"
sha256sum "$OUTPUT_ROOT/preflight.json" > "$OUTPUT_ROOT/SHA256SUMS"
printf '%s\n' "$(sha256sum "$OUTPUT_ROOT/SHA256SUMS" | awk '{print $1}')" > "$OUTPUT_ROOT/COMPLETE"
