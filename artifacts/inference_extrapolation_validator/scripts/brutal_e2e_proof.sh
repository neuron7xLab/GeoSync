#!/usr/bin/env bash
# End-to-end smoke test for the IEV CLI.
#
# Generates a fresh artifact in a tmpdir (dynamic witness/generated
# timestamps so the freshness gate never time-bombs the test), verifies
# it, corrupts the SHA, and asserts that re-verification fails closed
# with exit code 2.
set -euo pipefail

GEN=artifacts/inference_extrapolation_validator/generate_artifact.py
NOW=$(python3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).replace(microsecond=0).isoformat())")

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
ART="$TMPDIR/artifact.json"

python "$GEN" generate \
  --context "verified dataset v1" \
  --hypothesis "signal X remains stable beyond observed range" \
  --model-id ext_inf --model-version 3.0.0 --seed 1 \
  --prompt-hash 9db6f9e7c6e5d6f6ac7a72f01561f1f5f8d6759f4f8999f8fceca85f2f4e6eb4 \
  --requirements-lock-sha256 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --risk high --result survived --test-passed true \
  --tests-run "adversarial_test,boundary_check,null_model_comparison,counterexample_search,replay_test" \
  --falsifiers-run "counterexample_search,null_model_comparison" \
  --failure-modes-checked "boundary_violation,overfit" \
  --null-models-run "baseline_random_or_stationary,boundary_shuffle,counterexample_search_null,replay_consistency_null" \
  --null-model-results '{"baseline_random_or_stationary":0.2,"boundary_shuffle":0.1,"counterexample_search_null":0.15,"replay_consistency_null":0.05}' \
  --hypothesis-score 0.9 \
  --reality-probe-id probe-001 --reality-substrate exchange_sim \
  --observation-hash dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd \
  --drift-score 0.2 \
  --witness-status approved --witness-reviewer-id rev-1 \
  --witness-review-timestamp-utc "$NOW" \
  --witness-review-hash bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --generated-at-utc "$NOW" \
  --purpose-id risk_gate_v1 \
  --purpose-statement "Prevent unverified extrapolation promotion" \
  --purpose-alignment-score 0.9 --purpose-drift-check 0.1 \
  --out "$ART" >/dev/null

python "$GEN" verify --artifact "$ART" >/dev/null

python - <<PY "$ART"
import json, sys
p = sys.argv[1]
d = json.load(open(p))
d["sha256"] = "0" * 64
json.dump(d, open(p, "w"))
PY

set +e
python "$GEN" verify --artifact "$ART" >/dev/null 2>&1
CODE=$?
set -e
if [[ $CODE -ne 2 ]]; then
  echo "Expected contract violation code 2, got $CODE"
  exit 1
fi

echo "BRUTAL_E2E_PROOF PASS"
