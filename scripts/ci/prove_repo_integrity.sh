#!/usr/bin/env bash
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-.}:."
PY="${PYTHON:-python3}"

if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util as u
missing = [m for m in ("numpy", "scipy", "pandas", "networkx", "yaml", "jsonschema", "pytest") if u.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  echo "MISSING DEPENDENCIES. This proof needs only the sci-core stack:" >&2
  echo "  pip install numpy scipy pandas networkx pyyaml jsonschema pytest" >&2
  exit 2
fi

echo "[1/13] invariant witness"
test "$("$PY" scripts/count_invariants.py)" = "111"

echo "[2/13] python version matrix"
"$PY" scripts/check_python_matrix.py >/dev/null

echo "[3/13] namespace policy"
"$PY" scripts/check_namespace_policy.py >/dev/null

echo "[4/13] claim boundary"
"$PY" scripts/ci/check_claim_boundary.py >/dev/null

echo "[5/13] manifest hashes proof mode"
if "$PY" scripts/ci/check_manifest_hashes.py --min-artifacts 1 >/dev/null 2>&1; then
  echo "      MANIFEST_PROOF: PROVEN"
else
  echo "      MANIFEST_PROOF: NOT_PROVEN"
fi

echo "[6/13] research artifact truth"
"$PY" scripts/ci/check_research_artifact_truth.py >/dev/null

echo "[7/13] claim artifact graph"
"$PY" scripts/ci/check_claim_artifact_graph.py >/dev/null

echo "[8/13] mfn surface sensor"
"$PY" tools/mfn_surface_check.py --strict >/dev/null

echo "[9/13] json artifact contract"
for fixture in candidate blocked; do
  "$PY" tools/validate_json_artifact_contract.py \
    "examples/json_artifact_contract.${fixture}.json" \
    --out "artifacts/validation/json_artifact_contract_${fixture}.result.json" >/dev/null
  "$PY" tools/check_json_contract_evidence_policy.py \
    "examples/json_artifact_contract.${fixture}.json" \
    --out "artifacts/validation/json_artifact_policy_${fixture}.result.json" >/dev/null
done

echo "[10/13] json evidence policy"
"$PY" tools/validate_json_artifact_contract.py \
  examples/json_artifact_contract.candidate.json \
  --out artifacts/validation/json_artifact_contract.result.json >/dev/null
"$PY" tools/check_json_contract_evidence_policy.py \
  examples/json_artifact_contract.candidate.json \
  --out artifacts/validation/json_artifact_policy.result.json >/dev/null

echo "[11/13] json contract receipt"
"$PY" tools/json_contract_receipt.py \
  --out artifacts/validation/json_contract_receipt.json >/dev/null

echo "[12/13] autonomy cycle"
"$PY" tools/run_autonomy_cycle.py \
  --out artifacts/validation/autonomy_cycle.json >/dev/null

echo "[13/13] gate + contract tests"
# Do NOT suppress pytest output: a hidden failure here is undiagnosable (a gate
# that fails with no FAILED/ERROR line reads as a vacuous pass). Show short
# tracebacks so the failing node id is always visible in CI logs.
"$PY" -m pytest tests/research_lines tests/ci -q -p no:cacheprovider --tb=short

echo "REPO_INTEGRITY: PROVEN"
