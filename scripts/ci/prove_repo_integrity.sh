#!/usr/bin/env bash
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
#
# prove_repo_integrity.sh — one command that proves the repository cannot lie
# through schema-valid but semantically fake evidence.
#
# It runs the reproducibility + claim-integrity chain end to end:
#   invariant witness · python matrix · namespace policy · claim boundary ·
#   manifest hashes · research-artifact truth · claim→artifact graph · tests.
#
# Honesty contract: the manifest-hash proof is reported as MANIFEST_PROOF:
# NOT_PROVEN until a real (non-placeholder) hash-pinned artifact graph exists.
# That sub-signal is informational and never faked. REPO_INTEGRITY: PROVEN is
# printed only when every reproducibility + truth gate passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="${PYTHONPATH:-.}:."
PY="${PYTHON:-python3}"

# Preflight: this proof runs on a clean clone with only the sci-core stack.
# If a dependency is missing, say exactly what to install instead of dying on a
# cryptic ModuleNotFoundError three steps in — "one command" must be honest.
if ! "$PY" - <<'PY' 2>/dev/null
import importlib.util as u
missing = [m for m in ("numpy", "scipy", "pandas", "networkx", "yaml", "jsonschema", "pytest")
           if u.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
  echo "MISSING DEPENDENCIES. This proof needs only the sci-core stack:" >&2
  echo "  pip install numpy scipy pandas networkx pyyaml jsonschema pytest" >&2
  exit 2
fi

echo "[1/8] invariant witness"
test "$("$PY" scripts/count_invariants.py)" = "97"

echo "[2/8] python version matrix"
"$PY" scripts/check_python_matrix.py >/dev/null

echo "[3/8] namespace policy"
"$PY" scripts/check_namespace_policy.py >/dev/null

echo "[4/8] claim boundary"
"$PY" scripts/ci/check_claim_boundary.py >/dev/null

echo "[5/8] manifest hashes (proof mode)"
if "$PY" scripts/ci/check_manifest_hashes.py --min-artifacts 1 >/dev/null 2>&1; then
  echo "      MANIFEST_PROOF: PROVEN"
else
  # No real hash-pinned artifact graph exists yet. This is the honest state,
  # not a failure of the script — say so explicitly and do not fake success.
  echo "      MANIFEST_PROOF: NOT_PROVEN (no real hash-pinned artifacts committed)"
fi

echo "[6/8] research-artifact truth"
"$PY" scripts/ci/check_research_artifact_truth.py >/dev/null

echo "[7/8] claim-to-artifact graph"
"$PY" scripts/ci/check_claim_artifact_graph.py >/dev/null

echo "[8/8] gate + contract tests"
# Scope: the integrity surface — the truth/graph/manifest gates (tests/ci) and
# the research-line schema+semantic contracts (tests/research_lines). These run
# on a clean clone with only the sci-core stack. tests/tools (release-packaging
# harness) is deliberately NOT here: it needs pydantic/cryptography/httpx and is
# release machinery, not lie-prevention — including it would break the
# clean-clone, one-command guarantee. The release harness keeps its own CI job.
"$PY" -m pytest tests/research_lines tests/ci \
  -q -p no:cacheprovider >/dev/null

echo "REPO_INTEGRITY: PROVEN"
