#!/usr/bin/env bash
set -euo pipefail

python -m unittest artifacts/inference_extrapolation_validator/test_generate_artifact.py -v
python artifacts/inference_extrapolation_validator/falsifier.py
bash artifacts/inference_extrapolation_validator/scripts/brutal_e2e_proof.sh
make iev-gate

GIT_HEAD=$(git rev-parse HEAD)
TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "{\"event\":\"final_handoff_gate\",\"status\":\"pass\",\"module\":\"inference_extrapolation_validator\",\"git_head\":\"${GIT_HEAD}\",\"timestamp_utc\":\"${TS}\"}"
