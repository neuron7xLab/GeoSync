#!/usr/bin/env bash
set -euo pipefail

ART=artifacts/inference_extrapolation_validator/example_artifact.json
GEN=artifacts/inference_extrapolation_validator/generate_artifact.py

python -m unittest artifacts/inference_extrapolation_validator/test_generate_artifact.py -v >/dev/null
python artifacts/inference_extrapolation_validator/falsifier.py >/dev/null
python "$GEN" verify --artifact "$ART" >/dev/null

TMP=$(mktemp)
cp "$ART" "$TMP"
python - <<'PY' "$TMP"
import json,sys
p=sys.argv[1]
d=json.load(open(p))
d['sha256']='0'*64
json.dump(d,open(p,'w'))
PY
set +e
python "$GEN" verify --artifact "$TMP" >/tmp/iev_bad_out 2>&1
CODE=$?
set -e
if [[ $CODE -ne 2 ]]; then
  echo "Expected contract violation code 2, got $CODE"
  exit 1
fi

echo "BRUTAL_E2E_PROOF PASS"
