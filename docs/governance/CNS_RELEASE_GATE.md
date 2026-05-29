# CNS Release Gate

## Scope

This component validates repository release-gate artifacts for the CNS program. The gate checks configuration state coverage, manifest integrity, report presence, and a weighted quality verdict.

## Local verification

```bash
python tools/cns_program/deploy_release_gate.py && \
python tools/cns_program/build_release_manifest.py && \
python tools/cns_program/verify_release_manifest.py && \
python tools/cns_program/verify_reports_contract.py && \
python tools/cns_program/quality_gate.py && \
python -m pytest -q tests/cns_program
```

## Runtime artifacts

- `results/cns_deploy_gate.json`
- `results/cns_release_manifest.json`
- `results/cns_manifest_verification.json`
- `results/cns_reports_contract.json`
- `results/cns_quality_gate.json`

## Merge rule

Merge is valid when the CNS Release Gate workflow passes on Python 3.11 and Python 3.12.
