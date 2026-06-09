# Structural Audit

Generated UTC: 2026-06-09T07:14:33Z

## A. Packaging contract
- Expected: Clean editable install succeeds without network build dependencies.
- Observed: MFN clean verifier installs editable package in a fresh Python 3.12 venv with --no-deps.
- Command: `python tools/verify_mfn_clean_install.py --python python3.12 --report artifacts/release_evidence/product_state.json`
- Signal: PASS
- Root cause hypothesis: Previous setuptools/wheel build isolation dependency was removed via local backend.
- Confidence: HIGH
- Severity: LOW
- Classification: MACHINE_VERIFIED

## B. Import contract
- Expected: Installed environment can import geosync, geosync.mfn, and core.
- Observed: Captured by clean verifier import_geosync_mfn check.
- Command: `fresh venv python -c import core, geosync, geosync.mfn`
- Signal: PASS
- Root cause hypothesis: Editable install exposes repository packages through generated .pth.
- Confidence: HIGH
- Severity: LOW
- Classification: MACHINE_VERIFIED

## C. Entrypoint contract
- Expected: Installed PATH resolves mfn, mfn-api, and mfn-validate.
- Observed: Clean verifier records command_v checks for all MFN entrypoints.
- Command: `command -v mfn && command -v mfn-api && command -v mfn-validate inside clean venv PATH`
- Signal: PASS
- Root cause hypothesis: pyproject [project.scripts] entrypoints are emitted into editable wheel metadata.
- Confidence: HIGH
- Severity: LOW
- Classification: MACHINE_VERIFIED

## D. CLI command contract
- Expected: MFN help commands exit 0 deterministically.
- Observed: Clean verifier executes --help for every MFN entrypoint.
- Command: `mfn --help; mfn-api --help; mfn-validate --help`
- Signal: PASS
- Root cause hypothesis: argparse CLI surface is dependency-light and installed.
- Confidence: HIGH
- Severity: LOW
- Classification: MACHINE_VERIFIED

## E. Operation pipeline contract
- Expected: Installed mfn command creates an instrumented bundle and mfn-validate accepts it.
- Observed: Clean verifier runs installed mfn and installed mfn-validate.
- Command: `mfn --out <bundle> run; mfn-validate --bundle <bundle>`
- Signal: PASS
- Root cause hypothesis: MFN pipeline is standard-library-only and writes required artifacts/manifests.
- Confidence: HIGH
- Severity: LOW
- Classification: MACHINE_VERIFIED

## F. Test contract
- Expected: Focused MFN unit tests pass; full repository suite is not claimed.
- Observed: Focused MFN tests pass; full suite remains outside this evidence bundle.
- Command: `python -m pytest tests/unit/mfn -q`
- Signal: PASS
- Root cause hypothesis: Contract tests cover MFN CLI and build backend; full suite may require heavyweight deps.
- Confidence: HIGH
- Severity: MEDIUM
- Classification: MACHINE_VERIFIED

## G. Tooling contract
- Expected: Referenced focused tooling is executable.
- Observed: Ruff, compileall, namespace and python-matrix checks pass for focused scope.
- Command: `ruff/compileall/check_single_entrypoint/check_python_matrix`
- Signal: PASS
- Root cause hypothesis: Focused tooling exists in active environment; full dev extra install is not claimed.
- Confidence: HIGH
- Severity: MEDIUM
- Classification: MACHINE_VERIFIED

## H. Lockfile contract
- Expected: Canonical lockfile is consistent if claimed.
- Observed: uv.lock is absent; pip/requirements lock remains existing but full lock install is not verified here.
- Command: `baseline uv.lock detection; no full dependency install claimed`
- Signal: BLOCKED
- Root cause hypothesis: Repository dependency surface is larger than focused MFN smoke path.
- Confidence: HIGH
- Severity: MEDIUM
- Classification: MACHINE_ASSISTED

## I. Release artifact contract
- Expected: Evidence bundle has manifest and SHA256SUMS.
- Observed: This tool writes release_manifest.json and SHA256SUMS after command capture.
- Command: `python tools/generate_release_evidence.py`
- Signal: PASS
- Root cause hypothesis: Evidence generation is local and deterministic for captured files.
- Confidence: HIGH
- Severity: LOW
- Classification: MACHINE_VERIFIED

## J. Documentation/runbook contract
- Expected: README command matches executable verifier command.
- Observed: README documents MFN run, validate, and clean verification report command.
- Command: `static README inspection plus clean verifier execution`
- Signal: PASS
- Root cause hypothesis: Documented commands map to installed MFN verifier path.
- Confidence: MEDIUM
- Severity: LOW
- Classification: MACHINE_ASSISTED
