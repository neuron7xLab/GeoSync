# Release Checklist (100% Gate)

- [ ] `python -m unittest artifacts/inference_extrapolation_validator/test_generate_artifact.py -v`
- [ ] `python artifacts/inference_extrapolation_validator/falsifier.py`
- [ ] `bash artifacts/inference_extrapolation_validator/scripts/brutal_e2e_proof.sh`
- [ ] `python artifacts/inference_extrapolation_validator/generate_artifact.py verify --artifact artifacts/inference_extrapolation_validator/example_artifact.json`
- [ ] `make iev-gate`

If any step fails, release is blocked.
