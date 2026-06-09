# FINAL STATUS

status: YELLOW
readiness_score_0_100: 79
machine_verified_passes: 7
remaining_blockers:
- full_dependency_lockfile_install_not_verified_by_this_mfn_smoke
exact_commands_that_passed:
- clean_install_fact: `/root/.pyenv/versions/3.14.4/bin/python tools/verify_mfn_clean_install.py --python python3.12 --report /workspace/GeoSync/artifacts/release_evidence/product_state.json`
- mfn_tests: `/root/.pyenv/versions/3.14.4/bin/python -m pytest tests/unit/mfn -q`
- ruff: `/root/.pyenv/versions/3.14.4/bin/python -m ruff check tools/build_backend.py tools/verify_mfn_clean_install.py geosync/mfn conftest.py`
- compileall: `/root/.pyenv/versions/3.14.4/bin/python -m compileall -q geosync/mfn conftest.py tests/unit/mfn tools/build_backend.py tools/verify_mfn_clean_install.py`
- entrypoint_guard: `/root/.pyenv/versions/3.14.4/bin/python scripts/check_single_entrypoint.py`
- python_matrix: `/root/.pyenv/versions/3.14.4/bin/python scripts/check_python_matrix.py`
- operation_direct: `/root/.pyenv/versions/3.14.4/bin/python -m geosync.mfn.cli --out /tmp/mfn-release-evidence run`
exact_commands_that_failed:
- none
first_artifact_path_to_open: artifacts/release_evidence/FINAL_STATUS.md
unsupported_claims: none; full production readiness is not claimed.
