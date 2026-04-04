# Workflows (Canonical Set)

Only these workflows are active:

1. `pr-gate.yml`
   - Required merge gate for PRs into `main`.
2. `main-validation.yml`
   - Deeper validation on `main` after merge.
3. `security-deep.yml`
   - Scheduled/manual deep security scans.

If a proposed workflow does not define a distinct decision boundary, do not add it.

## Runtime hardening note

All canonical workflows opt into GitHub's Node.js 24 runtime for JavaScript-based actions via:

`FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: 'true'`

Frontend workflows also pin `actions/setup-node` to `node-version: '24'` to avoid mixed runtime drift.

Python environment setup now follows a contract-based protocol:
- bootstrap tooling is pinned in `.github/config/python-bootstrap.lock`,
- runtime verification is centralized in `.github/scripts/verify_toolchain_contract.py`,
- `pytest` version is reconciled against `requirements-dev.lock`.
