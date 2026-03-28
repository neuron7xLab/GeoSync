# Required Checks for `main`

Single source of truth for branch protection/ruleset required checks.

## Pull Request Required Checks

1. `repo-policy`
2. `python-quality`
3. `python-fast-tests`
4. `frontend-gate`
5. `dependency-review`
6. `secrets-supply-chain`

These checks are emitted by `.github/workflows/pr-gate.yml` for both `pull_request` (to `main`) and `merge_group` compatibility.
