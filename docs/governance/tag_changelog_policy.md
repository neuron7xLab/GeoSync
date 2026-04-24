# Tag and Changelog Policy

## Tag format
- Release tags must use semantic versioning: `vMAJOR.MINOR.PATCH`.
- Pre-releases: `vMAJOR.MINOR.PATCH-rc.N`.

## Changelog contract
- Every PR that changes runtime behavior must include a `newsfragments/*` entry.
- Release changelog is generated from fragments (towncrier) and must include:
  - behavior changes,
  - migration notes,
  - rollback notes when applicable.

## CI enforcement
- Reject release PRs without changelog fragments.
- Reject non-semver release tags.

## Minimal release gate
1. `pytest` green on required suites.
2. `mypy/ruff/black` green.
3. `claim_status_applied: applied`.
4. Changelog generated and reviewed.
