# GitHub Actions Workflows

Canonical workflow set for the GeoSync repository. Every workflow is pinned
to a single, unambiguous responsibility so the branch-protection contract
for `main` stays auditable.

## Workflows

| Workflow | Trigger | Purpose | Blocks merge? |
|---|---|---|---|
| [`pr-gate.yml`](./pr-gate.yml) | `pull_request` → `main`, `merge_group` | Synchronous merge gate — lint, types, fast tests, frontend, deps, secrets | **Yes** (required checks) |
| [`main-validation.yml`](./main-validation.yml) | `push` → `main` | Post-merge deep validation (slow/heavy suites, integration) | No |
| [`codeql.yml`](./codeql.yml) | `push` → `main`, weekly cron, manual | CodeQL SAST — Python, JavaScript, Go | No (reports to Security tab) |
| [`security-deep.yml`](./security-deep.yml) | Weekly cron, manual | Out-of-band security scans — `pip-audit`, `gitleaks`, `trivy-fs` | No |

Any additional workflow **must** define a distinct decision boundary that is
not already covered by the four above. Overlapping gates are rejected.

## Required status checks for `main`

The branch-protection ruleset enforces exactly the job names emitted by
`pr-gate.yml` — see [`../BRANCH_PROTECTION_MAIN.md`](../BRANCH_PROTECTION_MAIN.md).

1. `repo-policy`
2. `python-quality`
3. `python-fast-tests`
4. `frontend-gate`
5. `dependency-review`
6. `secrets-supply-chain`

All six jobs are fail-closed (`continue-on-error: false`). Merge queue is
supported because every required job declares the `merge_group` trigger.

## Runtime hardening

All canonical workflows opt into GitHub's Node.js 24 runtime for JavaScript
actions via:

```yaml
env:
  FORCE_JAVASCRIPT_ACTIONS_TO_NODE24: 'true'
```

Frontend workflows additionally pin `actions/setup-node` to `node-version:
'24'` to avoid mixed-runtime drift. Python environments are provisioned
through the composite action [`../actions/setup-geosync`](../actions/setup-geosync),
which enforces pinned `pip` and `pytest` versions from `requirements-dev.lock`
for reproducible CI.

## Supply-chain hygiene

Every third-party action reference **must** be pinned to a 40-character commit
SHA (`uses: owner/repo@<sha> # vX.Y`). The `repo-policy` job in `pr-gate.yml`
fails the build if any unpinned reference is introduced. Re-pinning happens
via the vetted [Renovate rules](../renovate.json) or explicit maintainer commits.

## Adding a new workflow

1. Confirm it does not duplicate an existing gate.
2. Declare least-privilege `permissions:` at the top level.
3. Never use `pull_request_target:` — `repo-policy` will reject it.
4. Pin every `uses:` reference by SHA.
5. Update this README and `../BRANCH_PROTECTION_MAIN.md` if the change affects
   required status checks.
