# Workflows (Canonical Set)

Only these workflows are active:

1. `pr-gate.yml`
   - Required merge gate for PRs into `main`.
2. `main-validation.yml`
   - Deeper validation on `main` after merge.
3. `security-deep.yml`
   - Scheduled/manual deep security scans.

If a proposed workflow does not define a distinct decision boundary, do not add it.
