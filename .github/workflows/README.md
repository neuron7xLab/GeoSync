# Workflows (Canonical Set)

Only these workflows are active:

1. `ci.yml`
   - Required validation gate for pull requests and pushes to the repository default branch.
   - Also supports manual execution via `workflow_dispatch`.
2. `security-deep.yml`
   - Scheduled/manual deep security scans.

If a proposed workflow does not define a distinct decision boundary, do not add it.
