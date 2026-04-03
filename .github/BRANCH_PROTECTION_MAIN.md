# Branch Protection / Ruleset Manifest for `main`

Apply this exactly in GitHub repository settings.

## Branch targeting
- Target branch: `main`

## Pull request requirements
- Require a pull request before merging: **enabled**
- Required approving reviews: **1**
- Require review from Code Owners: **enabled**
- Dismiss stale pull request approvals when new commits are pushed: **enabled**
- Require approval of the most recent reviewable push: **enabled**
- Require conversation resolution before merging: **enabled**

## Required status checks (strict, fail-closed)
- Require status checks to pass before merging: **enabled**
- Require branches to be up to date before merging: **enabled**
- Required checks:
  - `repo-policy`
  - `python-quality`
  - `python-fast-tests`
  - `frontend-gate`
  - `dependency-review`
  - `secrets-supply-chain`

## Branch safety
- Allow force pushes: **disabled**
- Allow deletions: **disabled**
- Bypass list: **empty** (except explicit admin break-glass policy)

## Merge queue compatibility
- Merge queue: optional, supported because required checks are emitted on `merge_group`.
