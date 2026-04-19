# `archive/attic/` — Retired Top-Level Subtrees

This directory consolidates top-level folders that accumulated at the
repository root during earlier iterations and are no longer part of
the live development surface.

## What lives here

| Subtree | Former role | Why archived |
|---------|-------------|--------------|
| `legacy/` | Legacy code / scripts from before the current architecture | No production path imports from it; kept for historical reference |
| `handoff/` | Team-handoff notes, questions, trackers | Replaced by ADRs + issue tracker |
| `patches/` | One-off patches applied before the Sprint-0 baseline | Already applied to main; files kept as provenance |
| `scope/` | Scope-definition documents for earlier planning rounds | Superseded by the 7-cycle remediation docs |

## Discipline

* **Nothing in `archive/attic/` is imported by production code.** If a
  grep for `from archive` or `import archive` returns a row, that is a
  bug.
* **Removal policy.** After a full release cycle during which no
  grep / search / ticket requires the attic, a release may drop the
  attic wholesale. Until then, it is cold storage, not live code.
* **Additions.** Put something into the attic only when (a) it lived
  at the repo root, (b) it no longer has a live consumer, and (c) the
  git history alone is not a friendly enough reference for the
  content.
