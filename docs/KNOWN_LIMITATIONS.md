# Known Limitations

This document is the honest counterweight to the README badges. Every claim
in the codebase that would benefit from a caveat lives here so that external
reviewers, auditors, and future contributors can calibrate their
expectations without hunting through source files.

## Scope and discipline

* Every limitation listed here is **declared before it is felt** — no one
  should discover these by surprise from production.
* When a limitation is closed, the corresponding bullet is removed **and**
  the fix is linked from the [changelog](../CHANGELOG.md) or the relevant
  ADR under [`docs/adr/`](adr/).
* `BASELINE.md` at the repo root is the quantitative companion — if a
  limitation is measurable, its number belongs there.

## L-1 · Execution surface is paper-trading only

The execution layer (`execution/`) ships smart routing, Kelly / MV sizing,
compliance checks, and a full order-lifecycle ledger, but the venues it
talks to in-repo are **paper connectors**. No credential in the public
source tree connects to a live exchange. Production deployment requires:

1. A private venue-credentials bundle that is *not* committed.
2. A staged rollout against a paper account first, reproducing the same
   order trace with `pytest tests/integration/` green.
3. An operator signing off on `runtime/kill_switch` defaults for that
   specific venue.

The README describes execution as a production-ready fabric. That is true
for the *platform*; it is **not** a live trading track record.

## L-2 · Strategy catalogue is intentionally thin

`strategies/` contains two first-party strategies (`neuro_geosync`,
`quantum_neural`) and a registry. The platform is capable of hosting
many more; the catalogue is deliberately thin because most strategies
the author researches live outside this repository.

Do **not** extrapolate platform maturity from strategy count. Treat the
two shipped strategies as reference implementations, not a catalogue.

## L-3 · Single performance record lives in `L2_ROBUSTNESS.json`

The one number referenced by the README — "Deflated Sharpe DSR = 15.1,
Pr(real) = 1.0" — lives in the `L2_ROBUSTNESS.json` artefact produced by
the L2 demo gate. It is **not** an annualised trading Sharpe on real
capital. Other Sharpe-like numbers scattered across docs (`docs/HPC_AI_V4.md`,
`docs/automated_risk_testing.md`, `docs/operations/PRODUCT_PAIN_SOLUTION.md`)
are illustrative values from their respective subsystems and are not
consolidated into a single performance ledger yet.

When a canonical live-performance ledger exists, it will live at
[`docs/PERFORMANCE_LEDGER.md`](PERFORMANCE_LEDGER.md) and this limitation
will close. Until then, **assume no live-capital track record**.

## L-4 · Dual Python tree (`core/` + `src/geosync/`)

The repository carries two parallel Python packages:

* `core/` — the legacy tree, predates the canonical layout.
* `src/geosync/` — the canonical tree, where new code must land.

For a long-lived migration, every module in `core/` should have a
thin shim that re-exports from `src/geosync/core/...` (see
`core/compat.py` for the reference pattern). Anywhere this shim is
absent, the duplication is a real risk of divergence under concurrent
edits. [ADR-0017](adr/0017-core-to-src-migration.md) captures the
migration plan and the acceptance criteria for retiring `core/`.

Until the migration closes, new code:

* imports from `geosync.core...` (canonical path);
* if legacy `core.X` must be touched, a shim is mandatory before any
  change is merged.

## L-5 · Bus factor = 1

The repository is authored by a single person with bot assistance
(dependabot, review bots). Independent review and outside contributors
are welcome and required for production use beyond the author's own
deployment. Until that external review layer is visible in the git
history, the bus factor is 1 and must be treated as such in any
risk / compliance analysis.

## L-6 · Organisational history before Sprint-0

Top-level layout before 2026-04 carried legacy subtrees (`legacy/`,
`handoff/`, `patches/`, `scope/`, `archive/`) that accumulated over
earlier iterations. They are consolidated under `archive/attic/` as
part of the canonical-honesty release and preserved for historical
reference. No production path imports from the attic; removing it
entirely in a future release is an acceptable next step once a grace
window has passed.

---

If you find a limitation of the platform that is not listed here, that
is itself a limitation. File it, even as a paragraph, before relying
on the affected surface in production.
