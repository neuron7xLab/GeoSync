# Physics Release Scorecard (Task 7 / P1-5)

## Claim

A release may be labelled *physics-ready* only when every `required` dimension
is `CI_VERIFIED` or `EVIDENCE_BEARING`; no dimension may claim a verified state
without an existing binding artifact.

## Instrument

- `physics_release_scorecard.yml` — one discrete state per release-critical
  dimension, drawn from the ladder
  `FALSE < UNTESTED < PARTIAL < LOCAL_VERIFIED < CI_VERIFIED < EVIDENCE_BEARING`.
- `tools/release/check_physics_release_scorecard.py` — validates schema,
  enforces **anti-theater** (a `CI_VERIFIED`/`EVIDENCE_BEARING` state whose
  `binding` file is missing is a hard error), and the **readiness gate**
  (`--assert-physics-ready` exits 1 while any required dimension is below
  `CI_VERIFIED`).
- `tests/tools/test_physics_release_scorecard.py` — schema, anti-theater,
  fail-closed readiness, ladder enforcement.

## Honest snapshot (the refusal is the instrument working)

| Dimension | State | Required |
|---|---|---|
| dependency_security_gate | CI_VERIFIED | ✔ |
| codeql_gate | CI_VERIFIED | ✔ |
| claim_boundary_gate | CI_VERIFIED | ✔ |
| kuramoto_k_ownership | CI_VERIFIED | ✔ |
| descriptor_promotion_firewall | CI_VERIFIED | ✔ |
| negative_evidence_ledger | CI_VERIFIED | ✔ |
| **ricci_l2_evidence_protocol** | **PARTIAL** | ✔ |
| **second_order_stability_audit** | **PARTIAL** | ✔ |
| **same_sha_ci_proof** | **PARTIAL** | ✔ |
| scorecard_gate | UNTESTED | — |
| artifact_replayability | PARTIAL | — |

`--assert-physics-ready` therefore exits **1** today: GeoSync is **not**
physics-ready by its own scorecard, blocked by the Ricci L2 protocol
(HYPOTHESIS-tier), the second-order audit (diagnostics-only, issue #1109), and
same-SHA release proof. Recording those as PARTIAL — rather than rounding up — is
the entire point.

## Verification chain

| Link | Evidence |
|---|---|
| command | `check_physics_release_scorecard.py` → exit 0 (valid); `--assert-physics-ready` → exit 1 (honest) |
| test | `pytest tests/tools/test_physics_release_scorecard.py` → 6 passed |
| static | ruff + black + `mypy --strict` clean; descriptor-firewall EXIT 0 |
| failure mode | a verified state with a missing binding, or a PARTIAL required dimension asserted ready, fails the gate |

## Blocked claims

The scorecard proves **declared-state consistency and binding existence**, not
the empirical truth of any physics dimension. A `CI_VERIFIED` state means the
gate exists and passes in CI — not that the underlying science is established.
Promotion to `EVIDENCE_BEARING` requires real data + replay + baseline +
falsifier + semantic validation, which the scorecard records but does not itself
produce.
