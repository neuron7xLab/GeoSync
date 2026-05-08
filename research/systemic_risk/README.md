# Systemic Risk as Phase Transition — Kuramoto on Interbank Networks

> **Tier (per `CLAIMS.md`):** `HYPOTHESIS` until the falsification battery
> in `falsification.py` returns `HARD_PASS` on ≥ 2 independent crises.

## Hypothesis (pre-registered)

The interbank market is a network of phase oscillators coupled through
exposure-weighted lending. Approach to the Kuramoto bifurcation
:math:`K_c` manifests as elevated rolling order parameter `R(t)`,
positive slope of `R(t)`, and growing variance of `R(t)` (canonical
critical-slowing-down diagnostics, Scheffer et al. 2009, *Nature*
**461**: 53). The composite early-warning score

```
score(t) = R̄(t) · |slope(R)(t)| · √var(R)(t)
```

is *elevated* in pre-event windows preceding banking-crisis dates
relative to null windows drawn from the same series at safe distance
from any event.

## Modules

| File | Role |
|------|------|
| `event_ledger.py` | `BankingCrisisLedger` from Laeven & Valencia 2018 + 2023 anchor events |
| `topology.py` | Empirical exposure → adjacency adapter + Barabási-Albert null |
| `phase_extraction.py` | Interbank-rate spread → :math:`\theta_i(t)` via Hilbert (wraps `core.kuramoto.phase_extractor`) |
| `early_warning.py` | Rolling-window CSD predictor (level + slope + variance composite) |
| `falsification.py` | Pre-registered AUC + permutation p + Benjamini-Hochberg battery |

## Falsification protocol (pre-registered, frozen)

For each crisis *c* in the ledger with at least one valid pre-event window:

1. Extract pre-event score values from the `pre_event_window_days`-long
   window ending one day before `c.start`.
2. Sample `null_window_count` non-overlapping null windows of the same
   length, each ≥ `min_distance_from_event_days` from any event.
3. Compute :math:`AUC_c` (Mann-Whitney U) and one-sided permutation p
   with `n_permutations` resamples.
4. Apply Benjamini-Hochberg FDR correction across all crises that
   yielded valid windows.

### Decision rule

| Verdict | Condition |
|---------|-----------|
| `HARD_FAIL` | ∃ c with :math:`AUC_c \le` `fail_auc` (default 0.55) |
| `HARD_PASS` | ≥ 2 crises with :math:`AUC_c \ge` `pass_auc` AND :math:`p^{BH}_c \le` `pass_alpha` (defaults 0.70, 0.01) |
| `UNDECIDED` | neither — collect more data, do not promote tier |

The thresholds are written *once* and never edited after seeing data.
Any change to thresholds is a new pre-registration on a fresh branch.

## Dataset manifest

| Source | Status | Notes |
|--------|--------|-------|
| Laeven-Valencia 2018 (IMF WP/18/206) | inline | systemic banking crisis dates |
| post-LV2020 designations (USA-2023, CHE-2023) | inline | flagged `source="post_LV2020"` |
| e-MID Italian interbank (2009-2015) | external — caller-supplied | feed `from_exposure_matrix` |
| BIS Locational Banking Statistics | external — caller-supplied | quarterly aggregate, sensitivity-only |

The package ships *no* real interbank exposure data; the topology
loader expects user-supplied parquet/CSV. The Barabási-Albert null
exists exclusively as a falsification baseline (Boss et al. 2004).

## Physics anchors (from `CLAUDE.md`)

* INV-K1 — :math:`0 \le R(t) \le 1` (universal, P0); enforced in
  `EarlyWarningResult.__post_init__`.
* INV-K5 — :math:`\langle R \rangle \sim O(1/\sqrt{N})` for incoherent
  phases (statistical, P1); test
  `test_early_warning::test_inv_k5_incoherent_finite_size`.
* INV-EVT1, INV-EVT2 — event-ledger date and country invariants
  (this module).
* INV-TOP1..3 — topology adjacency / weights / labels invariants
  (this module).
* INV-EW1, INV-EW2 — early-warning config invariants (this module).

## Maintenance-hierarchy role

Sustainer (Layer 2). The module emits a diagnostic score; it never
takes execution action. A `HARD_PASS` outcome would only motivate
promotion to a Protector role downstream — it does not itself protect
any gradient.

## Status

Initial commit:

* All public APIs typed (`mypy --strict` clean).
* 57 tests covering invariants + statistical sanity (BH FWER,
  null-AUC ≈ 0.5).
* Hypothesis remains `HYPOTHESIS` tier; first real-data run pending
  on user-supplied e-MID dump.
