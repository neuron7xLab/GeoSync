# CTC-FALSIFY-001 — Consolidated Result (in-silico, scoped, hypothesis-level)

**Status: a finished negative-as-product, including a recursive
self-retraction.** This document closes the in-silico arc
(L1 → L2 → C3 → **C4 self-audit**). It is **not** a verdict on the
Communication-through-Coherence (CTC) theory and **not** a real-data result.

## Claim actually established (post-C4)

> On a physics-grounded generative ground truth with a *known, toggleable*
> directed A→B gamma-phase channel, **two orthogonal standard CTC estimands
> (PLV-residual, time-reversed PSI) are blind to that channel by their own
> pre-registered gates.** The C3 boundary probe *suggested* the channel was
> "recoverable in principle" by a privileged phase-offset estimator —
> **C4 audited that escape-hatch and it FAILED its own gates** (Cohen's
> d ≈ 0.48 < 1.0; confound false-positive ≈ 0.47; 20/32 false channels at
> `channel_strength = 0`). The C3 "recoverable in principle / blindness is
> an estimand property" inference is therefore **RETRACTED to scope.**

What survives: *on this generative model, **no** estimator tested — standard
or privileged — robustly and admissibly separates the directed channel from
confounds under adversarial audit.* Whether the blindness is an estimand
property or an identifiability limit of these observables is **OPEN**, not
decided. The earlier C3 "−1.38 vs +0.26" separation was a 5-seed artifact
(the exact small-N fragility class caught in DOPA A1→A2).

Epistemic tier: **INFERENCE / hypothesis** (in-silico). NOT a theory
falsification. NOT real electrophysiology. The instrument falsified its
**own** convenient conclusion — that is the result, not a flaw.

## Method (provenance-pinned)

| Layer | What | Merged | SHA |
|---|---|---|---|
| L1 | Two-population Sakaguchi–Kuramoto generative GT + confound nulls N1/N2/N3 + N⁺ + standard PLV/coherence pipeline; fail-closed gates | PR #748 | `cbfd1abe` |
| L2 | Standardized-residual layer, jointly-matched surrogate, 8 self-audit fixes as gates (v1 phase-randomization estimator) | PR #750 | `eb521856` |
| C3 | v2 time-reversed-surrogate directed PSI + boundary probe | PR #752 | `5ba34fd7` |
| C0 | consolidated RESULTS (pre-C4) | PR #756 | `f8c70ba1` |
| **C4** | **adversarial self-audit of the C3 boundary probe** | this PR | — |

Supporting infra (separate, honest, not bundled): calib determinism de-flake
(`39916212`, #753), pytest-9-safe sharded fast-gate (`d123b8ea`, #754).

## Results

| Estimator | finding | verdict |
|---|---|---|
| v1 phase-randomization PLV-residual | N⁺ z ≈ 1.6 (< 3.0); confound z ≈ 2.7 | `INADMISSIBLE_NPLUS_INSITU_BLIND` |
| v2 time-reversed directed PSI | N⁺ z ≈ 0.15 (< 3.0); confound z ≈ 2.5 | `INADMISSIBLE_NPLUS_INSITU_BLIND` |
| privileged mean γ-phase offset (C3 escape-hatch) | d ≈ 0.48 (< 1.0); confound FP ≈ 0.47; sweep 20/32 false channels at ch=0; sign-flip OK | `C4_INADMISSIBLE_ESTIMATOR_CANT_SEPARATE` |

Every estimand failed its own pre-registered gate **without threshold
tuning** (forbidden by the acceptor falsifiers). The standard ones killed
their hypotheses; the privileged one killed **C3's conclusion about them**.

## C4 self-retraction (the recursive result)

C3 used a privileged estimator to argue the channel was recoverable, hence
the blindness was the *standard estimands'* fault. C4 subjected that
privileged estimator to the **same** fail-closed discipline (positive
control, confound rejection, sign-flip, parameter sweep). It failed G1/G2/G4
(passed only G3, directionality). Therefore the system's *own escape from
its negative was itself a self-lie*, caught and retracted — not buried.
This is the apex behaviour of the contract: it falsifies its own
conclusions, not only external hypotheses.

## What this is NOT

- NOT "CTC is false." The theory is untouched.
- NOT "the channel is recoverable / unrecoverable" — that is now **OPEN**;
  C3's recoverability inference is withdrawn.
- NOT a real-data finding. No electrophysiology was bound.
- NOT a closed research line. `ctc_falsify_001` stays **OPEN**.

## Honesty invariant

If a principled estimator recovers N⁺ and, on real data, the CTC residual
survives the jointly-matched null — that **survival is reported, not spun**.
Symmetric thresholds, identical template. The machine is audience-blind.

## Next (only if it sharpens the truth — not reflexive continuation)

C4/C5 is justified **only** to test whether the privileged
phase-offset estimator (the one the boundary probe identified) remains
admissible and whether the blindness reproduces on **real** paired
LFP+spike data under a pre-committed dataset rule. Absent that, this
document is the terminal artifact: a scoped, provenance-pinned,
hypothesis-level negative about standard CTC estimands. One proven
negative, delivered — not an open cycle.
