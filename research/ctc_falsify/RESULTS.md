# CTC-FALSIFY-001 — Consolidated Result (in-silico, scoped, hypothesis-level)

**Status: a finished negative-as-product.** This document closes the
in-silico arc (L1 → L2 → C3) into one defensible finding. It is **not** a
verdict on the Communication-through-Coherence (CTC) theory and **not** a
real-data result. It is a scoped statement about *instruments*.

## Claim actually established

> On a physics-grounded generative ground truth with a *known, toggleable*
> directed A→B gamma-phase channel, **two orthogonal standard CTC estimands
> are blind to that channel by their own pre-registered positive-control
> gate, while a privileged mean-gamma-phase-offset estimator separates the
> channel cleanly.** Therefore the blindness is a property of the standard
> estimands, not of the ground truth.

Epistemic tier: **INFERENCE / hypothesis** (in-silico, generative model,
N=2 estimator families). NOT a theory falsification. NOT real electrophysiology.

## Method (provenance-pinned)

| Layer | What | Merged | SHA |
|---|---|---|---|
| L1 | Two-population Sakaguchi–Kuramoto generative GT + confound nulls N1/N2/N3 + N⁺ + standard PLV/coherence pipeline; fail-closed gates | PR #748 | `cbfd1abe` |
| L2 | Standardized-residual layer, jointly-matched surrogate, 8 self-audit fixes as gates (v1 phase-randomization estimator) | PR #750 | `eb521856` |
| C3 | v2 time-reversed-surrogate directed Phase-Slope Index estimator + boundary probe | PR #752 | `5ba34fd7` |

Supporting infra (separate, honest, not bundled): calib determinism de-flake
(`39916212`, #753), pytest-9-safe sharded fast-gate (`d123b8ea`, #754).

## Results

| Estimator | N⁺ residual | worst confound | verdict |
|---|---|---|---|
| v1 phase-randomization PLV-residual | z ≈ 1.6 (< 3.0) | z ≈ 2.7 (> 1.96) | `INADMISSIBLE_NPLUS_INSITU_BLIND` |
| v2 time-reversed directed PSI | z ≈ 0.15 (< 3.0) | z ≈ 2.5 (> 1.96) | `INADMISSIBLE_NPLUS_INSITU_BLIND` |
| **boundary probe** (privileged mean γ-phase offset) | N⁺ ≈ −1.38 rad vs N1 ≈ +0.26 rad; xcorr lag ≈ −251 samples | — | **channel recoverable** |

Both standard estimands fail their own pre-registered gate **without any
threshold tuning** (tuning would be a #199-class rescue and is forbidden by
the acceptor falsifiers). The instruments killed their own hypotheses.

## What this is NOT

- NOT "CTC is false." The theory is untouched; only the standard *evidential
  estimands* are shown blind on this generative model.
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
