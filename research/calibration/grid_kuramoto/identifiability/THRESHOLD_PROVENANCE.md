# THRESHOLD_PROVENANCE — swing identifiability front-gate

> **Pre-committed before any calibration validation was run.** This file
> fixes the identifiability score formula and the REFUSE threshold from
> numerical / statistical theory alone. It contains **no** measured
> calibration value. A no-peek drift test
> (`test_threshold_provenance_no_peek`) asserts the constants in
> `core.kuramoto.identifiability` equal the theory values committed here;
> changing either side without changing both fails CI.
>
> Lineage: upgrade #2 on top of CALIB-GRID-001 (PR #749) and its R1
> swing-aware refinement (PR #751). This is **reliability / instrument
> honesty infrastructure**, not a science claim. It does **not** close
> the frozen `noisy.frobenius` gate; it makes the instrument *declare*
> that it fails there.

## 1. What is being graded

The merged R1 swing path solves the linear identity, per node `i`,

    m_i θ̈_i(t) + d_i θ̇_i(t)  =  P_i + Σ_{j≠i} (−K_ij) sin(θ_i(t) − θ_j(t)) ,

stacked over all nodes into one global least-squares problem
`D_std · c_std ≈ y` on the **column-standardised** design `D_std`
(scale factors `s_p`), with `K̂` recovered as `K = −c / s`,
`c = c_std / s`.

The merged persistent-excitation guard is **binary**: it raises
`PersistentExcitationError` only when the *reciprocal condition number*
of `D_std` falls below a hard floor. It is blind to the dominant
calibration failure mode (see § 4): additive phase noise, after double
differentiation, inflates the residual variance by ~5 orders of
magnitude and biases `K̂` by ~20× while *improving* the design
conditioning. Conditioning alone therefore cannot self-report the
noisy-regime failure. The graded layer below propagates **both** the
design conditioning **and** the measurement-noise floor.

## 2. Identifiability score (theory-derived, not a tuned knob)

Linearised ordinary-least-squares covariance of the standardised
solution (Gauss–Markov; the swing identity is linear in
`(K, P)` once `(θ̇, θ̈)` are fixed):

    σ̂²       = ‖y − D_std ĉ_std‖²  /  (n_obs − n_param)        (1)
    Cov(ĉ_std) = σ̂² · (D_stdᵀ D_std)⁺                          (2)
    SE(K̂_p)   = sqrt( [Cov(ĉ_std)]_{pp} )  /  s_p              (3)

Equation (1) is the unbiased residual-variance (noise-floor) estimate;
(2) is the Cramér–Rao / Fisher-information covariance for a linear
Gaussian model (`(D_stdᵀD_std)⁺` is the inverse Fisher information up to
σ̂²); (3) back-transforms to physical units. The Moore–Penrose pseudo-
inverse is used so a rank-deficient design degrades gracefully into the
existing hard PE guard rather than raising inside the covariance.

Per **edge** `p` we form the **Wald / inverse-coefficient-of-variation
ratio**

    w_p = |K̂_p| / SE(K̂_p)                                      (4)

and the scalar **identifiability score** is the worst edge ratio mapped
into a bounded, monotone, documented range:

    IDENTIFIABILITY = w_min / (1 + w_min),  w_min = min_p w_p     (5)

`IDENTIFIABILITY ∈ [0, 1)`. It → 0 as the weakest edge becomes
unidentifiable (`SE ≫ |K̂|`, e.g. noise-dominated or near phase-locked)
and → 1 as every edge is sharply estimated (`SE ≪ |K̂|`). It is
**scale-free** (ratio of like units), **monotone decreasing in the
measurement-noise σ** (σ̂² ∝ σ² enters every `SE` linearly, numerator
fixed) and **monotone increasing in record length** (`SE ∝ 1/√n_obs`
under persistent excitation). It is derived from the model's Fisher
information and residual noise floor — there is no fitted constant.

## 3. REFUSE threshold (numerical + statistical theory)

A coupling entry is **not statistically identifiable** when its
two-sided 95 % Wald confidence interval contains zero, i.e.

    |K̂_p|  ≤  z_{0.975} · SE(K̂_p) ,    z_{0.975} = 1.959963984540054

(standard-normal 0.975 quantile; large-sample Wald CI for the linear
Gaussian model). Equivalently `w_p ≤ z_{0.975}`. Mapping the binding
edge through (5), the REFUSE region is

    IDENTIFIABILITY  <  REFUSE_SCORE
    REFUSE_SCORE  =  z / (1 + z)  with  z = 1.959963984540054
                  =  0.6621429206633470…                          (6)

i.e. the front-gate REFUSES exactly when the weakest coupling entry's
95 % CI straddles zero (the estimate is statistically indistinguishable
from "no edge"). This is the canonical "estimate is not separable from
its own uncertainty" boundary; it is the smallest theory point at which
a point estimate is *misleading rather than imprecise*.

Conjoined numerical floor (kept as the **hard** fail, unchanged from the
merged code): the reciprocal condition number of `D_std` below

    PE_HARD_FLOOR  =  sqrt(eps_float64)  =  1.4901161193847656e-08    (7)

means the linear solve has lost more than half of the float64 mantissa
(`eps ≈ 2.22e-16`; losing ½ the 52-bit mantissa ⇒ relative error
≳ √eps). That extreme remains `PersistentExcitationError` (fail-closed).
The graded REFUSE in (6) sits **above** that hard floor: it is the
statistical envelope, not the numerical catastrophe.

Decision lattice (fail-closed, evaluated in this order):

1. reciprocal cond `< PE_HARD_FLOOR`  → `PersistentExcitationError`
   (unchanged hard fail).
2. else `IDENTIFIABILITY < REFUSE_SCORE` → typed `REFUSE` verdict +
   score + reason + the offending edge + its CI. **No point estimate is
   promoted as trustworthy.**
3. else → `ACCEPT` with `K̂`, per-edge SE and 95 % CIs.

## 4. Why conditioning alone is insufficient (motivation, not a gate)

The reciprocal condition number of the standardised design is *larger*
(better) in the σ=0.02 calibration case than in the noiseless one,
because additive phase noise adds variance to the sine regressors. The
merged binary PE guard therefore passes the noisy case while the
recovered `K̂` is biased ~20×. The residual variance σ̂² (eq. 1) is the
quantity that separates the regimes by ~5 orders of magnitude; that is
why the score is built on the *noise-aware* regression covariance (2),
not on conditioning. (The specific calibration numbers belong in
`RESULTS.md`, produced **after** this provenance was committed.)

## 5. Frozen-gate discipline

Nothing here changes `PREREGISTRATION.md`, `gates.py`, the five frozen
acceptance thresholds, the embedded IEEE data, the seeds, σ, the θ₀
perturbation or the decision rule of CALIB-GRID-001 / R1. The first-
order and R1 swing point estimates are bit-identical when the new
covariance fields are ignored (additive `SwingCouplingEstimate` fields;
verified by the existing R1 bit-stability test plus a new ignore-fields
regression test). The frozen `noisy.frobenius` gate **stays NEGATIVE**;
the only change is that the instrument now self-reports that it fails
there instead of silently emitting a misleading `K̂`.
