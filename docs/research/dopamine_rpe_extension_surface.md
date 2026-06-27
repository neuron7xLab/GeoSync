# Dopamine RPE Extension Surface

Status: experimental research layer, not a canonical public claim.

## Boundary

The canonical dopamine invariant remains:

```text
delta = reward + gamma * next_value - value
```

This PR does not replace `DopamineController.compute_rpe` and does not rewrite
INV-DA1 / INV-DA3 / INV-DA7. It adds a separate optional surface for testing
whether richer reward-processing layers improve decision quality without
corrupting the anchored TD(0) axis.

## Why this shape

The repository already treats dopamine as a P0-anchored algebraic control path.
Changing that path directly would weaken the falsifier surface. The safer mature
architecture is a two-tier design:

1. Canonical TD(0) remains the proof anchor.
2. Experimental extensions are computed around the anchor and must earn promotion
   through tests, backtests, and claim-registry evidence.

## Added extension layers

| Layer | Function | Role |
| --- | --- | --- |
| Canonical anchor | `canonical_td_error` | Recomputes raw TD(0) without hidden nonlinear processing. |
| Distributional RPE | `distributional_td_error` | Pushes reward quantiles through the same affine TD(0) map. |
| Asymmetric learning | `asymmetric_value_update` | Allows different update magnitudes for positive and negative deltas. |
| Risk-adjusted reward | `risk_adjusted_reward` | Penalizes volatility/drawdown before secondary RPE evaluation. |
| Average-reward vigor | `average_reward_vigor` | Converts reward-above-baseline into bounded execution vigor. |
| Unified surface | `compute_rpe_extension_surface` | Returns all extension outputs without mutating controller state. |

## The unified estimator: `ExpectileEnsembleValue`

The functions above are stateless probes. `expectile_value.py` promotes them
into one coherent **online distributional value learner** — the part that makes
the surface a real system rather than a set of disconnected helpers.

The key result from the verified literature (Dabney et al. 2020, *Nature*
577:671) is that asymmetric learning rates are not a free heuristic: a channel
with positive/negative rates `(alpha_pos, alpha_neg)` converges on the
**expectile** at level

```text
tau = alpha_pos / (alpha_pos + alpha_neg)
```

So an ensemble of channels with spread `tau in (0, 1)` *is* the distributional
code. This single mechanism unifies four of the layers:

| Need | How the ensemble provides it |
| --- | --- |
| Canonical anchor (mean) | the `tau = 0.5` channel — the 0.5-expectile *is* the mean, so it reproduces scalar TD(0) exactly (`canonical_value`). |
| Distributional RPE (L1) | the full `(tau, value)` expectile curve (`expectile_curve`). |
| Asymmetric learning (L2) | each channel's `(alpha_pos, alpha_neg)` ⇒ its `tau`. |
| Risk-adjusted value (L4) | a low-`tau` expectile is risk-averse; `risk_adjusted_value(tau)` reads it off directly — risk emerges from the ensemble, not an ad-hoc penalty. |
| Variance / dispersion | `dispersion()` = top minus bottom expectile, the variance-rescale signal of adaptive coding (Tobler 2005). |

Average-reward vigor (L3) then consumes `canonical_value()` as its baseline.

**Non-invasiveness is structural, not promised.** A symmetric single channel
reduces *exactly* to the canonical TD(0) target — the P0 anchor is a special
case of the estimator, proven in
`tests/.../test_expectile_value.py::test_symmetric_channel_recovers_canonical_mean`.
The estimator is fully deterministic (no RNG; identical inputs ⇒ bit-identical
state) and every contract violation fails closed with `ValueError`.

### Invariants and the falsifier

| ID | Type | Statement |
| --- | --- | --- |
| INV-EXP1 | algebraic | `tau == alpha_pos / (alpha_pos + alpha_neg)` to 1e-12. |
| INV-EXP2 | algebraic | the `0.5`-expectile equals the arithmetic mean. |
| INV-EXP3 | asymptotic | a symmetric channel converges to the sample mean (recovers canonical TD(0)). |
| INV-EXP4 | monotonic | `tau_a < tau_b ⇒ V_a ≤ V_b` at convergence. |
| INV-EXP5 | universal | every `V_i` stays inside the convex hull of the targets. |

The **falsifier** is two-part: (A) the analytic expectile is an exact fixed
point of the asymmetric-weighted residual (`Σ w_i (x_i − e) = 0` to 1e-9); (B)
the online learner converges into the `O(√eta)` stationary neighbourhood of that
fixed point. Part B's tolerance is the genuine constant-rate fluctuation band,
not a slackened bound — it still separates an expectile from a quantile (for
`tau = 0.9` on `N(0,1)` the quantile ≈ 1.28 vs expectile ≈ 0.94).

## Neuroscience provenance

Each extension layer mirrors a published result in dopamine / value
neuroscience. Tiers below reflect the consensus of the cited *primary*
sources, adversarially verified (3-vote refute-or-confirm). The neuroscience
licenses the architectural *form* of each layer — **not** any numeric
parameter. Neuro-to-market transfer of effect sizes, learning rates, or
reversal points is explicitly **not** licensed: the data come from primate /
rodent midbrain and cortex, not from a P&L process.

### Anchor — scalar TD(0) error
- **[established]** Phasic dopamine encodes a quantitative reward-prediction
  error `delta = r + gamma*V' - V`. Schultz, Dayan & Montague 1997
  (*Science* 275:1593); Bayer & Glimcher 2005 (*Neuron*, DA firing ∝ RPE
  magnitude). This is the empirical justification for the P0 anchor (INV-DA1,
  INV-DA7).
- **[contested / refinement]** Within-trial dopamine may track the temporal
  derivative of an evolving value signal `V(s)` more tightly than the scalar
  `delta` (Hamid et al. 2016, *Nat. Neurosci.* nn.4173). *Not independently
  confirmed in our verification pass — treat as open.*
- *Falsifier:* if measured DA responses do not scale monotonically with
  `r + gamma*V' - V` under controlled cue/reward manipulation, the anchor is
  wrong.

### Layer 1 — Distributional RPE (`distributional_td_error`)
- **[established]** Dopamine neurons collectively encode a *distribution* over
  future reward, not a scalar mean. Dabney, Kurth-Nelson, Uchida et al. 2020,
  *Nature* 577:671 (mouse VTA: 40 cells/6 mice magnitude task, 31 cells/5
  probability task). Diverse, reliable reversal points: split-half
  **R = 0.58, p = 1.8e-5**.
- **[established]** Replicated outside midbrain: macaque ACC reversal-point
  diversity **R = 0.44, P = 0.003** (n = 41), *Nat. Neurosci.*
  s41593-023-01535-w.
- *Boundary:* "distributional" at the **single-neuron expectile** level
  (Dabney) is distinct from **population / projection-defined** heterogeneity
  (Avvisati et al. 2024, *Cell Reports* S221112472400408X). Do not conflate.
- *Falsifier:* if reversal points are not reproducible across data splits
  (R -> 0), or the decoded density fails to recover ground-truth modes, the
  distributional code collapses to a scalar.

### Layer 2 — Asymmetric learning / expectile (`asymmetric_value_update`)
- **[established]** Asymmetric scaling of positive vs negative RPEs is real and
  diverse cell-to-cell. The encoded statistic is an **expectile** with quantile
  level

  ```text
  tau_i = alpha_i+ / (alpha_i+ + alpha_i-)
  ```

  (Dabney 2020 supplement, linear response `f(delta)=delta`). This makes the
  module's `positive_alpha` / `negative_alpha` a principled expectile selector,
  not a free heuristic — an ensemble with spread `tau in (0,1)` *is* the
  Layer-1 distributional code.
- **[established]** Diversity is data-justified: a per-cell asymmetry model
  lowered cross-validated MSE by **3.4% (p = 1.4e-11)**; that 3.4% in
  firing-rate space ≈ **50%** in distribution space. Asymmetry <-> reversal
  point correlation **p = 8.1e-5**, robust to a convex-Hill control (r = 0.41,
  p = 0.006). Separate `alpha+ != alpha-` learning rates beat single-rate fits
  on ACC data (n = 94, R = 0.62, P = 0.0001).
- *Falsifier:* if `tau = alpha+/(alpha+ + alpha-)` does not predict which
  quantile a unit encodes, the expectile interpretation is dead.

### Layer 3 — Average-reward / tonic dopamine -> vigor (`average_reward_vigor`)
- **[established as theory]** Tonic dopamine encodes the long-run average reward
  rate, which acts as the opportunity cost of time and sets response vigor /
  latency. Niv, Daw, Joel & Dayan 2007, *Psychopharmacology* 191:507.
- **[contested]** Direct causal vigor evidence (slow mesolimbic DA ∝ reward
  rate; Hamid 2016) was **not confirmed** in our verification pass — the theory
  is strong, the causal empirics remain open here.
- *Falsifier:* if raising tonic DA (pharmacological / optogenetic) at fixed
  phasic signal does not increase response vigor, the link fails.

### Layer 4 — Risk-adjusted reward (`risk_adjusted_reward`)
- **[established]** Phasic DA RPE encodes **marginal utility** (the first
  derivative of a nonlinear, S-shaped utility), not raw magnitude: convex /
  risk-seeking at low juice volumes, concave / risk-averse at high (2 rhesus
  monkeys, certainty-equivalent utilities, p < 0.001 to 1e-7). Stauffer, Lak &
  Schultz 2014, *Current Biology* 24(20):2491 (PMC4228052). 3-vote verified.
- **[established]** DA carries an integrated scalar **subjective value**; risk
  enters the signal *only to the extent it changes choice* (~63% of neurons
  match each animal's risk weighting). Lak, Stauffer & Schultz 2014, *PNAS*
  111:2343 (PMC3926061). 3-vote verified.
- **[established phenomenon]** A distinct **sustained ramp** covaries with
  reward **variance**, maximal at P=0.5 (Bernoulli P(1−P)), dissociating a
  variance term from the monotonic value signal. Fiorillo, Tobler & Schultz
  2003, *Science* 299:1898. **[contested mechanism]** the ramp may be a
  TD-averaging artifact or emergent state-uncertainty signal (Niv, Duff &
  Dayan 2005) — existence robust, interpretation open.
- **[established]** DA uses **adaptive / efficient coding**: the RPE zero-point
  shifts to expected reward and the gain rescales to the reward variance
  (Tobler, Fiorillo & Schultz 2005, *Science* 307:1642) — the mean-baseline +
  variance-rescale structure this estimator reproduces.
- *Boundary (numbers do NOT transfer):* prospect-theory probability weighting is
  real but species-specific — monkeys show **concave** weighting (δ ≈ 0.57),
  **not** the human inverse-S. Use the functional form, never the fitted
  parameters.
- *Falsifier:* if DA RPE responses scale with raw reward magnitude rather than a
  concave / risk-weighted utility, the risk-coding rationale is unsupported.

> **Engineering boundary.** The licensed transfer is *structural only*:
> expectile ensemble (Layers 1+2), average-reward baseline (Layer 3), and
> risk-penalized reward (Layer 4). Specific neural `alpha+/-`, reversal points,
> and R^2 values are **not** valid market-model parameters and must be
> fit/locked on in-domain data through the promotion gates below.

## Promotion gates

Do not promote this layer into `DopamineController.step` until all gates pass:

1. Unit tests prove the canonical TD(0) identity is untouched.
2. Distributional output preserves affine reward differences.
3. Risk penalties are monotone and never increase reward under non-negative risk.
4. Asymmetric learning changes only update size, not raw delta.
5. Backtest evidence shows stable improvement against the current dopamine TD
   baseline across walk-forward folds.
6. A new claim entry is added only after evidence paths and falsifier tests exist.

## Suggested next PR

Add an evaluation harness that compares:

```text
canonical_td0
canonical_td0 + distributional_rpe
canonical_td0 + asymmetric_alpha
canonical_td0 + risk_adjusted_reward
canonical_td0 + average_reward_vigor
combined_extension_surface
```

Required output:

```text
results/dopamine_rpe_extension/EVAL_SUMMARY.json
results/dopamine_rpe_extension/WALKFORWARD_REPORT.md
results/dopamine_rpe_extension/PARAMETER_LOCK.json
```

Promotion rule: if the combined surface beats canonical TD(0) only in-sample, it
stays research. If it survives walk-forward, null/spike, and parameter-lock tests,
it can become an EXTRAPOLATED claim. Only after a dedicated falsifier block should
it become ANCHORED.
