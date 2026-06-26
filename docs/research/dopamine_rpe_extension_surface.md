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
