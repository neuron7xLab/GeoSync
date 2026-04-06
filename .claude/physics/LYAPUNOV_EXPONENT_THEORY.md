# Lyapunov Exponent Theory — Chaos/Order Detection for GeoSync

## Core concept

The Maximal Lyapunov Exponent (MLE) measures the average rate of exponential
divergence of nearby trajectories in phase space:

    λ_max = lim_{t→∞} (1/t) · ln(|δx(t)| / |δx(0)|)

For a scalar time series (e.g. Kuramoto R(t), portfolio returns):
- **λ > 0** → chaotic / unpredictable (nearby trajectories diverge)
- **λ ≈ 0** → marginal / edge-of-chaos (critical transition zone)
- **λ < 0** → stable / predictable (nearby trajectories converge)

## Algorithm: Rosenstein (1993)

1. Delay-embed scalar series: x_i → (x_i, x_{i+τ}, ..., x_{i+(m-1)τ})
2. For each embedded point, find nearest neighbor (excluding temporal vicinity)
3. Track log-divergence of neighbor pairs over time
4. λ_max = slope of mean(ln(divergence)) vs time (initial linear region)

Key parameters:
- **dim** (embedding dimension): 2 for 1D maps, 3-5 for continuous systems
- **tau** (delay): first minimum of mutual information, or 1 for discrete maps
- **max_divergence_steps**: shorter = captures initial exponential growth better

## Calibration results

| System | Theoretical λ | GeoSync MLE | Accuracy |
|--------|--------------|-------------|----------|
| Logistic map r=4 | ln(2) = 0.693 | 0.689 | 99.5% |
| White noise | 0 | 0.001 | — |
| Damped oscillator | < 0 | -0.006 | correct sign |
| Kuramoto R(t) subcritical | > 0 (noisy) | 0.222 | correct sign |
| Kuramoto R(t) supercritical | < subcritical | < subcritical | correct ordering |

## Connection to GeoSync modules

- **R(t) → MLE**: tells you if synchronization dynamics are predictable
- **MLE → Kelly**: λ < 0 → higher conviction → larger fraction
- **MLE → Cryptobiosis**: extreme positive MLE → gradient turbulent → consider DORMANT
- **MLE → GVS**: feeds chaos_health component (|λ| < 0.5 = healthy)

## Invariants

- INV-LE1: MLE finite for any finite bounded input
- INV-LE2: MLE sign matches dynamical regime (noise ≈ 0, stable < 0, chaos > 0)

## References

Rosenstein, Collins & De Luca (1993). Physica D, 65(1-2), 117-134.
Wolf, Swift, Swinney & Vastano (1985). Physica D, 16(3), 285-317.
