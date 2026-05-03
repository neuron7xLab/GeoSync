# Stability bounds for damped phase synchronization

## FACT — implemented dynamics

Continuous form:

```
dθ_i/dt = K · sin(θ_i* − θ_i)
```

Discrete (Euler) step used by `geosync.neuroeconomics.reset_wave_engine.run_reset_wave`:

```
θ_i^{t+1} = θ_i^t + Δt · K · sin(θ_i* − θ_i^t)
```

Phase-alignment potential:

```
V(θ) = (1/N) · Σ_i (1 − cos(θ_i* − θ_i))
```

`V(θ) ≥ 0` always; `V(θ) = 0` iff `θ_i = θ_i*` for all i (modulo the manifold seam at ±π).

## MODEL

`run_reset_wave` is a numerical relaxation solver on the compact phase manifold `[-π, π)` with fail-closed lock at threshold `max_phase_error`. The RK4-fixed integrator path is the canonical default; Euler is exposed for cross-checking.

## ANALOGY

Terms like "reset-wave" or "homeostasis" are interpretation only. Per IERD §4.2 the `coupling_gain` parameter is named in IERD-compliant lexicon (no `serotonin_gain` / `thermodynamic` literals).

## Empirical bound (tested under Monte Carlo n=400)

For this implementation and the Monte Carlo + grid stress tests in `tests/test_reset_wave_stress_validation.py`, monotone potential decrease is reliable in the practical regime:

```
0 < coupling_gain · dt ≤ 0.2
```

Outside this region, oscillation or divergence may occur. The negative tests cover that regime explicitly so the boundary itself is part of the contract surface, not a hidden assumption.

## Cross-references

* Source: [`geosync/neuroeconomics/reset_wave_engine.py`](../geosync/neuroeconomics/reset_wave_engine.py)
* Critical centers: [`docs/reset_wave_critical_centers.md`](reset_wave_critical_centers.md)
* Validation report: [`docs/reset_wave_validation_report.md`](reset_wave_validation_report.md)
* Calibration artefacts: [`reports/reset_wave_validation_summary.json`](../reports/reset_wave_validation_summary.json)
