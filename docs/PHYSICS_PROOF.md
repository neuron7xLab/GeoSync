# Physics Proof — CoherenceBridge Signal Contract

## 1. PROVEN (mathematical derivations in code)

**γ = 2H + 1** (fBm path scaling)
- H from DFA log-log slope (Peng et al. 1994)
- Verified numerically: white noise H≈0.5→γ≈2.0, persistent H>0.5→γ>2.0
- INV-PK1: γ derived, never assigned

**R ∈ [0, 1]** (Kuramoto order parameter)
- R = |1/N Σ exp(iθ_k)| is a magnitude of complex mean
- INV-K1: bounded by construction

**Δg = σ²/2 ≥ 0** (ergodicity gap non-negativity)
- Peters (2019): time growth rate = μ - σ²/2
- Gap is always non-negative (variance ≥ 0)

**adjusted_size ≤ intended_size** (risk gate absolute bound)
- CoherenceRiskGate: fail-closed, never amplifies

**risk_scalar = max(0, 1 - |γ - 1|)** (algebraic identity)
- Symmetric around metastable point γ = 1.0
- Lipschitz-1, piecewise linear

## 2. EMPIRICALLY CALIBRATED (requires real data)

**Ricci lead time: PENDING**
- κ < 0 indicates concurrent topology fragility
- Predictive lead time NOT established on synthetic data
- RicciLeadTimeCalibrator includes shuffle test to prevent false positives
- Calibrate on real Askar EURUSD tick data before claiming edge

**DFA H on live FX: PENDING**
- DFAGammaEstimator validated on synthetic (white noise, persistent, anti-persistent)
- Real FX tick validation required
- regime_shift flag: |H_global - H_local| > 0.2

**NEI thresholds: PROVISIONAL**
- NEI < 0.5 → ERGODIC, NEI 0.5-1 → MILD, NEI > 1 → SIGNIFICANT, NEI > 2 → SEVERE
- Thresholds from Peters (2019) theory, not calibrated on FX
- Adjust after live backtest

## 3. KNOWN LIMITATIONS

- γ estimator valid only under local stationarity (DFA mitigates but does not eliminate)
- Active Inference ergodicity assumption corrected via Peters (2019) — correction is first-order
- Ricci lead time not yet out-of-sample validated — honest about this
- Lyapunov (Rosenstein) O(N²) — not real-time for tick-by-tick
- Bootstrap CI on DFA invalid for strongly non-stationary paths (block bootstrap mitigates)

## 4. REFERENCES

- Peng, C.-K. et al. (1994). Mosaic organization of DNA nucleotides. *Physical Review E*, 49(2), 1685.
- Peters, O. (2019). The ergodicity problem in economics. *Nature Physics*, 15(12), 1216-1221.
- Rosenstein, M.T. et al. (1993). A practical method for calculating largest Lyapunov exponents. *Physica D*, 65(1-2), 117-134.
- Kantelhardt, J.W. et al. (2002). Multifractal detrended fluctuation analysis. *Physica A*, 316(1-4), 87-114.
- Forman, R. (2003). Bochner's method for cell complexes and combinatorial Ricci curvature. *Discrete & Computational Geometry*, 29(3), 323-374.
