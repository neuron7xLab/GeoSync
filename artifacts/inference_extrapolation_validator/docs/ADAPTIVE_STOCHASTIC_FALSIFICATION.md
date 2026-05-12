# Adaptive Stochastic Falsification

IEV keeps fail-closed semantics but allows bounded stochastic tolerance in STN thresholding.

- `stochastic_falsification.tolerance` in [0,1]
- `noise_seed` ensures deterministic replay
- `adaptive_threshold = base_threshold + U(0, tolerance)` (capped at 1.0)

Purpose: reduce cognitive rigidity for borderline high-value hypotheses while preserving mechanical conflict inhibition.


Enterprise rule: tolerance must remain 0.0 unless explicitly approved and replay-deterministic evidence attached.
Status: EXPERIMENTAL, excluded from enterprise readiness claims.
