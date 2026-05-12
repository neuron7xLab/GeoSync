# STN Conflict Inhibition Model

This is not a biological STN simulation.

This is a computational control abstraction inspired by the STN hyperdirect pathway: fast inhibition under high conflict before unsafe action selection.

- ACC role: `compute_acc_conflict_score()` builds conflict vector from internal contradictions.
- STN role: `stn_hyperdirect_gate()` blocks promotion when conflict exceeds threshold.
- IEV role: evidence promotion only after conflict remains low and all other gates pass.
