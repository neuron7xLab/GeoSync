# STN Hyperdirect Gate (Executable)

This module implements a mechanical conflict brake analogous to STN hyperdirect inhibition:
- ACC analogue: `acc_conflict_score`
- threshold: `0.5`
- action: `exit_2_fail_closed`

If conflict exceeds threshold, promotion is physically blocked by contract violation.
