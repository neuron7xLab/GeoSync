# PR 1400 Verification Matrix

Status: draft gate matrix for `second-order-energy-stream-clean`.

## Change under review

`core/kuramoto/second_order.py` changes the audit energy evaluator from trajectory-wide dense pairwise allocation to one-snapshot pairwise evaluation.

## Invariant ledger

| Invariant | Required outcome | Evidence source |
|---|---:|---|
| Swing-energy formula unchanged | must hold | code review + physics tests |
| Peak pairwise memory reduced | `O(N^2)` | code review + metadata |
| No stability overclaim | must hold | docs + metadata |
| Partial audit remains partial | must hold | `remaining_gaps`, `promotion_allowed` |
| Mutation floor not lowered | must hold | mutation ratchet |
| Python quality intact | must hold | PR Gate / Code Hygiene |
| Import architecture intact | must hold | Import Architecture Gate |
| Physics behavior intact | must hold | Physics Invariants / Kernel / Reliability |

## Gate policy

Draft remains draft until all required gates are green:

- Mutation Kill Gate
- Commit Acceptor Gate
- PR Gate / python-quality
- Physics Invariants
- Physics Kernel Gate
- Physics Reliability Gate
- Code Hygiene Gate
- Import Architecture Gate
- Readiness Gate

## Final verdict rule

- Any required red gate => `BLOCK`.
- Any missing required gate => `REVISE`.
- All required gates green and claim scope unchanged => `ACCEPT`.
- Growing entropy without added falsification => `CLOSE`.
