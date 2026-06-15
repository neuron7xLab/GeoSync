# GeoSync Measured-Maturity — Next Patch

The only thing standing between `HYPOTHESIS` and `Measured-Single` for the L2
Ricci line is **real data**. Everything else (schemas, guard, nulls, cost model,
replay command, market limits) is already in place and fail-closed.

## Single critical-path step

1. Capture **one** real, hash-pinned depth-5 crypto-perp order-book session,
   ≥ 3600 s contiguous, gap-checked:
   - `python scripts/collect_binance_perp_l2.py` (or `fetch_dukascopy_l2.py`).
2. Replace the committed placeholder with the real dataset descriptor:
   - non-zero `data_sha256`, `is_synthetic: false`, `min_data_depth_sec >= 3600`,
     `data_source: real_single`.
3. Run the deterministic full cycle and emit a replayable artifact:
   - `PYTHONPATH=. python scripts/run_l2_full_cycle.py`.
4. Re-generate the maturity bundle — **no hand edits**:
   - `python -m tools.research.build_geosync_maturity_report`.
   - If `evaluate_promotion()` clears all nulls net-of-cost, the bundle records
     `TESTED_REAL_SINGLE`; otherwise it records the honest `REJECT` / `BLOCKED`.

## Then, for `Measured-Multi`

Repeat steps 1–4 for a **second independent session** (different day/symbol) and
satisfy `multi_session_replay` with sign-stable OOS IC. Only then may the claim
reach `Measured-Multi`.

## Hard stops (unchanged)

- A null matching/exceeding the signal → **reject**, archive sha-pinned negative.
- Negative net-of-cost → `BLOCKED_COST_MODEL`, never hidden.
- Any future-data leak / non-determinism → reject.
- Synthetic data can never advance the claim tier past `Instrumented`.
