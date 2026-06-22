# BLOCKED — release_gate `--deep` cannot reach exit 0

`python scripts/ci/release_gate.py --deep --json artifacts/release_gate/scorecard.json`
returns **exit 1 (VERDICT: RED)**. This file records, fail-closed, exactly why,
which artifacts are missing, the commands run, and the next deterministic
action. Full GREEN is **not** achievable without either fabricating market
data (refused) or a repackaging/manifest campaign outside this work order.

Scorecard: `artifacts/release_gate/scorecard.json` — **10 GREEN / 7 RED / 0 MANUAL of 17**.

---

## A. Proof gates in scope (this work order: E/G/H/K/M/Q)

| Gate | Verdict | Evidence artifact |
|------|---------|-------------------|
| E.clean_clone | **GREEN** | `artifacts/release_gate/clean_clone.json` |
| H.falsification | **GREEN** (8/8 SURVIVED) | `artifacts/falsification/ledger.json` |
| K.execution | **GREEN** (OUT_OF_SCOPE + firewall) | `artifacts/execution/contract.json` |
| M.benchmarks | **GREEN** (determinism + budget) | `artifacts/benchmarks/baseline.json` |
| Q.replication | **GREEN** (hash-lock matches) | `artifacts/replication/expected_hashes.json` |
| **G.real_data** | **RED — BLOCKED** | `artifacts/evidence/real_data_manifest.json` |

Five of six MANUAL gates were converted to executable, machine-verified GREEN
probes. The sixth is the single in-scope blocker:

### BLOCKER G.real_data — no real-market-data evidence

* **Why:** the repository ships only synthetic single-session fixtures.
  `data/sample_ohlc.csv` is marked `forbidden_use: not for live trading`; all
  18 ingestion adapters under `artifacts/d002j/ingestion/staged/` are
  stub-only (empty). There is no venue/symbol/license/provenance manifest with
  ≥2 independent sessions. A real-data tier (`MEASURED_SINGLE`/`MEASURED_MULTI`)
  cannot be attested, and **must not be fabricated**.
* **Missing artifact(s):** `artifacts/evidence/real_data/<id>.json` conforming
  to the schema in `scripts/ci/real_data_probe.py` (`REQUIRED_FIELDS`), with a
  non-zero `data_sha256`, real `license_provenance`, a `replay_command`, and
  `git_dirty=false`.
* **Next deterministic action:**
  1. Stage ≥1 real venue session file under `artifacts/evidence/real_data/`.
  2. For a `MEASURED_MULTI` tier, stage ≥2 sessions with distinct
     `time_start_utc` windows and `independent_sessions ≥ 2`.
  3. Re-run `python scripts/ci/real_data_probe.py` (exit 0 ⇒ tier attested).

---

## B. Pre-existing gating failures (NOT in this work order)

These six probes are **existing machine-RED gates** that were already failing
on `origin/main` before this work — they are not MANUAL items and were not
touched here. They are listed for an honest total, not closed by this change.

| Gate | Verdict | Root cause (machine evidence) | Deterministic next action |
|------|---------|-------------------------------|---------------------------|
| B.src_imports | RED | 19 first-party files `import src.*` (e.g. `application/api/authorization.py`) | rewrite imports to the canonical package path |
| B.path_hacks | RED | 70 first-party files mutate `sys.path` | remove runtime `sys.path` insertion |
| B.single_pkg | RED | both `geosync/` and `src/geosync/` exist (dual canonical root) | collapse to a single package root |
| B.wheel | RED | wheel ships 18 top-level packages, 16 outside `geosync*` | restrict distributed packages or move code under `geosync/` |
| C.dep_truth | RED | dependency-truth validator reports drift | reconcile `requirements.lock` ↔ declared deps |
| D.manifest | RED | `MANIFEST.sha256` cold-verify: 1458/2827 entries broken | regenerate `MANIFEST.sha256` against the current tree |

These are large, independent campaigns (repackaging, manifest regeneration,
import-architecture migration). Forcing them GREEN here would either weaken the
existing gating probes (forbidden) or exceed this work order's scope.

---

## C. Commands run

```bash
python scripts/ci/check_claim_boundary.py                                   # exit 0
python scripts/count_invariants.py                                          # 108, exit 0
python scripts/ci/probe_clean_clone.py                                      # PASS
python scripts/ci/falsification_ledger.py                                   # promotion_allowed=True (8/8)
python scripts/ci/execution_contract.py                                     # PASS (OUT_OF_SCOPE)
python scripts/ci/benchmark_spine.py --establish                            # ESTABLISHED
python scripts/ci/real_data_probe.py                                        # exit 1 (BLOCKED)
python scripts/ci/replication_packet.py --verify                            # PASS
python scripts/ci/release_gate.py --deep --json artifacts/release_gate/scorecard.json  # exit 1 (RED)
python -m pytest tests/scripts/test_proof_gates.py -q                       # 25 passed
```

## D. The only valid GREEN state

`release_gate --deep` returns exit 0 **only** when every gating probe is GREEN.
While `G.real_data` is BLOCKED (and the pre-existing B/C/D gates are RED), exit 1
is the **correct fail-closed verdict** — not a regression. Truth is the
executable artifact, not the paragraph.
