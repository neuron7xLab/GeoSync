# BLOCKED — release_gate `--deep` cannot reach exit 0

`python scripts/ci/release_gate.py --deep --json artifacts/release_gate/scorecard.json`
returns **exit 1 (VERDICT: RED)**. This file records, fail-closed, exactly why,
which artifacts are missing, the commands run, and the next deterministic
action per blocker. Full GREEN is **not** achievable this session without
fabricating market data (refused) or running multi-day architectural /
dependency campaigns that would recklessly break a 6500-file, 1599-test repo.

Scorecard: `artifacts/release_gate/scorecard.json` — **11 GREEN / 6 RED / 0 MANUAL of 17**.

---

## A. Proof gates in scope (work order E/G/H/K/M/Q) — 5 GREEN, 1 honest BLOCK

| Gate | Verdict | Evidence artifact |
|------|---------|-------------------|
| E.clean_clone | **GREEN** | `artifacts/release_gate/clean_clone.json` |
| H.falsification | **GREEN** (8/8 SURVIVED) | `artifacts/falsification/ledger.json` |
| K.execution | **GREEN** (OUT_OF_SCOPE + firewall) | `artifacts/execution/contract.json` |
| M.benchmarks | **GREEN** (determinism + budget) | `artifacts/benchmarks/baseline.json` |
| Q.replication | **GREEN** (hash-lock matches) | `artifacts/replication/expected_hashes.json` |
| **G.real_data** | **RED — BLOCKED** | `artifacts/evidence/real_data_manifest.json` |

### BLOCKER G.real_data — real data exists but has NO license/provenance

* **Why:** genuinely real market data *is* present on disk
  (`data/askar_full/panel_*.parquet`, ~9 years, 53 assets, from "Askar/OTS
  Capital" via an internal API — see `agent/providers.py`). But it carries
  **no `license.txt`, no provenance/origin marker**, and is registered only as
  a **P0 escalation task** (received as resampled OHLC bars when L2 was
  promised), not an audited/licensed source. The repo's own
  `research/systemic_risk/real_data_contract.py` fails closed without a
  license; the synthetic fixtures all carry `forbidden_use: not for live
  trading`. Attesting a tier over unlicensed data would **fabricate
  provenance** — refused.
* **Missing artifact:** `artifacts/evidence/real_data/<id>.json` per
  `scripts/ci/real_data_probe.py::REQUIRED_FIELDS` with a real `data_sha256`,
  verifiable `license_provenance`, `replay_command`, `git_dirty=false`.
* **Next deterministic action:** obtain a written license / public-domain
  attestation from OTS Capital (or substitute a genuinely public-domain series,
  e.g. FRED), stage the manifest, `python scripts/ci/real_data_probe.py`.

---

## B. Pre-existing gating failures (NOT in the original work order)

These were already machine-RED on `origin/main`. One is now closed; the rest
are real campaigns, not session fixes. None were faked or papered over.

### B.1 — D.manifest → **CLOSED this session (now GREEN)**

* **Root cause:** `MANIFEST.sha256` was a stale 2827-entry snapshot while the
  tree holds 6528 tracked files (1458 entries broken). It also had **no
  committed generator** (so it rotted silently), and the probe's path parser
  used character-based `str.lstrip("./")`, which ate the leading dot of
  dotfiles (`./.claude/x → claude/x`) and reported 619 false "missing".
* **Fix:** added `scripts/ci/generate_manifest.py` — a principled, reviewable
  generator covering every tracked file **except** itself and the volatile
  `artifacts/` tree (machine outputs the gate regenerates each `--deep` run;
  their integrity is carried by each artifact's own `artifact_sha256`).
  Corrected the probe's prefix-stripping (a genuine bug fix that *strengthens*
  coverage). Regenerated to 6364 current entries; cold-verify clean and stable
  across `--deep` runs.

### B.2 — C.dep_truth → **RED (48-drift dual-lock reconciliation campaign)**

* **Root cause (verified):** `tools/deps/validate_dependency_truth.py
  --exit-on-drift` reports **48 actionable drifts** after the accepted D1
  backlog filter — NOT the 5 floor bumps a first glance suggests. Breakdown:
  * **35× D3** — `requirements.lock` vs `requirements-scan.lock` pin different
    versions of the same package (aiodns 3.5.0 vs 3.6.1, altair 5.5.0 vs 6.0.0,
    numpy/scipy/grpcio/opentelemetry-*, …).
  * **2× D2** — `requirements-scan.lock` pins below the pyproject floor.
  * **3× D4** — Dockerfile (`coherence_bridge/`, `cortex_service/`,
    `sandbox/`) install paths diverge.
  * **1× D6** — deptry finding. **3× D7** — security pins (cryptography, pyjwt,
    python-multipart).
* **Why not a session fix:** naively raising the 5 requirements.txt floors
  desynced `requirements-scan.lock` and *exposed* the 35 D3 divergences
  (reverted). Honest resolution requires regenerating **both** lock files
  consistently (`pip-compile` over the full 43-dep tree incl. torch≥2.11 /
  jax / numba), syncing 3 Dockerfiles, and resolving deptry + the D7 security
  pins — then verifying via a full install. Unverifiable here without that
  install; high blast radius.
* **Next action:** `pip-compile requirements.txt` and `requirements-scan.lock`
  from a single resolution against `constraints/security.txt`; align
  Dockerfiles; re-run the validator to zero.

### B.3 — B.path_hacks → **RED (35-file shipped-script refactor)**

* **Root cause:** 66 first-party files mutate `sys.path`; ~35 are in
  wheel-shipped packages (`scripts/`, `tools/`, `backtest/`) using a uniform
  standalone bootstrap `if str(ROOT) not in sys.path: sys.path.insert(0, ROOT)`.
  The remaining ~29 are in non-shipped auxiliary dirs (`examples/`, `bench/`,
  `spikes/`, …) that legitimately bootstrap for `python path/script.py`.
* **Why not a session fix:** removing the bootstrap breaks standalone
  invocation unless each script is verified to run only as `-m`/entry-point.
  That is a per-file refactor + runtime verification across 35 files; a
  half-done pass that breaks scripts is worse than RED. (The repo already
  treats this gate as law — see `scripts/ci/check_math_boundaries.py`.)
* **Next action:** convert shipped scripts to package-relative imports invoked
  via console entry points / `python -m`; verify each; only then remove the
  `sys.path` bootstrap.

### B.4 — B.single_pkg / B.src_imports / B.wheel → **RED (architectural migration)**

* **Root cause (verified):** `geosync/` (neural-research modules) and
  `src/geosync/` (application SDK) are **two different packages**, both
  referenced by entry points — not a deletable duplicate. 18 top-level
  packages (`core`, `scripts`, `application`, …) are shipped because the entry
  points import them. Of the 19 `src.*` imports, 7 files import
  `src.audit` / `src.data` / `src.risk` / `src.security`, which have **no
  top-level equivalent** (`audit/` and `data/` aren't even Python packages).
* **Why not a session fix:** "single `geosync` package" requires collapsing the
  flat multi-package layout under one root and rewriting hundreds of imports
  across the codebase + 1599 test files — a multi-week migration. Forcing it
  would either break everything or require weakening the probes (forbidden).
* **Next action:** a dedicated package-consolidation epic (choose one canonical
  root; migrate infra namespaces; rewrite imports; restrict
  `[tool.setuptools.packages.find]`).

---

## C. Commands run (this session)

```bash
python scripts/ci/check_claim_boundary.py                                   # exit 0
python scripts/count_invariants.py                                          # 108, exit 0
python scripts/ci/generate_manifest.py                                      # 6364 entries
python scripts/ci/generate_manifest.py --check                             # cold-verify clean
python tools/deps/validate_dependency_truth.py --exit-on-drift             # 48 actionable drifts (campaign)
python scripts/ci/release_gate.py --deep --json artifacts/release_gate/scorecard.json  # 11 GREEN / 6 RED, exit 1
python -m pytest tests/scripts/test_proof_gates.py -q                       # 25 passed
```

## D. The only valid GREEN state

`release_gate --deep` returns exit 0 **only** when every gating probe is GREEN.
While `G.real_data` is BLOCKED (no licensed data) and `C.dep_truth` /
`B.path_hacks` / `B.single_pkg` / `B.src_imports` / `B.wheel` remain genuine
campaigns, exit 1 is the **correct fail-closed verdict** — not a regression.
Truth is the executable artifact, not the paragraph.
