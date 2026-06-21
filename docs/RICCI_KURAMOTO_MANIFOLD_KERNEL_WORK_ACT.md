# GeoSync — Act of Work: Ricci-Kuramoto Manifold Kernel

Status: canonical planning / execution-control artifact  
Scope: `ricci_microstructure_v1` → `ricci_kuramoto_manifold_kernel_v1`  
Repository category: verification-first quantitative research infrastructure  
Claim boundary: hypothesis-tier until real-data evidence, null baselines, cost model, and reproduction gates pass

---

## 1. Purpose

This act formalizes the work required to move the current GeoSync Ricci research line from an instrumented hypothesis into a falsifiable, evidence-bearing, reviewer-grade research kernel.

The target is not to claim alpha, not to market a trading bot, and not to dress a chart as physics. The target is to build a deterministic Ricci-Kuramoto manifold kernel that can test whether market microstructure geometry and phase synchronization contain predictive information about regime breaks, volatility expansion, liquidity fragility, or systemic transition states.

Canonical framing:

```text
L2_ORDERBOOK_STREAM
  → microstructure tensors
  → dynamic multiplex graph
  → Ricci curvature field
  → Kuramoto phase field
  → curvature-modulated coupling
  → regime / fragility score
  → null-tested evidence artifact
```

---

## 2. Current Repository State

The repository already contains the correct epistemic spine:

1. Claim promotion is explicitly tiered.
2. Evidence-bearing artifacts require deterministic provenance.
3. Forbidden claim language is separated from admissible status language.
4. Placeholder artifacts are not allowed to support measured claims.
5. `ricci_microstructure_v1` is correctly marked as `HYPOTHESIS`, `INSTRUMENTED`, and `NOT EVIDENCE-BEARING`.

The existing Ricci implementation operates on cross-asset return-correlation graphs. This is useful as a structural adapter and deterministic topology test surface, but it is not yet an L2 microstructure Ricci manifold and must not be described as predictive evidence.

---

## 3. Defect / Gap Statement

Current blocker:

```text
B-001: No final Ricci-Kuramoto manifold kernel exists.
```

Impact:

```text
GeoSync cannot yet promote the Ricci line from HYPOTHESIS to MEASURED.
GeoSync cannot yet claim predictive power.
GeoSync cannot yet support external reviewer-grade scientific ranking for this mechanism.
```

Root cause:

```text
The repository has a strong verification framework and a Ricci correlation-graph adapter, but lacks real L2 market microstructure ingestion, deterministic manifold construction, Kuramoto phase extraction, null-superiority tests, and signed multi-session evidence artifacts.
```

---

## 4. Work Packages

### WP-1 — Data Contract: Real L2 Session Substrate

Objective: introduce immutable real market microstructure sessions.

Required outputs:

- `data/contracts/l2_session.schema.json`
- `configs/ricci_kuramoto/l2_session.yaml`
- `artifacts/data/l2_sessions/<session_id>/manifest.json`
- SHA-256 for raw input, normalized input, config, and environment
- session metadata: venue, symbol, date, timezone, depth level, sampling mode, gaps, dropped packets, normalization rules

Acceptance criteria:

```text
PASS if every session has immutable hash provenance.
PASS if replay can reconstruct the exact normalized tensor.
FAIL if any dataset is mutable, undocumented, or hashless.
```

---

### WP-2 — Microstructure Tensor Builder

Objective: transform raw L2/order-book events into deterministic tensors.

Canonical features:

- bid/ask depth at levels 1-5 or 1-10
- spread
- mid-price
- order-book imbalance
- depth imbalance
- trade intensity when available
- signed order-flow proxy
- realized volatility window
- liquidity withdrawal / replenishment rate
- gap / halt markers

Required output:

```text
X[t, node, channel]
```

Where nodes may represent:

- price levels
- liquidity bands
- assets
- venues
- sectors
- synthetic exposure nodes

Acceptance criteria:

```text
PASS if tensors are deterministic under fixed input.
PASS if missing/non-finite values are handled fail-closed or explicitly masked.
FAIL if preprocessing silently imputes market structure without audit metadata.
```

---

### WP-3 — Dynamic Multiplex Graph Constructor

Objective: construct market microstructure graphs per rolling window.

Required graph layers:

1. Price-level liquidity graph.
2. Order-flow coupling graph.
3. Volatility co-movement graph.
4. Optional cross-asset coupling graph.

Edge construction candidates:

- distance-weighted liquidity adjacency
- correlation / partial-correlation adjacency
- transfer entropy candidate, only if sample size supports it
- Granger-style directed proxy, only as baseline unless validated
- exposure/coupling matrix when exogenous structure exists

Acceptance criteria:

```text
PASS if graph construction is deterministic and versioned.
PASS if edge sparsification threshold is recorded.
PASS if disconnected graph behavior is explicit.
FAIL if graph topology changes without a config hash.
```

---

### WP-4 — Ricci Curvature Field

Objective: compute graph curvature over dynamic microstructure graphs.

Required methods:

- Ollivier-Ricci adapter for transport-like graph curvature.
- Forman-Ricci adapter as computationally cheaper baseline.
- curvature summary fields:
  - `kappa_min`
  - `kappa_mean`
  - `kappa_std`
  - `negative_curvature_mass`
  - `curvature_shock_index`
  - `curvature_persistence`

Acceptance criteria:

```text
PASS if curvature fields are reproducible from graph artifacts.
PASS if zero-edge and disconnected cases return declared neutral/fail-closed output.
PASS if curvature output has no predictive language by itself.
FAIL if curvature is interpreted as market law without falsification.
```

---

### WP-5 — Kuramoto Phase Field

Objective: convert microstructure dynamics into phase synchronization observables.

Phase source candidates:

- Hilbert phase of normalized flow imbalance.
- phase of volatility burst cycle.
- phase of liquidity replenishment/depletion cycle.
- phase of spread expansion/compression cycle.
- phase of cross-asset return/flow oscillation.

Required fields:

- `theta_i(t)` phase
- `omega_i(t)` natural frequency estimate
- `R(t)` global order parameter
- `R_local(t)` local synchronization
- `phase_velocity`
- `phase_slip_rate`
- `synchronization_persistence`

Acceptance criteria:

```text
PASS if phase extraction is explicitly defined and deterministic.
PASS if phase source is recorded in config.
PASS if shuffled-phase null destroys the claimed signal.
FAIL if phase is arbitrary, decorative, or unbound to a market observable.
```

---

### WP-6 — Ricci-Kuramoto Coupling Kernel

Objective: integrate curvature and synchronization into one deterministic inference kernel.

Candidate kernel:

```text
K_ij(t) = base_coupling_ij(t) * sigmoid(-alpha * kappa_ij(t)) * liquidity_weight_ij(t)
```

Regime score candidate:

```text
fragility_score(t) = f(
  negative_curvature_mass(t),
  curvature_shock_index(t),
  R(t),
  phase_slip_rate(t),
  spread_expansion(t),
  depth_withdrawal(t),
  volatility_response(t)
)
```

Primary target:

```text
predict regime break / liquidity fragility / volatility expansion
```

Non-target:

```text
predict next tick direction as headline claim
```

Acceptance criteria:

```text
PASS if output is a calibrated risk/regime score, not an alpha claim.
PASS if the same input produces bit-stable or tolerance-stable output.
PASS if all hyperparameters are config-bound and hashed.
FAIL if parameters are tuned after seeing the target without preregistration.
```

---

### WP-7 — Baselines and Null Falsifiers

Objective: prove the kernel is not just detecting obvious volatility or overfitting noise.

Required baselines:

- naive volatility baseline
- spread-only baseline
- depth-imbalance baseline
- return autocorrelation baseline
- VAR/HMM/GARCH baseline where appropriate
- current return-correlation Ricci adapter baseline

Required nulls:

- shuffled phase
- shuffled graph edges
- time-permuted labels
- block bootstrap
- asset/session permutation
- randomized coupling matrix
- synthetic control with known planted transition

Acceptance criteria:

```text
PASS if Ricci-Kuramoto score beats declared baselines OOS.
PASS if nulls fail to reproduce the measured effect.
PASS if multiple-testing correction is applied.
FAIL if nulls match or exceed the claimed signal.
```

---

### WP-8 — Evaluation Protocol

Objective: convert the kernel into a measured research result only if it survives rigorous evaluation.

Required metrics:

- AUC / PR-AUC for regime-break label
- Brier score for calibrated probability
- lead-time before event
- false-positive rate per hour/session
- stability across sessions
- ablation contribution:
  - Ricci only
  - Kuramoto only
  - microstructure baseline only
  - Ricci + Kuramoto fused
- cost-aware trading simulation only as secondary evidence, not headline proof

Hard-pass candidate:

```text
AUC lower confidence bound >= 0.60 against all primary baselines
AND phase/edge/null tests fail to reproduce the signal
AND OOS multi-session replay passes
AND cost model does not erase downstream utility if trading simulation is included
```

Hard-fail candidate:

```text
AUC <= 0.52 OOS
OR null models match/exceed score
OR signal disappears after spread/volatility baseline
OR artifact cannot be replayed
OR data provenance is incomplete
```

---

### WP-9 — Evidence Artifact Contract

Objective: every run must emit a machine-checkable artifact.

Required artifact fields:

```text
run_id
git_sha
git_dirty
data_sha256
config_sha256
seed
timestamp_utc
method_version
session_id
input_window_sec
prediction_horizon_sec
target_definition
score
score_source
uncertainty
baseline
null_results
ablation_results
falsification_status
decision
claim_tier
artifact_role
replay_command
```

Required locations:

```text
schemas/research/ricci_kuramoto_artifact.schema.json
artifacts/runs/ricci_kuramoto_manifold_kernel_v1/<run_id>/result.json
artifacts/runs/ricci_kuramoto_manifold_kernel_v1/<run_id>/VERDICT.md
```

Acceptance criteria:

```text
PASS if schema validation accepts the artifact.
PASS if replay command reproduces the same result.
FAIL if artifact is placeholder but supports a claim.
```

---

### WP-10 — CI / Release / Reproduction Gates

Objective: make the result externally checkable.

Required gates:

- `claims-evidence-gate.yml`
- `ricci-kuramoto-artifact-gate.yml`
- `replay-ricci-kuramoto.yml`
- `null-baseline-gate.yml`
- `cost-model-gate.yml` when trading simulation is claimed
- `external-reproduction-capsule.yml`

Required evidence bundle:

```text
commands
raw_logs
exit_codes
manifest
checksums
schema_validation
CI_status
environment_drift_result
```

Acceptance criteria:

```text
PASS if clean checkout can reproduce the artifact.
PASS if CI preserves raw logs and checksums.
FAIL if green CI exists without evidence bundle.
```

---

## 5. Claim Promotion Plan

Initial status:

```text
HYPOTHESIS
```

Allowed progression:

```text
HYPOTHESIS
  → PREREGISTERED
  → INSTRUMENTED
  → TESTED_SYNTHETIC
  → TESTED_REAL_SINGLE
  → TESTED_REAL_MULTI
  → MEASURED
  → REPLICATED
```

Promotion rules:

1. No claim may be promoted by prose.
2. No synthetic result may promote to `MEASURED`.
3. No artifact without data hash may support evidence.
4. No predictive claim may exist without baseline and null comparison.
5. No public headline may exceed the tier recorded in `CLAIMS.md`.

---

## 6. Official Rank Target

Current rank:

```text
Ambitious falsifiable quantitative research platform / HYPOTHESIS-tier predictive mechanism
```

Target rank after WP-1 through WP-10:

```text
Evidence-bearing Ricci-Kuramoto microstructure research kernel / MEASURED-tier if real multi-session results pass declared gates
```

Target rank after independent reproduction:

```text
Externally reproducible quantitative research software artifact / REPLICATED-tier candidate
```

External publication route:

```text
1. Zenodo DOI for release artifact.
2. JOSS-style software paper if installation, tests, documentation, examples, and research scope are stable.
3. arXiv/preprint only after claim ledger and artifacts support every headline claim.
```

Forbidden rank language until gates pass:

```text
verified alpha
proven market physics
production trading model
predictive law
institutional-grade signal
```

Allowed rank language now:

```text
verification-first research infrastructure
falsifiable Ricci/Kuramoto market-structure hypothesis platform
instrumented hypothesis-tier research kernel
```

---

## 7. Deliverables Checklist

| ID | Deliverable | Path | Status |
| --- | --- | --- | --- |
| D-001 | L2 session schema | `data/contracts/l2_session.schema.json` | TODO |
| D-002 | L2 ingestion module | `src/geosync/data/l2_ingest.py` | TODO |
| D-003 | Microstructure tensor builder | `src/geosync/features/microstructure_tensor.py` | TODO |
| D-004 | Dynamic graph constructor | `src/geosync/features/dynamic_graph.py` | TODO |
| D-005 | Forman/Ollivier Ricci field | `src/geosync/features/ricci_microstructure.py` | TODO |
| D-006 | Kuramoto phase extractor | `src/geosync/features/kuramoto_phase.py` | TODO |
| D-007 | Fusion kernel | `src/geosync/models/ricci_kuramoto_kernel.py` | TODO |
| D-008 | Null/baseline suite | `research/ricci_kuramoto/nulls.py` | TODO |
| D-009 | Evaluation protocol | `research/ricci_kuramoto/evaluate.py` | TODO |
| D-010 | Artifact schema | `schemas/research/ricci_kuramoto_artifact.schema.json` | TODO |
| D-011 | Replay command | `scripts/replay_ricci_kuramoto.py` | TODO |
| D-012 | Verdict generator | `scripts/write_ricci_kuramoto_verdict.py` | TODO |
| D-013 | CI gate | `.github/workflows/ricci-kuramoto-gate.yml` | TODO |
| D-014 | Claims update | `CLAIMS.md` | TODO after evidence |
| D-015 | Public status update | `README.md` | TODO after claims update |

---

## 8. Minimum One-Command Interface

Required final command:

```bash
python -m research.ricci_kuramoto.run \
  --config configs/ricci_kuramoto/l2_session.yaml \
  --out artifacts/runs/ricci_kuramoto_manifold_kernel_v1/latest
```

Required verification command:

```bash
python scripts/replay_ricci_kuramoto.py \
  --artifact artifacts/runs/ricci_kuramoto_manifold_kernel_v1/latest/result.json
```

Required gate command:

```bash
python scripts/validate_research_artifact.py \
  --schema schemas/research/ricci_kuramoto_artifact.schema.json \
  --artifact artifacts/runs/ricci_kuramoto_manifold_kernel_v1/latest/result.json
```

---

## 9. Final Acceptance Verdict Logic

```text
FINAL_READY =
  real_data_hash_present
  AND config_hash_present
  AND deterministic_replay_passed
  AND schema_validation_passed
  AND nulls_failed_to_explain_signal
  AND baselines_beaten_OOS
  AND ablations_recorded
  AND claim_tier_updated
  AND forbidden_language_gate_passed
  AND external_reproduction_path_exists
```

If any condition fails:

```text
VERDICT = NOT FINAL
claim_tier = HYPOTHESIS or BLOCKED_*
```

If all conditions pass on a single real session only:

```text
VERDICT = TESTED_REAL_SINGLE
claim_tier = not yet MEASURED
```

If all conditions pass across multiple real sessions:

```text
VERDICT = MEASURED_CANDIDATE
claim_tier = eligible for MEASURED after claims-evidence gate
```

If independent clean-room replay passes:

```text
VERDICT = REPLICATED_CANDIDATE
claim_tier = eligible for REPLICATED
```

---

## 10. Final Work Act Conclusion

GeoSync already has a serious verification-first architecture and an honest claim-control system. The current Ricci line is correctly constrained as hypothesis-tier and not evidence-bearing. The missing work is not more language, branding, or conceptual decoration. The missing work is the final Ricci-Kuramoto manifold kernel: real L2 data contract, microstructure tensorization, dynamic graph construction, curvature field, phase synchronization field, fused kernel, null baselines, OOS evaluation, signed artifacts, and replayable CI evidence.

Until those gates pass, the correct official rank is:

```text
HYPOTHESIS-tier verification-first quantitative research kernel
```

After the gates pass, the legitimate target rank becomes:

```text
MEASURED-tier reproducible Ricci-Kuramoto market-microstructure research artifact
```

No weaker standard is acceptable, because a model that cannot survive its own falsifiers is not research. It is just a confident parrot wearing a lab badge.
