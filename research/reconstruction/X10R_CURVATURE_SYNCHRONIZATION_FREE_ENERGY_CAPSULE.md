# X10R Curvature-Synchronization Free-Energy Reconstruction Capsule

**Repository:** `neuron7xLab/GeoSync`  
**Base commit:** `196063b36512ba55048d3f49d08224c2c316ab16`  
**Date:** 2026-06-17  
**Status:** `EXECUTION_CONTRACT_READY / NOT_CI_VERIFIED_IN_THIS_COMMIT`  

This capsule turns the physics decomposition into an executable research contract. It is deliberately not a trading claim, alpha claim, AGI claim, or proof that markets obey universal physics. It is a verification-first reconstruction lane for falsifiable market-structure hypotheses using graph geometry, nonlinear synchronization, thermodynamic-style objectives, neuro-symbolic control, and fail-closed governance.

## 1. Claim Boundary

GeoSync may claim only the following after this capsule is executed and CI evidence is attached:

```text
A reproducible software pipeline can construct market-derived graphs, compute topology/curvature/synchronization/energy observables, enforce explicit invariants, and reject corrupted or unsupported states through executable falsifiers.
```

GeoSync must not claim:

- profitable trading edge;
- universal market physics;
- biological realism beyond explicit neuro-control analogues;
- production readiness without CI, artifact hashes, and negative-evidence ledger entries.

## 2. First-Principles Source Map

| Layer | Repository boundary | Required invariant |
| --- | --- | --- |
| Topological geometry | `core/indicators/gauss_bonnet.py`, `physics_contracts/catalog.yaml`, `tests/physics_contracts/` | Exact `Σ K(x) - χ(G) == 0` over rationals for finite simple undirected graphs |
| Curvature stress | `core/indicators/ricci.py`, `core/physics/forman_ricci.py` | Curvature bounds and margin escalation are policy gates, not universal physical constants |
| Kuramoto synchronization | `core/kuramoto/`, `core/indicators/multiscale_kuramoto.py`, `core/config/kuramoto_ricci.py` | Canonical ownership of scale: `θ_dot = ω + K * A_norm * sin(Δθ)`, reject negative coupling |
| Energy ontology | `core/energy.py`, `core/physics/`, `tests/physics/test_energy_contract.py` | Do not conflate `operational_cost_energy` with canonical `thermo_free_energy = U - T*S` |
| Neuro-control | `core/neuro/`, `tests/physics_contracts/`, contradiction ledger | Neuromodulatory trajectories must expose falsifiable correlation and boundedness witnesses |
| Governance | `governance/verification_protocol.py`, `data/governance/`, `.claude/commit_acceptors/` | Weighted score, conformance bands, and weakest-link clamp execute from declared artifacts |
| Provenance | `artifacts/`, `docs/audit/`, `data/audit/` | Residuals, reports, and evidence must be hash-pinned and reproducible |

## 3. Reconstruction Objective

Implement **First-Principles Reconstruction of Market Curvature-Synchronization Free-Energy Landscape with Topological Integrity Enforcement**.

The pipeline must produce a verifiable X10R artifact bundle:

```text
artifacts/x10r_reconstruction/
  manifest.json
  input_graph.json
  curvature_edges.json
  gauss_bonnet_residuals.json
  kuramoto_trajectory.json
  free_energy_landscape.json
  neuro_control_trace.json
  falsifiers.json
  verdict.json
  ro_crate_metadata.json
```

The final verdict is valid only if all blocking gates pass and every generated artifact is SHA-256 pinned in `manifest.json`.

## 4. Execution Lanes

### Lane A: Input Graph Construction

- Accept synthetic or empirical market microstructure input.
- Build a finite simple undirected weighted graph.
- Reject directed graphs, self-loops, multigraphs, empty graphs, non-finite weights, and invalid timestamps.
- Normalize adjacency with zero diagonal and one declared scale owner.
- Derive natural frequencies `ω_i` from returns, volatility, or explicitly declared synthetic fixtures.

### Lane B: Curvature and Topological Integrity

- Compute Forman-Ricci and Ollivier-Ricci observables where valid.
- Compute Knill vertex curvature through exact rational arithmetic.
- Assert `gauss_bonnet_residual(graph) == 0`.
- Run inverse falsifier: tamper one vertex curvature value and require fail-closed rejection.

### Lane C: Curved-Manifold Kuramoto Dynamics

- Integrate `θ_dot = ω + K * Σ_j A_norm[i,j] * sin(θ_j - θ_i)`.
- Reject `K < 0`, non-finite phase vectors, and mismatched graph/phase dimensions.
- Track `R(t) = |mean(exp(iθ))|` and verify `0 <= R(t) <= 1`.
- Add curvature feedback only as an explicit optional operator, never by silently rescaling `K` twice.

### Lane D: Free-Energy Landscape

- Compute canonical `F = U - T*S` separately from operational cost.
- Assert unit ontology: canonical physics quantity and dimensionless operational-cost objective are not interchangeable.
- Verify boundedness of all declared state variables.
- Run sign falsifier: increasing entropy must decrease canonical free energy when `T > 0`, while it must not lower the operational entropy-penalty objective.

### Lane E: Neuro-Symbolic Control

- Emit dopamine/serotonin or declared neuromodulatory control traces only as operational control analogues.
- Enforce bounded signals and declared correlation invariant.
- Fail on positive or flat correlation where the contract requires inverse relation.
- Write all contradictions or failed witnesses into the negative-evidence ledger.

### Lane F: Governance and Provenance

- Execute `VerificationProtocol` from the declared kernel, not copied constants.
- Apply weakest-link clamp: a high numerical score cannot promote an artifact above missing evidence.
- Emit RO-Crate-compatible metadata for code commit, input fixtures, command lines, outputs, and hashes.
- Produce a final verdict: `PASS`, `FAIL`, or `BLOCKED_BY_MISSING_EVIDENCE`.

## 5. Acceptance Gates

A commit can call this capsule complete only when:

1. `gauss_bonnet_residuals.json` reports exact rational zero for all valid graph families.
2. The tampered-curvature falsifier fails closed.
3. Kuramoto trajectories are deterministic under fixed seed and golden reference tolerance.
4. `R(t)` remains in `[0, 1]` for every emitted step.
5. Negative coupling is rejected before integration.
6. Canonical free energy and operational-cost energy are verified as separate functions with opposite entropy semantics.
7. Neuro-control invariants produce both positive witnesses and inverse falsifiers.
8. Governance verdict is produced through `VerificationProtocol.evaluate`.
9. Manifest includes SHA-256 for every output artifact.
10. CI evidence is attached; otherwise the maximum honest state is `BLOCKED_BY_MISSING_EVIDENCE`.

## 6. Minimal Command Contract

The target implementation should converge to a single local command:

```bash
python -m research.reconstruction.x10r_reconstruct \
  --input fixtures/x10r/market_microstructure.json \
  --out artifacts/x10r_reconstruction \
  --seed 1729 \
  --strict
```

Expected terminal terminal state:

```text
X10R_RECONSTRUCTION: PASS|FAIL|BLOCKED_BY_MISSING_EVIDENCE
manifest: artifacts/x10r_reconstruction/manifest.json
verdict: artifacts/x10r_reconstruction/verdict.json
```

## 7. Stop Rules

Stop immediately and mark the capsule failed if:

- topology is outside finite simple undirected graph domain;
- any exact topological residual is non-zero;
- any physics contract is missing a witness;
- any artifact lacks a hash;
- any governance score bypasses weakest-link clamp;
- any market/trading claim is emitted from this pipeline.

## 8. Current Execution Verdict

```text
CAPSULE_SPEC_CREATED
BASE_COMMIT_PINNED=196063b36512ba55048d3f49d08224c2c316ab16
CODE_EXECUTION_NOT_PERFORMED_IN_THIS_COMMIT
CI_EVIDENCE_NOT_ATTACHED
HONEST_STATE=EXECUTION_CONTRACT_READY
NEXT_STATE=IMPLEMENT_PIPELINE_AND_ATTACH_ARTIFACT_HASHES
```

This is the operational bridge from the repository's existing physics contracts to an end-to-end reconstruction pipeline. It intentionally refuses ceremonial completion until executable code, falsifiers, hashes, and CI evidence exist. Humanity may continue calling Markdown "done"; this file does not.