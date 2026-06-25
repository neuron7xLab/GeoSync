# PHYSICS REVIEW PACKET

Status: PASS — 10/10 laws verified.

## Scope

Executable falsification-contract layer over the verified GeoSync physics
engines (Ricci/Kuramoto, metric consistency, causality, Landauer, empirical
falsification, numerical precision, governance). SEPARATE from and additive to
the canonical 112-invariant catalog, which is left untouched.

## Reproduce

```
python tools/validate_physics_contracts.py
python -m pytest tests/physics -q
python tools/build_physics_review_packet.py
```

## Laws and formal invariants

- **ricci_energy_nonincrease** (ricci_kuramoto): E(curv_{t+1}) <= E(curv_t) * (1 + tol), E(c) = sum_i (c_i - mean(c))^2  
  measured: max over a seeded flow trajectory of E(t+1) - E(t)*(1+tol); threshold: tol = 1e-9 (relative); any positive excess violates
- **kuramoto_order_parameter_bounds** (ricci_kuramoto): 0 <= R(t) <= 1 for all t (INV-K1)  
  measured: min and max of R over a seeded mean-field trajectory; threshold: R in [0.0, 1.0]; any R<0 or R>1 violates
- **metric_distortion_bound** (metric_consistency): distortion(D_orig, D_emb) - 1 <= max_distortion  
  measured: multiplicative bi-Lipschitz distortion of the embedded distance matrix; threshold: max_distortion explicit per call (no magic constant); default probe 0.05
- **causal_no_lookahead** (causality): source_time <= read_time for every admitted read  
  measured: max(source_time - read_time) over admitted reads; threshold: <= 0; any positive lookahead violates
- **provenance_identity** (causality): provenance_hash(observed) == expected <=> bytes unchanged  
  measured: equality of recomputed vs declared provenance hash; threshold: exact SHA-256 equality; any mismatch violates and tombstones
- **landauer_nonzero_erasure_cost** (landauer): bits > 0 and T > 0 => cost = k_B*T*ln2*bits > 0  
  measured: bit_erasure_cost(bits, T) for bits>0, T>0; threshold: > 0 strictly; a zero/negative cost violates
- **irreversible_cost_dominance** (landauer): C_irr >= C_rev, strict when irreversibility_score > 0  
  measured: C_irr - C_rev over a seeded action sweep; threshold: >= 0 always; > 0 when score>0
- **empirical_synthetic_only_rejection** (falsification): validate_empirical(witness) is True => witness.data_source == 'empirical' and provenance present and not leaked  
  measured: verdict of rejection_report on synthetic / shuffled / leaked / unprovenanced witnesses; threshold: all four adversarial witnesses must be REJECTED; only a clean provenanced empirical witness passes
- **numerical_drift_bound** (precision): deterministic_hash(run(seed)) constant; assert_numerical_stability fails on NaN/Inf or cond(A) > kappa_max  
  measured: hash equality across repeats; cond(A) vs kappa_max; finiteness; threshold: hash identical; cond(A) <= kappa_max (explicit); no NaN/Inf
- **law_requires_positive_and_negative_witness** (governance): for each blocking law L: collected(L.positive_witness) and collected(L.negative_control) and exists(L.evidence_output)  
  measured: count of blocking laws missing a witness, a negative control, or evidence; threshold: 0 missing; any missing fails promotion

## Evidence artifacts

- `evidence/physics/causality_report.json`
- `evidence/physics/falsification_report.json`
- `evidence/physics/governance_report.json`
- `evidence/physics/landauer_report.json`
- `evidence/physics/metric_consistency_report.json`
- `evidence/physics/precision_report.json`
- `evidence/physics/ricci_kuramoto_report.json`
