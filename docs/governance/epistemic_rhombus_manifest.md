# GeoSync Epistemic Rhombus manifest validation framework

> Status: governance policy index  
> Owner: platform-governance@geosync  
> Effective date: 2026-06-04  
> Validation: schema-backed JSONL validation via `scripts/validate_epistemic_rhombus_manifest.py`

## Purpose

The Epistemic Rhombus is a repository-level acceptance index for mapping
engineering assets to GeoSync governance expectations. It is not a cryptographic
ledger and does not claim tamper resistance. Repository history plus accepted
signed identity artifacts are the provenance boundary; this manifest records the
policy criteria and external gates that must supply the actual evidence.

## Rhombus criteria

| Axis | Criterion | Deployment blocker |
|------|-----------|--------------------|
| Axiomatic Basis | Every operational claim derives from a pre-registered data contract or defined physical/logical invariant. | Claim is discarded when no contract or invariant is named. |
| Structural Integrity | Every module/function has a falsification-ledger entry, mapped unit/property test, and fail-closed condition. | Module is not deployment-ready without test mapping and execution evidence. |
| Operational Determinism | Execution state is immutable where possible; side effects are isolated behind declared control gates. | State transition is rejected when replay cannot reproduce it in a compliant environment. |
| Provenance Governance | Significant claims, evidence, and deployment metadata are attributable to repository history and signed identity artifacts. | Deployment is blocked when attributable identity evidence is missing. |

## Required manifest fields

Each JSONL record in `governance/epistemic_rhombus_manifest.jsonl` must satisfy
`schemas/governance/epistemic_rhombus_manifest.schema.json` and contain:

- `record_id`: stable identifier for the criterion record; the numeric suffix is unbounded.
- `axis`: one of the four Rhombus criteria defined by the schema enum.
- `criterion`: the acceptance statement.
- `invariant_or_contract`: the named invariant, data contract, or logic contract.
- `validation_gate`: the acceptance or rejection rule expressed for operators.
- `fail_closed_condition`: the condition that blocks deployment.
- `evidence_required`: artifact classes required before promotion.
- `control_gates`: side-effect gates covered by the criterion, using the schema-defined `G<n>` pattern.
- `enforcement`: the external workflow or command that owns executable gate logic.
- `provenance`: required signed identity boundary for the record's evidence.

## Validation command

```bash
python scripts/validate_epistemic_rhombus_manifest.py
```

The validator streams JSONL records, applies the JSON schema directly, checks
that every schema-declared axis appears exactly once, and verifies that external
file-backed enforcement targets exist. It does not validate signatures or replace
Git object integrity; signature verification remains the responsibility of the
referenced external gates.
