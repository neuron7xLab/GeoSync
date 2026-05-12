# HANDOFF PACKET

1. Problem: prevent unsafe extrapolation promotion.
2. Refuses to claim universal truth.
3. Run: final_handoff_gate.sh
4. Verify: generate_artifact.py verify --artifact ...
5. Fail-closed on contract violations.
6. Exit codes: 0/2/3.
7. Contract files: schema/*.json, spec.json, purpose_contract.json, INVARIANTS.yaml.
8. Proof tests: test_generate_artifact.py + falsifier.py + brutal_e2e_proof.sh.
9. Missing before enterprise license: third-party audit, benchmark, compliance mapping.
10. Integrator must provide witness flow, null models, external falsification inputs.
