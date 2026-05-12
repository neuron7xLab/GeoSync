# Enterprise License Readiness — Gap Analysis

## 1) Blockers to standalone enterprise license
1. No formal release manifest (checksums + signed bill of materials).
2. No versioned API compatibility policy (breaking-change contract).
3. CI gate does not yet include static analysis + dependency CVE scan + reproducibility replay matrix.
4. No immutable audit evidence bundle format (release-time evidence pack).
5. No benchmark dossier quantifying false-positive reduction and cost/latency impact.
6. No legal/commercial boundary docs (license terms, warranty/disclaimer, support SLA).
7. No formal operational runbook for incidents + paging + RTO/RPO targets.
8. No external independent verification report attached.

## 2) Exact files to add/change
### Add
- `artifacts/inference_extrapolation_validator/release/RELEASE_MANIFEST.json`
- `artifacts/inference_extrapolation_validator/release/EVIDENCE_BUNDLE_SPEC.md`
- `artifacts/inference_extrapolation_validator/API_COMPATIBILITY_POLICY.md`
- `artifacts/inference_extrapolation_validator/docs/OPERATIONS_RUNBOOK.md`
- `artifacts/inference_extrapolation_validator/docs/BENCHMARK_PLAN.md`
- `artifacts/inference_extrapolation_validator/docs/LEGAL_LICENSE_BOUNDARY.md`
- `artifacts/inference_extrapolation_validator/docs/INDEPENDENT_VERIFICATION_CHECKLIST.md`
- `.github/workflows/iev-enterprise-gate.yml`

### Change
- `generate_artifact.py` (emit release evidence pack hooks, compatibility version header).
- `test_generate_artifact.py` (compatibility tests + replay determinism tests).
- `falsifier.py` (enterprise failure-mode battery incl. malformed release bundle).
- `README.md` (enterprise package section + support matrix).

## 3) Tests / falsifiers required
1. API contract backward-compat test for previous minor version artifact.
2. Deterministic replay test across Python 3.11/3.12 matrix.
3. Evidence-bundle completeness test (manifest + schemas + hashes).
4. CVE policy gate test (fail on critical vulnerabilities).
5. License-boundary compliance test (required docs exist in release package).
6. Falsifier case: stale witness timestamp rejected.
7. Falsifier case: release manifest checksum mismatch fails.
8. Falsifier case: API-version mismatch fails verify.

## 4) Commercial trust gaps (remaining)
- No third-party signed assurance letter.
- No audited customer case study with quantified prevented loss.
- No formal support SLA with response windows.
- No legal package for procurement review.
- No benchmark narrative with before/after operational KPIs.

## 5) Implementation plan
- **PR-1 (this cycle):** enterprise gap analysis + roadmap + first diff spec.
- **PR-2:** release manifest, API compatibility policy, legal boundary docs.
- **PR-3:** enterprise CI gate (lint/mypy/security scan/repro matrix).
- **PR-4:** audit evidence bundle generator + verifier tests.
- **PR-5:** benchmark harness + report template + buyer demo script.
- **PR-6:** independent verification checklist + dry-run evidence package.

## 6) First PR diff (proposed)
- Add `ENTERPRISE_GAP_ANALYSIS.md`.
- Add `ENTERPRISE_IMPLEMENTATION_PLAN.md`.
- Add `FIRST_PR_DIFF.md` with concrete file-level tasks and acceptance tests.
