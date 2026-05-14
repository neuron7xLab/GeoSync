# D-002J-P1A — Source Registry Provenance Audit

**Schema:** `D002J-SOURCE-PROVENANCE-AUDIT-v1`
**Parent registry:** `artifacts/d002j/data_registry/source_registry_v1.json` (sha256 `0fae24d4c3ef3165509166bec89d6dc5eee806888f352358ad77851e51079b7b`)
**Audit JSON:** `artifacts/d002j/data_registry/source_provenance_audit_v1.json`
**Smoke JSON:** `artifacts/d002j/data_registry/source_access_smoke_v1.json`
**Evidence lock JSON:** `artifacts/d002j/data_registry/source_evidence_lock_v1.json`
**Summary JSON:** `artifacts/d002j/data_registry/source_registry_audit_summary_v1.json`
**Companion docs:** [`D002J_SOURCE_DOWNGRADE_LOG.md`](D002J_SOURCE_DOWNGRADE_LOG.md), [`D002J_P1A_AUDIT_BOUNDARY.md`](D002J_P1A_AUDIT_BOUNDARY.md)

P1A is the **first verification gate** of the D-002J Frontier Benchmark Program. P1A does not add sources. P1A does not trust well-formatted metadata. P1A AUDITS the 25 sources committed in PR #695 and DOWNGRADES anything that fails one of ten verification dimensions: provider, official URL, documentation URL, access method, license boundary, coverage window, frequency, variables, crisis-window relevance, mechanistic relevance.

## Summary table

| Source ID | Class | Registry status (before) | Audit status |
|-----------|-------|--------------------------|--------------|
| BIS_CBS | banking | USABLE_NOW | VERIFIED |
| FDIC_CALL_REPORTS | banking | USABLE_NOW | VERIFIED |
| ECB_CBD | banking | USABLE_NOW | DOWNGRADED |
| FED_Y9C | banking | USABLE_NOW | PARTIAL |
| NYFED_SOFR | repo | USABLE_NOW | VERIFIED |
| OFR_REPO_DATA | repo | USABLE_NOW | VERIFIED |
| FED_H15 | repo | USABLE_NOW | VERIFIED |
| ECB_MMSR | repo | CANDIDATE_REQUIRES_LICENSE_REVIEW | PARTIAL |
| FRED | macro_financial | USABLE_NOW | VERIFIED |
| ALFRED | macro_financial | USABLE_NOW | PARTIAL |
| OFR_FSI | macro_financial | USABLE_NOW | VERIFIED |
| STLFSI | macro_financial | USABLE_NOW | VERIFIED |
| KCFSI | macro_financial | USABLE_NOW | PARTIAL |
| CBOE_VIX | market_structure | USABLE_NOW | PARTIAL |
| ICAP_MOVE | market_structure | CANDIDATE_REQUIRES_LICENSE_REVIEW | DOWNGRADED |
| BIS_QR_NETWORK | market_structure | USABLE_NOW | DOWNGRADED |
| OFR_WP_NETWORK | market_structure | USABLE_NOW | VERIFIED |
| NBER_RECESSION | crisis_window | USABLE_NOW | VERIFIED |
| FED_TIMELINE | crisis_window | USABLE_NOW | DOWNGRADED |
| ECB_FSR | crisis_window | USABLE_NOW | VERIFIED |
| BOE_LDI_REVIEW | crisis_window | USABLE_NOW | DOWNGRADED |
| FDIC_SVB_POSTMORTEM | crisis_window | USABLE_NOW | PARTIAL |
| LIT_INTERBANK_CONTAGION | literature_support | USABLE_NOW | VERIFIED |
| LIT_NETWORK_RECON | literature_support | USABLE_NOW | VERIFIED |
| LIT_REPO_FUNDING | literature_support | USABLE_NOW | PARTIAL |

**Aggregate:** 13 VERIFIED, 7 PARTIAL, 5 DOWNGRADED, 0 REJECTED.
**verified_or_partial = 20** (floor 18 — PASS).
**verified_usable_now = 13** (floor 12 — PASS).
**Crisis-window retention (≥3 verified/partial each):** CW1 17, CW2 12, CW3 13, CW4 14, CW5 8, CW6 12 — PASS.
**Mechanism-family retention (≥2 verified/partial each):**

| Mechanism family | Verified+partial |
|------------------|------------------|
| contagion | 4 |
| liquidity_funding | 7 |
| balance_sheet | 3 |
| market_wide_stress | 6 |
| official_response | 5 |
| information_constraint | **1** |

`information_constraint` carries only one source (ALFRED) and ALFRED's audit_status is PARTIAL (documentation surface partially unreachable). The ≥2 floor is **NOT** satisfied for this mechanism family. Per master doc §10, the audit decision is therefore **`SOURCE_REGISTRY_REJECTED`** — a scientifically valid outcome that requires P1 repair before P2 can open.

## By source class

### Banking (4 sources)

| Source | Status | Key evidence | Reason |
|--------|--------|--------------|--------|
| BIS_CBS | VERIFIED | bis.org/statistics/consstats.htm HEAD 200; consbankstatsguide.htm 200 | provider + URLs + 1983Q4 origin + quarterly + cross-border claims variables all live |
| FDIC_CALL_REPORTS | VERIFIED | cdr.ffiec.gov/public/ 200; fdic.gov/resources/bankers/call-reports/ 200 | FFIEC CDR portal + 1976Q1 origin + schedule RC/RC-E/RC-N variables all live |
| ECB_CBD | DOWNGRADED | data.ecb.europa.eu/ live; sdw.ecb.europa.eu NXDOMAIN | official_url points to decommissioned SDW; ECB migrated to data.ecb.europa.eu; access_method `public_sdw_api` is stale |
| FED_Y9C | PARTIAL | federalreserve.gov/apps/mdrm/data-dictionary 200; ffiec.gov/nicpubweb/nicweb/Y9CMain.aspx 404 | MDRM dictionary live (substantive anchor); FFIEC NIC Y-9C page moved to /npw/FinancialReport |

### Repo (4 sources)

| Source | Status | Key evidence | Reason |
|--------|--------|--------------|--------|
| NYFED_SOFR | VERIFIED | newyorkfed.org/markets/reference-rates/sofr 200; .../sofr-method 200 | full pin chain live; 2018-04-02 origin; percentile bands + volume in docs |
| OFR_REPO_DATA | VERIFIED | financialresearch.gov/short-term-funding-monitor/ 200; .../data/ 200 | dashboard + CSV export live since 2014-01-01 |
| FED_H15 | VERIFIED | federalreserve.gov/releases/h15/ 200; .../about.htm 200 | 1962-01-02 origin; daily; FRED mirror DGS10/DFF/DPRIME |
| ECB_MMSR | PARTIAL | ecb.europa.eu/stats/financial_markets_and_interest_rates/money_market/html/index.en.html 200; mmsr/html/index.en.html 404 | aggregate ESTR + volumes still public; mmsr nav URL moved under data.ecb portal |

### Macro-financial (5 sources)

| Source | Status | Key evidence | Reason |
|--------|--------|--------------|--------|
| FRED | VERIFIED | fred.stlouisfed.org/ 200; .../docs/api/fred/ 200 | API docs live; all listed series verifiable; TEDRATE cessation already in known_limitations |
| ALFRED | PARTIAL | alfred.stlouisfed.org/ 200; docs/api/alfred/ unreachable HEAD | landing live; documentation surface partially unreachable from sandbox; substantive vintage product valid |
| OFR_FSI | VERIFIED | financialresearch.gov/financial-stress-index/ 200; .../indicators/index.html 200 | 2000-01-04 origin; 5 regions; subcomponents match Monin 2017 |
| STLFSI | VERIFIED | fred.stlouisfed.org/series/STLFSI4 200; research.stlouisfed.org publication 200 | Kliesen-Smith 2010 doc live; LIBOR-replacement break documented |
| KCFSI | PARTIAL | fred.stlouisfed.org/series/KCFSI 200; kansascityfed.org/research/.../measuring-financial-stress/ 404 | series live on FRED; KC Fed paper landing moved |

### Market structure (4 sources)

| Source | Status | Key evidence | Reason |
|--------|--------|--------------|--------|
| CBOE_VIX | PARTIAL | fred.stlouisfed.org/series/VIXCLS 200; cboe.com/.../vix_historical_data/ 403 | FRED-mirrored series is the redistributable surface; CBOE site blocks bot |
| ICAP_MOVE | DOWNGRADED | theice.com/iba/move-index 404; theice.com/publicdocs/MOVE_Index_Methodology.pdf 404 | both URLs dead; product exists but registry pin chain broken |
| BIS_QR_NETWORK | DOWNGRADED | bis.org/publ/qtrpdf/ 404; bis.org/statistics/ 200 | BIS reorganised QR PDF path; substantive QR still alive but registry pin broken |
| OFR_WP_NETWORK | VERIFIED | financialresearch.gov/working-papers/ 200 | index live; methodology references not raw data correctly forbidden |

### Crisis window (5 sources)

| Source | Status | Key evidence | Reason |
|--------|--------|--------------|--------|
| NBER_RECESSION | VERIFIED | nber.org/research/business-cycle-dating 200 + FAQ 200 | BCD Committee page live; 1854 origin; USREC FRED mirror canonical |
| FED_TIMELINE | DOWNGRADED | federalreserve.gov/ 200; foia-financial-crisis-timeline.htm 404; monetarypolicy/financial-stability.htm 404 | FRB reorganised educational tools; both pins dead |
| ECB_FSR | VERIFIED | ecb.europa.eu/pub/financial-stability/fsr/html/index.en.html 200 | semi-annual since 2004; Nov 2022 covers UK LDI |
| BOE_LDI_REVIEW | DOWNGRADED | bankofengland.co.uk/ 200; .../financial-stability-report/2022 404; .../working-paper 404 | both pins dead; BoE moved FSR slug + WP path |
| FDIC_SVB_POSTMORTEM | PARTIAL | fdic.gov/news/news/press/2023/index.html 200; fdicoig.gov/publications/material-loss-reviews 404 | docs URL live; official_url MLR path moved; substantive OIG product canonical |

### Literature support (3 sources)

| Source | Status | Key evidence | Reason |
|--------|--------|--------------|--------|
| LIT_INTERBANK_CONTAGION | VERIFIED | jstor.org/ 200; nber.org/papers 200 | Eisenberg-Noe / Battiston / Glasserman-Young / Anand / Aldasoro-Alves / Brunnermeier all preprint-available |
| LIT_NETWORK_RECON | VERIFIED | bundesbank.de/.../discussion-papers 200; ssrn.com/ 200 | Upper-Worms / Mistrulli / Anand-Craig-von-Peter / Battiston 2016 all verifiable |
| LIT_REPO_FUNDING | PARTIAL | nber.org/papers 200; federalreserve.gov/econres/feds.htm 404 | NBER lit anchors live; FEDS index path reorganised |

## How audit_status is assigned

Per master doc §5/§6:

* **VERIFIED** — all 10 verification dimensions are "yes": provider, official_url, documentation, access_method, license_boundary, coverage, frequency, variables, crisis_window_relevance, mechanistic_relevance.
* **PARTIAL** — most dimensions yes, 1–2 partial, none fail; typical case is a stale documentation_url with the substantive product still reachable.
* **DOWNGRADED** — at least one dimension fails AND at least one other is fine; source stays in registry but flagged for restricted use (specifically: P2 MUST repair URL pins before any W1 ingestion).
* **REJECTED** — multiple dimensions fail OR provenance fundamentally broken. P1A surfaced **0 REJECTED** sources (no source had fundamentally broken provider attribution); however the registry overall is **REJECTED** at the mechanism-family floor (see Summary).

## Decision

`SOURCE_REGISTRY_REJECTED`. The audit affirms 20/25 sources hold (VERIFIED+PARTIAL ≥ floor 18) and crisis-window floors hold (each CW retains ≥3 verified/partial), but the **information_constraint** mechanism family carries only one source (ALFRED) and ALFRED is PARTIAL not VERIFIED, so the ≥2 floor for that family is not satisfied. Per the audit contract, this triggers `SOURCE_REGISTRY_REJECTED` — the system working as designed.

The fix is **not** to add new sources in P1A (forbidden — P1A is audit-only) and **not** to soften the floor by re-coarsening families. The fix is a follow-up `fix(x10r,D-002J-P1)` PR that either:

1. Adds at least one more `information_constraint` source (e.g. Philadelphia Fed Real-Time Data Set; Atlanta Fed GDPNow vintages) to the P1 registry, OR
2. Documents `information_constraint` as a single-source SUPPORT family with explicit floor relaxation, justified in writing and re-pinned in the prereg's success_criteria.

Either path requires a new PR. P2 cannot open until the audit decision flips to `SOURCE_REGISTRY_VERIFIED` or `SOURCE_REGISTRY_PARTIALLY_VERIFIED`.

## Locked invariants (byte-exact)

* `docs/governance/D002C_CLAIM_LEDGER.yaml` — `eb0b7151d76e5409e6dc9bb4a023551de5e0704673d5ac9f726319ef84a32387`
* `docs/governance/D002G_PREREGISTRATION.yaml` — `1ab91f09370e4705a8b0849467bc1f56df2e58d58d5623d3b6d905cbd110bb04`
* `docs/governance/D002G_ACCEPTANCE_RULES.md` — `875b1e3eb031b8e5333dc8b455454f0a30419ead1ebe787aa01d5882e7d6ad31`
* `docs/governance/D002H_PREREGISTRATION.yaml` — `44b18b5a40ce9d188a9c3bd49339621f81a65a15f97a683247902450dd54acec`
* `docs/governance/D002I_PREREGISTRATION.yaml` — `b646989c032dc0e29f9b791e0b68209ff22b40f4757737712badc8656cf2db5f`
* `docs/governance/D002J_PREREGISTRATION.yaml` — `f3dc65b7e64b96eafe6f23ca8bdd0e05dc9bf95b12c2658b227bd0340f7975a0`
* `artifacts/d002j/data_registry/source_registry_v1.json` — `0fae24d4c3ef3165509166bec89d6dc5eee806888f352358ad77851e51079b7b`

P1A modifies none of the above. Tests `test_d002j_prereg_not_modified` / `test_d002c_ledger_not_modified` / `test_systemic_risk_source_code_not_modified` enforce this byte-exactly.

## Lineage

`D-002G → D-002H REFUSED → D-002I → D-002J #694 (prereg) → P1 #695 (registry) → P1A (this PR; REJECTED at mechanism-family floor)`

Next legal PR: `fix(x10r,D-002J-P1): repair source registry provenance` — either add an `information_constraint` source or relax the mechanism-family floor with explicit justification.
