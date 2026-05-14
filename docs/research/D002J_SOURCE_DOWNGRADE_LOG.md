# D-002J-P1A — Source Downgrade Log

Records every source that audit P1A demoted from its registry status. Entries only for DOWNGRADED and REJECTED. PARTIAL entries are tracked in `D002J_SOURCE_PROVENANCE_AUDIT.md` (companion).

| Source ID | Registry status (before) | Audit status (after) | Downgrade reason | Recommended next action |
|-----------|--------------------------|----------------------|------------------|--------------------------|
| ECB_CBD | USABLE_NOW | DOWNGRADED | official_url `https://sdw.ecb.europa.eu/browse.do?node=9691593` returns DNS NXDOMAIN (SDW decommissioned); documentation_url HTTP 404; ECB migrated to `data.ecb.europa.eu` portal; access_method `public_sdw_api` is stale | P2 must re-pin official_url + documentation_url to `https://data.ecb.europa.eu/data-collection/cbd` and update access_method to reflect the new Data Portal API |
| ICAP_MOVE | CANDIDATE_REQUIRES_LICENSE_REVIEW | DOWNGRADED | both URLs return HTTP 404; `theice.com/iba/move-index` and `theice.com/publicdocs/MOVE_Index_Methodology.pdf` are dead; ICE BofA reorganised methodology pages | P2 to either re-pin official_url to current ICE MOVE methodology page (verify before pinning) or accept DOWNGRADED status permanently and use only the FRED public partial-history mirror |
| BIS_QR_NETWORK | USABLE_NOW | DOWNGRADED | official_url `https://www.bis.org/publ/qtrpdf/` returns HTTP 404; BIS reorganised QR PDF tree; documentation_url `bis.org/statistics/` still 200 | P2 to re-pin official_url to `https://www.bis.org/quarterlyreviews.htm` or canonical issue-listing index; verify before pinning |
| FED_TIMELINE | USABLE_NOW | DOWNGRADED | both URLs return HTTP 404; `federalreserve.gov/aboutthefed/educational-tools/foia-financial-crisis-timeline.htm` and `monetarypolicy/financial-stability.htm` are dead; FRB reorganised educational tools tree | P2 to re-pin to current FRB financial-stability report landing `https://www.federalreserve.gov/publications/financial-stability-report.htm` plus FOMC archive paths |
| BOE_LDI_REVIEW | USABLE_NOW | DOWNGRADED | both URLs return HTTP 404; `bankofengland.co.uk/financial-stability-report/2022` and `/working-paper` are dead; BoE moved FSR slug + WP path | P2 to re-pin to current BoE Financial Stability Report path and `bankofengland.co.uk/working-paper/staff-working-papers`; pin BoE WP 1019 (Czech-Huang-Silva-Vause 2022) directly |

## Aggregate

5 sources DOWNGRADED. 0 sources REJECTED. Note however: the registry as a whole receives `SOURCE_REGISTRY_REJECTED` because the `information_constraint` mechanism-family floor (≥2 verified/partial) is not met (ALFRED is the only source carrying that mechanism and its audit_status is PARTIAL).

## Repair scope (for the follow-up fix PR)

A `fix(x10r,D-002J-P1)` PR must:

1. Re-pin the 5 downgraded sources' URL fields to live canonical pages, verified by HEAD probe at the time of writing.
2. Either add ≥1 new `information_constraint` source (e.g. Philadelphia Fed Real-Time Data Set, Atlanta Fed GDPNow vintages, BIS Long Series real-time data) or amend the prereg's mechanism-family floor with explicit, dated, signed justification.
3. Re-run the P1A audit script (or its successor) to confirm the decision flips to VERIFIED or PARTIALLY_VERIFIED.
4. The fix PR is registry repair, not new science — locked governance shas must remain byte-exact.

## What this log does NOT do

- Does NOT ingest any data
- Does NOT claim real-bank validation
- Does NOT reopen D-002H (REFUSED remains canonical)
- Does NOT pre-empt D-002I investigation
- Does NOT authorise a canonical run
- Does NOT modify the P1 registry JSON file (audit-only)
