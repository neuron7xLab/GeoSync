# D-002J — Workstream 2: Crisis Window Registry

Pre-registration anchor: `docs/governance/D002J_PREREGISTRATION.yaml`
Plan reference: §7
Status: **SCAFFOLD** — exact day-level start/end dates are populated
by downstream PRs (D-002J-W2-*).
Date: 2026-05-14

> Each crisis window below is the external stress anchor against
> which D-002J substrates and null models are evaluated. The window
> id (CW1..CW6) is part of the D-002J pre-registration content
> address; ids are stable across the lineage and CANNOT be
> renumbered without a fresh D-002K pre-registration.

---

## §7 — Crisis-window acceptance contract

Each row MUST carry:

- **id** — `CW1`..`CW6` (locked at pre-registration).
- **name** — short verbal label.
- **start_date** / **end_date** — `YYYY-MM-DD` boundaries.
- **justification** — explicit reason this is a systemic-stress
  window (peer-reviewed survey, central-bank chronology, or
  regulator filing — never op-ed / journalism alone).
- **observable_indicators** — concrete observable instruments
  whose stress signatures evidence the window (e.g. TED spread
  spike, repo rate dislocation, gilt yield breakout).
- **related_data_sources** — W1 row ids that anchor the window.
- **forbidden_interpretations** — explicit denial claims for this
  window (mirroring the per-window claim boundary).

---

## CW1 — 2007-2009 GFC (Global Financial Crisis)

- **id**: `CW1`
- **start_date**: TBD (early 2007 — first ABS market dislocation)
- **end_date**: TBD (mid-2009 — post-Lehman recovery onset)
- **justification**: peer-reviewed BIS / IMF / FSOC chronologies.
- **observable_indicators**: TED spread, LIBOR-OIS, repo haircut
  spike on agency MBS, financial-sector CDS widening.
- **related_data_sources**: `BIS_BNK_STATS`, `FRED_ALFRED`,
  `OFR_REPO_STF`, `RATES_FX_LIQ_STRESS`.
- **forbidden_interpretations**:
  - "D-002J detected the GFC ex-ante" (NOT permitted — D-002J runs
    against the historical window, not a live precursor).
  - "GFC validates a D-002J substrate" (NOT permitted without
    positive-control survival and power-first canonical sign-off).

## CW2 — 2011-2012 Eurozone Sovereign Crisis

- **id**: `CW2`
- **start_date**: TBD (mid-2011 — Italy / Spain sovereign spread
  blowout).
- **end_date**: TBD (mid-2012 — Draghi "whatever it takes" stabilisation).
- **justification**: ECB / BIS Eurozone chronology.
- **observable_indicators**: 10Y sovereign yield spread vs Bund,
  intra-euro funding fragmentation, TARGET2 imbalances.
- **related_data_sources**: `ECB_MMSR_PUB`, `BIS_BNK_STATS`,
  `RATES_FX_LIQ_STRESS`.
- **forbidden_interpretations**:
  - "D-002J generalises a substrate across CW1 ∧ CW2 without
    independent positive controls" (NOT permitted).

## CW3 — 2019 US Repo Spike (September 2019)

- **id**: `CW3`
- **start_date**: TBD (mid-September 2019).
- **end_date**: TBD (early-November 2019 — Fed standing repo
  intervention).
- **justification**: FRBNY / OFR repo-market chronology.
- **observable_indicators**: SOFR / IORB spread, GC repo rate
  spike, Federal Reserve emergency repo operations.
- **related_data_sources**: `OFR_REPO_STF`, `FRED_ALFRED`,
  `RATES_FX_LIQ_STRESS`.
- **forbidden_interpretations**:
  - "the repo spike alone validates a repo-collateral-haircut
    substrate" (NOT permitted without positive-control survival).

## CW4 — 2020 COVID Dash-for-Cash (March 2020)

- **id**: `CW4`
- **start_date**: TBD (late-February 2020).
- **end_date**: TBD (mid-April 2020 — Fed dollar swap line / PMCCF
  stabilisation).
- **justification**: BIS / FSB / IMF March 2020 chronology.
- **observable_indicators**: USD funding stress, basis swap blowout,
  Treasury market dislocation, money-market fund redemptions.
- **related_data_sources**: `BIS_BNK_STATS`, `FRED_ALFRED`,
  `OFR_REPO_STF`, `RATES_FX_LIQ_STRESS`.
- **forbidden_interpretations**:
  - "the March 2020 episode constitutes ground truth for a
    deleveraging substrate" (NOT permitted).

## CW5 — 2022 UK Gilt / LDI Crisis (September-October 2022)

- **id**: `CW5`
- **start_date**: TBD (late-September 2022).
- **end_date**: TBD (mid-October 2022 — BoE temporary purchase
  operation conclusion).
- **justification**: Bank of England LDI chronology.
- **observable_indicators**: 30Y gilt yield breakout, LDI fund
  margin calls, sterling depreciation episode.
- **related_data_sources**: `RATES_FX_LIQ_STRESS`,
  `BANK_BS_PROXY_PUB`.
- **forbidden_interpretations**:
  - "the LDI episode validates a margin-call-cascade substrate
    in general" (NOT permitted without independent positive
    controls).

## CW6 — 2023 US Regional Banking Stress (March 2023)

- **id**: `CW6`
- **start_date**: TBD (early-March 2023 — SVB run).
- **end_date**: TBD (mid-May 2023 — First Republic resolution).
- **justification**: FDIC / Federal Reserve / FSOC chronology.
- **observable_indicators**: regional-bank deposit outflows,
  bank-sector CDS widening, KBW Regional Bank Index drawdown.
- **related_data_sources**: `BANK_BS_PROXY_PUB`, `FRED_ALFRED`,
  `BIS_BNK_STATS`.
- **forbidden_interpretations**:
  - "D-002J explains the SVB collapse" (NOT permitted — D-002J is
    benchmark, not causal explanation).

---

## Forbidden interpretations of this registry (global)

This document, in its present scaffold form, does **NOT** constitute:

- a claim that the listed windows have been ingested into any
  D-002J pipeline,
- a claim that the listed observable indicators have been validated,
- a claim that the listed windows are exhaustive (the list is
  the locked initial set; expansion is a downstream D-002J PR with
  positive-control discipline, not a freelance addition),
- a claim of real-bank validation (D-002J pre-registration
  explicitly forbids such claims; see `D002J_PREREGISTRATION.yaml`
  `forbidden_claims`).

---

## Lock anchors

This scaffold inherits the locked-governance anchors recorded in
`docs/governance/D002J_PREREGISTRATION.yaml` `locked_anchors`. Any
edit to the registry CW1..CW6 ids, window count, or per-window
forbidden-interpretation block constitutes a fresh D-002K
pre-registration, not a patch.
