# Scenario Template

Use this template when documenting a new trading or data scenario. It complements the interactive **Scenario Studio** inside the web dashboard (`apps/web`) by capturing the context, guardrails, and validation notes in prose.

> 💡 **Tip:** Start with the Scenario Studio, adjust the sliders until the sanity checker reports no warnings, and paste the generated JSON snippet into the block below.

---

## Metadata

- **Name:** _Concise title (e.g. "Momentum Breakout Asia Session")_
- **Owner:** _Team or person responsible_
- **Last Reviewed:** _YYYY-MM-DD_
- **Data Window:** _e.g. 2021-01-01 → 2024-01-01_
- **Primary Markets:** _Tickers or exchanges_
- **Timeframe:** _Must match `<number><unit>` (e.g. 1h, 4h)_

## Strategy Posture

- **Hypothesis:** _What edge are we trying to capture?_
- **Entry Conditions:**
  - _Indicator thresholds, price structure, volume filters, etc._
- **Exit Conditions:**
  - _Stops, targets, time-based exits._
- **Max Concurrent Positions:** _Keep ≤ 10 to remain manageable._

## Risk Controls

- **Initial Balance:** _USD value used for backtests._
- **Risk Per Trade (%):** _Stay within 0.25–2 unless justified._
- **Expected Slippage & Fees:** _Document assumptions._
- **Drawdown Guardrails:** _Maximum tolerable drawdown and escalation path._

## Scenario JSON

```json
{
  "initialBalance": 10000,
  "riskPerTrade": 1,
  "maxPositions": 3,
  "timeframe": "1h"
}
```

Replace the defaults above with the output exported from the Scenario Studio once all validations pass.

## Validation Checklist

- [ ] **Data sanity** – Missing or malformed columns rejected by CLI sanity checks.
- [ ] **Risk review** – Scenario Studio warnings acknowledged or resolved.
- [ ] **Backtest repeatability** – Include deterministic seeds/configs if available.
- [ ] **Operational readiness** – Monitoring, alerting, and playbooks updated.

---

Document any deviations from the recommended guardrails and include justification links (Jira, Confluence, etc.).
