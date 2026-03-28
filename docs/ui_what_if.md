# UX Research: What-If Exploration Mode

The What-If exploration mode enables analysts to manipulate strategy parameters
in real time and immediately visualise the effect on portfolio metrics. This
mode is intended for research environments and feature-flagged for production.

## Goals

1. **Rapid hypothesis testing** – Allow analysts to tweak weights, thresholds,
   and indicator settings without triggering live trades.
2. **Instant feedback** – Display projected PnL, drawdown, and risk deltas within
   200 ms of a parameter change.
3. **Traceability** – Persist experiment snapshots so adjustments can be shared
   or promoted into backtest configurations.

## Interaction Model

- **Parameter panel** – Sliders and numeric inputs grouped by indicator (Kuramoto,
  Ricci, Hurst, entropy). Parameters support keyboard entry and provide tooltip
  explanations with default ranges.
- **Scenario timeline** – A scrubber overlays historical price series with
  recomputed signals. Dragging the scrubber replays historical metrics using the
  adjusted parameters.
- **Metric cards** – Cards update live to show delta vs. baseline for Sharpe,
  Sortino, win rate, and risk utilisation. Cards are colour-coded (green within
  tolerance, amber borderline, red exceeding limits).
- **Snapshot queue** – Analysts can pin configurations, compare them side-by-side,
  and export to JSON for collaboration or ingestion into the backtest pipeline.

## Data Flow

1. UI publishes parameter mutations over a websocket channel with the scenario
   ID and user identity.
2. The scenario service clones the current feature graph, applies parameter
   overrides, and recalculates indicators on a rolling window using cached
   market data.
3. Calculated metrics are streamed back to the UI as incremental deltas.
4. Each mutation is logged to `observability/audit/what_if.jsonl` with user,
   timestamp, diff digest, and optional comment.

## Guardrails

- **Rate limiting** – Mutations capped at 10 per second per user to prevent UI
  storms.
- **Sandbox enforcement** – What-if mode interacts with a sandbox portfolio and
  cannot place orders. Feature flag `ui.what_if.enabled` controls availability.
- **Reproducibility** – Each snapshot includes indicator seeds, dataset hashes,
  and version metadata for later reproduction in notebooks or CI experiments.
- **Access control** – Available to roles `quant-research` and `product-analyst`
  via OIDC claims; other roles see a request-access prompt.

## Research Questions

- Do analysts converge on viable parameter sets faster with the mode enabled?
- Which metrics require additional visualisation (e.g., risk heatmaps, order book
  impact)?
- How often are snapshots exported and promoted to backtesting workflows?
- Does live recomputation stress backend resources? Monitor CPU/memory on the
  scenario service during research sessions.

## Instrumentation

- **Frontend analytics** – Track component interactions, slider adjustments, and
  snapshot usage.
- **Backend tracing** – Use OpenTelemetry spans tagged with scenario ID and user
  to monitor recompute latency.
- **Quality hooks** – Run the DST/session boundary tests after applying parameter
  overrides to ensure calendar integrity before promoting snapshots.

## Rollout Plan

1. **Alpha (internal)** – Limited to UX research team; gather usability feedback
   and refine baseline metrics.
2. **Beta (select users)** – Enable for a subset of power users with guardrails
   on maximum portfolio exposure adjustments.
3. **General availability** – Publish runbook updates, document workflows in
   `docs/ui_what_if.md`, and integrate approvals into the governance UI.

Regularly sync with UX research to evaluate task completion time, satisfaction
scores, and feature adoption metrics. Document findings in the quarterly UX
insights report.
