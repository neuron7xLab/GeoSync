# Coverage Ascent Work Act 92-97

This PR records a measured coverage-improvement protocol for TD-023.

Baseline: release-line coverage 85.03 percent, below the 90 percent release gate.

Primary start surface: backtest, because it has the largest tracked gap: 68.55 / 98.

Next surfaces: risk, execution, core, ingestion, analytics.

No thresholds are lowered. No coverage omit rules are expanded. Runtime improvement must be proven by same-commit coverage artifacts.
