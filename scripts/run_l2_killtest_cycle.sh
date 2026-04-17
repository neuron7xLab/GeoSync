#!/usr/bin/env bash
# Orchestrator: run collector for N hours, then immediately fire the kill test.
# Streams both logs; writes final L2_KILLTEST_VERDICT.json into results/.
set -u

DURATION_SEC="${DURATION_SEC:-21600}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PY="$REPO_ROOT/.venv/bin/python"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR" "$REPO_ROOT/results"

COLLECTOR_LOG="$LOG_DIR/collector.log"
KILLTEST_LOG="$LOG_DIR/killtest.log"
STATUS_FILE="$LOG_DIR/killtest_cycle.status"

printf 'phase=collector start_utc=%s duration_sec=%s\n' \
  "$(date -u +%FT%TZ)" "$DURATION_SEC" > "$STATUS_FILE"

PYTHONPATH="$REPO_ROOT" "$PY" scripts/collect_binance_perp_l2.py \
  --duration-sec "$DURATION_SEC" \
  >> "$COLLECTOR_LOG" 2>&1
COLLECTOR_RC=$?

printf 'phase=killtest collector_rc=%s end_collector_utc=%s\n' \
  "$COLLECTOR_RC" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"

PYTHONPATH="$REPO_ROOT" "$PY" scripts/run_l2_killtest.py \
  >> "$KILLTEST_LOG" 2>&1
KILLTEST_RC=$?

printf 'phase=done killtest_rc=%s end_utc=%s verdict_json=results/L2_KILLTEST_VERDICT.json\n' \
  "$KILLTEST_RC" "$(date -u +%FT%TZ)" >> "$STATUS_FILE"

exit "$KILLTEST_RC"
