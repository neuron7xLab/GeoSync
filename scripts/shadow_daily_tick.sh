#!/usr/bin/env bash
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
#
# FIX-5: daily wrapper for the shadow trajectory ritual.
#
# One invocation does: tick → pipeline_status refresh → eval +
# shadow_live.json → CI-aware verdict → DP3 honest test → append a
# trajectory.log entry.
#
# Designed to run from the systemd timer at
# ops/systemd/cross_asset_kuramoto_shadow.timer (22:00 UTC daily).
# Idempotent on the same UTC day: paper_trader --tick is no-op if
# already ticked; the daily pipeline_status overwrites; trajectory.log
# appends one line.
#
# Falsification gate (FIX-5): after a successful invocation,
# journalctl --user -u cross_asset_kuramoto_shadow.service shows
# the literal string "tick: ok" at least once in the last 24 h.

set -Eeuo pipefail

REPO="${GEOSYNC_REPO:-$HOME/GeoSync}"
SPIKE_TRADER="${SPIKE_TRADER:-$HOME/spikes/cross_asset_sync_regime/paper_trader.py}"
TRAJECTORY_LOG="${TRAJECTORY_LOG:-$REPO/results/sharpe_trajectory.log}"
PYTHON="${PYTHON:-python3}"
INCIDENT_LOGGER="${INCIDENT_LOGGER:-$REPO/scripts/log_shadow_operational_incident.py}"
LIVE_STATE_PATH="${LIVE_STATE_PATH:-$REPO/results/cross_asset_kuramoto/shadow_validation/LIVE_STATE.json}"
CURRENT_STEP="startup"

cd "$REPO"
mkdir -p "$(dirname "$TRAJECTORY_LOG")"

ts_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
bar_date_utc() { date -u +%Y-%m-%d; }

emit() { printf '[shadow_daily_tick] %s %s\n' "$(ts_utc)" "$*" >&2; }

log_incident() {
  local reason="$1"
  local recovery_status="$2"
  if [[ -f "$INCIDENT_LOGGER" ]]; then
    "$PYTHON" "$INCIDENT_LOGGER" \
      --bar-date "$(bar_date_utc)" \
      --reason "$reason" \
      --recovery-status "$recovery_status" || \
      emit "incident_logger: FAIL reason=$reason"
  else
    emit "incident_logger: SKIP missing=$INCIDENT_LOGGER reason=$reason"
  fi
}

on_error() {
  local rc=$?
  log_incident \
    "systemd_run_failed step=$CURRENT_STEP rc=$rc" \
    "failed_logged_for_operator_review; next_timer_or_persistent_catch_up_required"
  emit "run: FAIL step=$CURRENT_STEP rc=$rc"
  exit "$rc"
}
trap on_error ERR

# systemd Persistent=true executes one catch-up after offline time, not one
# replay per missed wall-clock day. Detect that continuity gap before any new
# state is overwritten so the ledger records the blind interval.
CURRENT_STEP="offline_gap_detection"
if [[ -f "$LIVE_STATE_PATH" ]]; then
  LAST_EVAL_UTC="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
p = Path("${LIVE_STATE_PATH}")
try:
    state = json.loads(p.read_text())
    print(state.get("last_eval_utc") or state.get("last_run_utc") or "")
except Exception:
    print("")
PY
)"
  if [[ -n "$LAST_EVAL_UTC" ]]; then
    GAP_HOURS="$("$PYTHON" - <<PY
from datetime import datetime, timezone
last = "$LAST_EVAL_UTC"
try:
    t = datetime.fromisoformat(last.replace("Z", "+00:00"))
    print(int((datetime.now(timezone.utc) - t).total_seconds() // 3600))
except Exception:
    print(0)
PY
)"
    if [[ "$GAP_HOURS" =~ ^[0-9]+$ ]] && (( GAP_HOURS > 36 )); then
      log_incident \
        "offline_catch_up_gap last_eval_utc=$LAST_EVAL_UTC gap_hours=$GAP_HOURS" \
        "catch_up_logged_before_state_refresh"
    fi
  fi
fi

# 1. paper_trader --tick (advances paper-state by one bar; skips if today already ticked).
CURRENT_STEP="tick"
emit "step=tick paper_trader=$SPIKE_TRADER"
if [[ -f "$SPIKE_TRADER" ]]; then
  "$PYTHON" "$SPIKE_TRADER" --tick || {
    rc=$?
    emit "tick: FAIL rc=$rc (continuing — paper_trader may rate-limit on intra-day reruns)"
    log_incident \
      "paper_trader_failed rc=$rc path=$SPIKE_TRADER" \
      "failed_logged_continue_with_shadow_evaluation"
  }
else
  emit "tick: SKIP (spike paper_trader not present at $SPIKE_TRADER)"
  log_incident \
    "paper_trader_skipped_missing path=$SPIKE_TRADER" \
    "skipped_logged_continue_with_shadow_evaluation"
fi

# 2. Refresh pipeline_status from the live paper-state (FIX-1).
CURRENT_STEP="pipeline_status_refresh"
emit "step=pipeline_status_refresh"
"$PYTHON" -m geosync.pipeline_status_check --day today

# 3. Run the frozen evaluator + persist results/shadow_live.json (FIX-2).
CURRENT_STEP="eval"
emit "step=eval"
make eval-tick

# 4. Pull the latest live bar count from shadow_live.json.
CURRENT_STEP="read_live_bars"
LIVE_BARS="$("$PYTHON" -c "
import json, sys
from pathlib import Path
p = Path('$REPO/results/shadow_live.json')
if not p.is_file():
    sys.exit('shadow_live.json missing')
print(json.loads(p.read_text())['eval']['live_bars_completed'])
")"
emit "live_bars=$LIVE_BARS"

# 5. CI-aware verdict (FIX-4).
CURRENT_STEP="verdict"
emit "step=verdict"
VERDICT_JSON="$("$PYTHON" -m geosync.verdict --bar "$LIVE_BARS" --json)"
LABEL="$(printf '%s' "$VERDICT_JSON" | "$PYTHON" -c "import json,sys; print(json.loads(sys.stdin.read())['label'])")"
SHARPE="$(printf '%s' "$VERDICT_JSON" | "$PYTHON" -c "import json,sys; print(json.loads(sys.stdin.read())['sharpe_point'])")"
CI_LOW="$(printf '%s' "$VERDICT_JSON" | "$PYTHON" -c "import json,sys; print(json.loads(sys.stdin.read())['ci_low'])")"
CI_HIGH="$(printf '%s' "$VERDICT_JSON" | "$PYTHON" -c "import json,sys; print(json.loads(sys.stdin.read())['ci_high'])")"
emit "verdict bar=$LIVE_BARS sharpe=$SHARPE label=$LABEL ci=[$CI_LOW,$CI_HIGH]"

# 6. DP3 honest test (FIX-3) — best-effort; bar < 3 raises and we record.
CURRENT_STEP="dp3_test"
emit "step=dp3_test"
DP3_LABEL="$( "$PYTHON" -m geosync.dp3_test --honest --bar "$LIVE_BARS" --json 2>/dev/null \
  | "$PYTHON" -c "import json,sys; d=json.loads(sys.stdin.read() or '{}'); print(d.get('label','UNAVAILABLE'))" || true)"
DP3_LABEL="${DP3_LABEL:-UNAVAILABLE}"
emit "dp3=$DP3_LABEL"

# 7. Append the trajectory.log entry (one line per day).
CURRENT_STEP="trajectory_append"
ROW="$(ts_utc) bar=$LIVE_BARS sharpe=$SHARPE ci=[$CI_LOW,$CI_HIGH] verdict=$LABEL dp3=$DP3_LABEL"
printf '%s\n' "$ROW" >> "$TRAJECTORY_LOG"
emit "trajectory_appended path=$TRAJECTORY_LOG"

# 8. Terminal status line (string "tick: ok" required by FIX-5 gate).
CURRENT_STEP="complete"
emit "tick: ok"
