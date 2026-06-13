#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Compatibility wrapper for the structured PR preflight runner.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT}/artifacts/pr_preflight"
REPORT_PATH="${REPORT_DIR}/preflight_report.json"
LEDGER_PATH="${REPORT_DIR}/inference_ledger.jsonl"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    mkdir -p "${REPORT_DIR}"
    echo "ERROR: python3 or python is required to run PR preflight" >&2
    echo "FIRST_FILE_TO_OPEN=${REPORT_PATH}"
    exit 1
fi

"${PYTHON_BIN}" "${ROOT}/tools/ci/pr_preflight.py" --root "${ROOT}" --report-dir "${REPORT_DIR}" "$@"
exit_code=$?

if [[ -f "${REPORT_PATH}" ]]; then
    "${PYTHON_BIN}" "${ROOT}/tools/ci/preflight_ledger.py" --report "${REPORT_PATH}" --ledger "${LEDGER_PATH}"
    ledger_exit=$?
    if [[ "${exit_code}" -eq 0 && "${ledger_exit}" -ne 0 ]]; then
        exit_code="${ledger_exit}"
    fi
elif [[ "${exit_code}" -eq 0 ]]; then
    echo "ERROR: PR preflight completed without ${REPORT_PATH}" >&2
    exit_code=1
fi

echo "FIRST_FILE_TO_OPEN=${REPORT_PATH}"
exit "${exit_code}"
