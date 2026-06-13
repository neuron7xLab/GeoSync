#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Compatibility wrapper for the structured PR preflight runner.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT}/artifacts/pr_preflight"
RUNNER="${ROOT}/tools/ci/pr_preflight.py"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "❌ Python is required to run PR preflight" >&2
    echo "FIRST_FILE_TO_OPEN=${REPORT_DIR}/preflight_report.json"
    exit 1
fi

"${PYTHON_BIN}" "${RUNNER}" --root "${ROOT}" --report-dir "${REPORT_DIR}" "$@"
exit_code=$?

echo "FIRST_FILE_TO_OPEN=${REPORT_DIR}/preflight_report.json"
exit "${exit_code}"
