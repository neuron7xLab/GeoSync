#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Compatibility wrapper for the structured PR preflight runner.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${ROOT}/artifacts/pr_preflight"

if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    echo "ERROR: python3 or python is required to run PR preflight" >&2
    # No runner could execute, so no evidence report exists — do not point
    # callers at a non-existent artifact.
    exit 1
fi

"${PYTHON_BIN}" "${ROOT}/tools/ci/pr_preflight.py" --root "${ROOT}" --report-dir "${REPORT_DIR}" "$@"
exit_code=$?

# Only advertise the evidence artifact when it was actually written. On a
# BLOCKED run (e.g. the report directory could not be created) the report is
# absent, and pointing triage at a missing file is exactly wrong on the path
# where accurate triage matters most.
REPORT_FILE="${REPORT_DIR}/preflight_report.json"
if [ -f "${REPORT_FILE}" ]; then
    echo "FIRST_FILE_TO_OPEN=${REPORT_FILE}"
fi
exit "${exit_code}"
