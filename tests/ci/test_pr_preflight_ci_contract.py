# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""CI-alignment contract tests for the PR preflight runner."""

from __future__ import annotations

import json
from pathlib import Path

from tools.ci.pr_preflight import build_checks

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tools" / "ci" / "pr_preflight_ci_contract.json"


def test_ci_contract_represents_all_runner_checks(tmp_path: Path) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    represented = set(contract["checks"])
    checks = build_checks(ROOT, skip_install=False)
    check_ids = {check.id for check in checks}

    assert contract["schema_version"] == 1
    assert check_ids <= represented


def test_every_critical_check_has_ci_equivalent_or_local_reason() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    by_id = contract["checks"]

    for check in build_checks(ROOT, skip_install=False):
        entry = by_id[check.id]
        ci_equivalent = str(entry.get("ci_equivalent", "")).strip()
        local_only_reason = str(entry.get("local_only_reason", "")).strip()
        if check.critical:
            assert ci_equivalent or local_only_reason, check.id
        assert "production validated" not in ci_equivalent.lower()
