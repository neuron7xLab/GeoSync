from __future__ import annotations

import json
from pathlib import Path

from scripts.riee.package_scorecard import build_scorecard, main


def test_scorecard_generation() -> None:
    score = build_scorecard()
    assert score.automation_coverage_pct >= 90
    assert score.urgency_level == 'RED'
    assert main() == 0
    payload = json.loads(Path('artifacts/riee_offering_scorecard.json').read_text(encoding='utf-8'))
    assert payload['package_tier'] == 'RIEE_ENTERPRISE'
