from __future__ import annotations

from scripts.evaluate_dopamine_rpe_extension import build_rows, validate_rows


def test_evaluation_rows_vary() -> None:
    rows = build_rows(5, 20260629)
    values = [row["signed_score"] for row in rows]
    assert validate_rows(rows, 5) == []
    assert min(values) < 0.0
    assert max(values) > 0.0
