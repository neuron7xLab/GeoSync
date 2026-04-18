"""Tests for the taker-fee stress artifact."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

_ARTIFACT = Path("results/L2_FEE_STRESS.json")


@pytest.fixture(scope="module")
def stress() -> dict[str, Any]:
    if not _ARTIFACT.exists():
        pytest.skip("fee stress artifact not present")
    with _ARTIFACT.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def test_canonical_4bp_cell_matches_gate(stress: dict[str, Any]) -> None:
    cells = [c for c in stress["cells"] if float(c["taker_fee_bp"]) == 4.0]
    assert len(cells) == 1
    assert cells[0]["status"] == "BRACKET"
    assert abs(float(cells[0]["breakeven_maker_fraction"]) - 0.23167) < 1e-3


def test_rtc_rises_monotonically_with_fee(stress: dict[str, Any]) -> None:
    cells = sorted(stress["cells"], key=lambda c: float(c["taker_fee_bp"]))
    rtcs = [float(c["total_rtc_at_f0_bp"]) for c in cells]
    assert all(b > a for a, b in pairwise(rtcs))


def test_breakeven_rises_monotonically_with_fee(stress: dict[str, Any]) -> None:
    bracketed = [c for c in stress["cells"] if c["status"] == "BRACKET"]
    bracketed.sort(key=lambda c: float(c["taker_fee_bp"]))
    fs = [float(c["breakeven_maker_fraction"]) for c in bracketed]
    assert all(b > a for a, b in pairwise(fs))


def test_verdict_is_canonical(stress: dict[str, Any]) -> None:
    assert stress["verdict"] in {"RESILIENT", "BOUND", "FRAGILE"}


def test_every_cell_bracket_below_0p50(stress: dict[str, Any]) -> None:
    """All covered fee tiers must bracket below the 0.50 robust ceiling."""
    for cell in stress["cells"]:
        if cell["status"] == "BRACKET":
            assert float(cell["breakeven_maker_fraction"]) < 0.50


def test_max_viable_fee_at_least_canonical(stress: dict[str, Any]) -> None:
    assert float(stress["max_viable_taker_fee_bp"]) >= 4.0
