# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for the mutation-kill ratchet."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_MOD_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "check_mutation_kill_ratchet.py"
)
_spec = importlib.util.spec_from_file_location("check_mutation_kill_ratchet", _MOD_PATH)
assert _spec and _spec.loader
gate: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def _no_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: Any, **_k: Any) -> tuple[int, dict[str, Any], str, str]:
        raise AssertionError("ratchet probed despite no claim-critical change")

    monkeypatch.setattr(gate, "_probe_module", _boom)


def test_clean_tree_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_base_ledger", lambda ref: gate._load_ledger())
    monkeypatch.setattr(gate, "_changed_files", lambda ref: set())
    _no_probe(monkeypatch)
    assert gate.main(["--base-ref", "HEAD"]) == 0


def test_lowering_a_floor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = gate._load_ledger()
    mod = sorted(ledger)[0]
    higher = {m: dict(spec) for m, spec in ledger.items()}
    higher[mod]["floor"] = float(ledger[mod]["floor"]) + 0.5
    monkeypatch.setattr(gate, "_base_ledger", lambda ref: higher)
    monkeypatch.setattr(gate, "_changed_files", lambda ref: set())
    _no_probe(monkeypatch)
    assert gate.main(["--base-ref", "HEAD"]) == 1


def test_regression_below_floor_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = gate._load_ledger()
    mod = sorted(ledger)[0]
    monkeypatch.setattr(gate, "_base_ledger", lambda ref: ledger)
    monkeypatch.setattr(gate, "_changed_files", lambda ref: {mod})

    def _regressed(
        module: str, tests: str, floor: float
    ) -> tuple[int, dict[str, Any], str, str]:
        return (
            1,
            {
                "kill_rate": floor - 0.1,
                "killed": 0,
                "total": 1,
                "survivors": [{"lineno": 1, "kind": "boolop", "detail": "And->Or"}],
            },
            "",
            "",
        )

    monkeypatch.setattr(gate, "_probe_module", _regressed)
    assert gate.main(["--base-ref", "HEAD"]) == 1


def test_meeting_floor_on_change_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = gate._load_ledger()
    mod = sorted(ledger)[0]
    monkeypatch.setattr(gate, "_base_ledger", lambda ref: ledger)
    monkeypatch.setattr(gate, "_changed_files", lambda ref: {mod})

    def _ok(
        module: str, tests: str, floor: float
    ) -> tuple[int, dict[str, Any], str, str]:
        return (
            0,
            {"kill_rate": floor, "killed": 1, "total": 1, "survivors": []},
            "",
            "",
        )

    monkeypatch.setattr(gate, "_probe_module", _ok)
    assert gate.main(["--base-ref", "HEAD"]) == 0


def test_baseline_failure_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = gate._load_ledger()
    mod = sorted(ledger)[0]
    monkeypatch.setattr(gate, "_base_ledger", lambda ref: ledger)
    monkeypatch.setattr(gate, "_changed_files", lambda ref: {mod})
    monkeypatch.setattr(
        gate,
        "_probe_module",
        lambda *a: (2, {}, "", "baseline failed"),
    )
    assert gate.main(["--base-ref", "HEAD"]) == 2


def test_base_ledger_reads_committed_ledger() -> None:
    assert gate._base_ledger("HEAD") != {}
    assert gate._base_ledger("nonexistent-ref-zzzzzz") == {}


def test_broken_probe_report_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = gate._load_ledger()
    mod = sorted(ledger)[0]
    monkeypatch.setattr(gate, "_base_ledger", lambda ref: ledger)
    monkeypatch.setattr(gate, "_changed_files", lambda ref: {mod})
    monkeypatch.setattr(gate, "_probe_module", lambda *a: (1, {}, "", ""))
    assert gate.main(["--base-ref", "HEAD"]) == 2
