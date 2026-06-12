# SPDX-License-Identifier: MIT
"""Contract tests for the architecture-debt inference.

Pin the fail-closed semantics: a measure that cannot be computed is
UNKNOWN (never silently OK), by-design classes never count as actionable
debt, and the inference is deterministic and machine-readable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "architecture_debt_inference.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("architecture_debt_inference", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_public_api() -> None:
    m = _load()
    assert callable(m.infer)
    assert callable(m.main)
    assert m.CLASSES


def test_infer_shape() -> None:
    m = _load()
    r = m.infer()
    assert set(r) == {"worst_severity", "debt_classes", "actionable"}
    assert r["worst_severity"] in {m.HIGH, m.MEDIUM, m.LOW, m.INFO, m.UNKNOWN}
    for f in r["debt_classes"]:
        assert {"id", "title", "severity", "count", "threshold", "status",
                "by_design", "note"} <= set(f)
        assert f["status"] in {"OK", "DEBT", m.UNKNOWN}


def test_every_class_has_explicit_threshold_and_severity() -> None:
    m = _load()
    for c in m.CLASSES:
        assert c.severity in {m.HIGH, m.MEDIUM, m.LOW, m.INFO}
        assert isinstance(c.threshold, int)
        assert c.note.strip()


def test_by_design_never_actionable() -> None:
    m = _load()
    r = m.infer()
    by_design_ids = {c.id for c in m.CLASSES if c.by_design}
    assert by_design_ids
    assert not (set(r["actionable"]) & by_design_ids)


def test_evidence_ledger_class_reports_ratio() -> None:
    m = _load()
    r = m.infer()
    f = next(x for x in r["debt_classes"] if x["id"] == "empty_evidence_ledger")
    assert "total_acceptors" in f and "ratio" in f
    assert f["count"] >= 0
    if f["total_acceptors"]:
        assert 0.0 <= f["ratio"] <= 1.0


def test_failing_measure_is_unknown_not_ok() -> None:
    m = _load()

    def boom() -> int:
        raise RuntimeError("cannot measure")

    broken = m.DebtClass("boom", "boom", m.HIGH, 0, boom, "x")
    m.CLASSES = (*m.CLASSES, broken)
    try:
        r = m.infer()
        f = next(x for x in r["debt_classes"] if x["id"] == "boom")
        assert f["status"] == m.UNKNOWN
        assert "boom" not in r["actionable"]  # UNKNOWN is not actionable DEBT
    finally:
        m.CLASSES = tuple(c for c in m.CLASSES if c.id != "boom")


def test_negative_count_is_unknown() -> None:
    m = _load()
    broken = m.DebtClass("neg", "neg", m.LOW, 0, lambda: -1, "x")
    m.CLASSES = (*m.CLASSES, broken)
    try:
        r = m.infer()
        f = next(x for x in r["debt_classes"] if x["id"] == "neg")
        assert f["status"] == m.UNKNOWN
    finally:
        m.CLASSES = tuple(c for c in m.CLASSES if c.id != "neg")


def test_deterministic() -> None:
    m = _load()
    assert m.infer() == m.infer()


def test_main_json_runs(capsys) -> None:
    import json

    m = _load()
    rc = m.main(["--json"])
    data = json.loads(capsys.readouterr().out)
    assert "debt_classes" in data
    assert rc in (0, 1)
