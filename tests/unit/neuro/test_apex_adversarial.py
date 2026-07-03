# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""N4 — adversarial soundness of the No-Ungrounded-Act apex.

An adversary tries to make FORBIDDEN look ADMISSIBLE by forging a committed
artifact. The apex resists because it does not trust verdict fields: it re-checks
the numeric homeostasis grounds (H1/H3), rejects report-only runs, and RE-DERIVES
H4 by re-running the aggregator over the substrate — so a forged final verdict
that hides a FAIL/missing substrate is caught.

Honest boundary: substrate DATA artifacts (coverage, etc.) are trusted-as-data
here, protected by the manifest-hash / commit-acceptor gates, not re-derived.
Each test seeds a tmp root from the real artifacts, tampers ONE thing, and asserts
the apex fails closed.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOL = ROOT / "tools" / "audit_no_ungrounded_act.py"
_HOMEO = "artifacts/neuro/homeostasis_contract.json"
_FINAL = "artifacts/inference/final_inference_verdict.json"
_CONCURRENCY = "artifacts/concurrency/concurrency_matrix.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("nua_adv", TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nua = _load_tool()


@pytest.fixture()
def honest_root(tmp_path: Path) -> Path:
    # Seed a tmp root from the real committed artifacts -> honest ADMISSIBLE.
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    return tmp_path


def _patch(root: Path, rel: str, **changes: object) -> None:
    path = root / rel
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_honest_root_is_admissible(honest_root: Path) -> None:
    assert nua.build_apex(honest_root)["gate"] == "ADMISSIBLE"


def test_forged_h1_numeric_is_caught(honest_root: Path) -> None:
    # verdict left PASS, but the allostatic load is forged out of bound.
    _patch(honest_root, _HOMEO, worst_allostatic_load=99.0)
    report = nua.build_apex(honest_root)
    assert report["gate"] == "FORBIDDEN"
    assert next(g for g in report["grounds"] if g["id"] == "H1")["holds"] is False


def test_forged_h3_coherence_is_caught(honest_root: Path) -> None:
    _patch(honest_root, _HOMEO, coherence_locked=0.0)
    report = nua.build_apex(honest_root)
    assert report["gate"] == "FORBIDDEN"
    assert next(g for g in report["grounds"] if g["id"] == "H3")["holds"] is False


def test_report_only_homeostasis_cannot_ground(honest_root: Path) -> None:
    _patch(honest_root, _HOMEO, report_only=True)
    report = nua.build_apex(honest_root)
    assert report["gate"] == "FORBIDDEN"


def test_forged_aggregate_is_defeated_by_rederivation(honest_root: Path) -> None:
    # The adversary forges the final verdict to PASS but a substrate is FAIL.
    _patch(honest_root, _FINAL, verdict="PASS")
    _patch(honest_root, _CONCURRENCY, verdict="FAIL")
    report = nua.build_apex(honest_root)
    # H4 is re-derived from the substrate, so the hidden FAIL surfaces.
    assert report["gate"] == "FORBIDDEN"
    assert next(g for g in report["grounds"] if g["id"] == "H4")["holds"] is False


def test_missing_substrate_is_defeated_by_rederivation(honest_root: Path) -> None:
    (honest_root / _CONCURRENCY).unlink()
    report = nua.build_apex(honest_root)
    assert report["gate"] == "FORBIDDEN"
    assert next(g for g in report["grounds"] if g["id"] == "H4")["holds"] is False


def test_wrong_schema_id_substrate_is_defeated(honest_root: Path) -> None:
    # A renamed/unrelated file cannot masquerade as a substrate sub-verdict.
    _patch(honest_root, _CONCURRENCY, schema="geosync.some_other_thing.v1")
    report = nua.build_apex(honest_root)
    assert report["gate"] == "FORBIDDEN"
