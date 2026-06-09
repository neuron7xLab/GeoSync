# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Release-gate harness — behavioural tests.

The gate's own promise is falsifiable: it must (1) return only valid
statuses, (2) make the overall verdict RED whenever any gating probe is
RED or MANUAL, (3) cold-verify MANIFEST hashes by content (a one-byte
edit flips a GREEN to RED), and (4) classify first-party `src.*` imports
as a violation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "ci" / "release_gate.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("release_gate", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_gate"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> Any:
    return _load()


def test_all_probes_return_valid_status(mod: Any) -> None:
    valid = {mod.GREEN, mod.RED, mod.MANUAL}
    for r in mod.evaluate(False):
        assert r.status in valid, f"{r.pid} returned {r.status}"
        assert r.evidence, f"{r.pid} has empty evidence"


def test_red_or_manual_forces_red_verdict(mod: Any) -> None:
    rc = mod.main(["--json", "/dev/null"])
    # On the live tree at least one gating probe is RED/MANUAL, so the gate
    # must exit nonzero. (If the repo ever reaches all-GREEN this asserts the
    # inverse contract and should be updated deliberately.)
    assert rc == 1


def test_manifest_coldverify_detects_tamper(mod: Any, tmp_path: Path) -> None:
    target = tmp_path / "payload.txt"
    target.write_text("original\n", encoding="utf-8")
    good = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = tmp_path / "MANIFEST.sha256"
    manifest.write_text(f"{good}  payload.txt\n", encoding="utf-8")

    orig_root = mod.ROOT
    try:
        mod.ROOT = tmp_path
        status, _ = mod.probe_d_manifest_coldverify(False)
        assert status == mod.GREEN
        target.write_text("tampered\n", encoding="utf-8")
        status2, ev = mod.probe_d_manifest_coldverify(False)
        assert status2 == mod.RED
        assert "mismatch" in ev
    finally:
        mod.ROOT = orig_root


def test_src_import_regex_matches_first_party(mod: Any) -> None:
    assert mod._SRC_IMPORT_RE.search("from src.core import X")
    assert mod._SRC_IMPORT_RE.search("import src")
    assert not mod._SRC_IMPORT_RE.search("from geosync.core import X")
    assert not mod._SRC_IMPORT_RE.search("from mysrc import X")


def test_path_hack_regex(mod: Any) -> None:
    assert mod._PATH_HACK_RE.search("sys.path.insert(0, '..')")
    assert mod._PATH_HACK_RE.search("sys.path.append(x)")
    assert not mod._PATH_HACK_RE.search("path = sys.path")
