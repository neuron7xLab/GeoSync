from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "check_methodology_stack_contract.py"
CONTRACT = ROOT / "docs" / "governance" / "methodology_stack_contract.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_methodology_stack_contract", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_methodology_stack_contract_contains_required_controls() -> None:
    module = _load_module()
    missing = module.validate_contract_text(CONTRACT.read_text(encoding="utf-8"))
    assert missing == []


def test_methodology_stack_contract_fails_closed_on_missing_control() -> None:
    module = _load_module()
    text = CONTRACT.read_text(encoding="utf-8").replace("negative evidence path", "")
    assert "negative evidence path" in module.validate_contract_text(text)
