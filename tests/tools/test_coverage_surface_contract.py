from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.coverage.surface_contract import (
    list_unmapped_files,
    load_coverage_targets,
    map_file_to_surface,
    validate_target_contract,
)


def _write_targets(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


VALID_TOML = """
[global]
final_aspirational_gate = 98
current_release_gate = 90
diff_coverage_gate = 90

[surfaces.core]
paths = ["core/"]
short_term = 80
mid_term = 90
final = 95
claim_risk = "high"
rationale = "core rationale"

[surfaces.backtest]
paths = ["backtest/"]
short_term = 90
mid_term = 95
final = 98
claim_risk = "critical"
rationale = "backtest rationale"

[surfaces.execution]
paths = ["execution/"]
short_term = 80
mid_term = 90
final = 95
claim_risk = "critical"
rationale = "execution rationale"
""".strip()


def test_valid_coverage_targets_passes(tmp_path: Path) -> None:
    targets = load_coverage_targets(_write_targets(tmp_path / "targets.toml", VALID_TOML))
    assert validate_target_contract(targets) == []


def test_missing_global_threshold_fails(tmp_path: Path) -> None:
    bad = VALID_TOML.replace("diff_coverage_gate = 90\n", "")
    try:
        load_coverage_targets(_write_targets(tmp_path / "bad.toml", bad))
    except ValueError as exc:
        assert "Missing global threshold" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_invalid_threshold_order_fails(tmp_path: Path) -> None:
    bad = VALID_TOML.replace("short_term = 80", "short_term = 96", 1)
    targets = load_coverage_targets(_write_targets(tmp_path / "bad.toml", bad))
    errors = validate_target_contract(targets)
    assert any("threshold order invalid" in e for e in errors)


def test_threshold_outside_range_fails(tmp_path: Path) -> None:
    bad = VALID_TOML.replace("final = 98", "final = 101", 1)
    targets = load_coverage_targets(_write_targets(tmp_path / "bad.toml", bad))
    errors = validate_target_contract(targets)
    assert any("out of range" in e for e in errors)


def test_missing_rationale_fails(tmp_path: Path) -> None:
    bad = VALID_TOML.replace('rationale = "core rationale"', 'rationale = ""')
    targets = load_coverage_targets(_write_targets(tmp_path / "bad.toml", bad))
    errors = validate_target_contract(targets)
    assert any("missing rationale" in e for e in errors)


def test_known_core_file_maps_to_core(tmp_path: Path) -> None:
    targets = load_coverage_targets(_write_targets(tmp_path / "targets.toml", VALID_TOML))
    assert map_file_to_surface("core/engine/core.py", targets) == "core"


def test_known_backtest_file_maps_to_backtest(tmp_path: Path) -> None:
    targets = load_coverage_targets(_write_targets(tmp_path / "targets.toml", VALID_TOML))
    assert map_file_to_surface("backtest/engine.py", targets) == "backtest"


def test_known_execution_file_maps_to_execution(tmp_path: Path) -> None:
    targets = load_coverage_targets(_write_targets(tmp_path / "targets.toml", VALID_TOML))
    assert map_file_to_surface("execution/oms.py", targets) == "execution"


def test_unknown_file_returns_none(tmp_path: Path) -> None:
    targets = load_coverage_targets(_write_targets(tmp_path / "targets.toml", VALID_TOML))
    assert map_file_to_surface("docs/readme.md", targets) is None
    assert list_unmapped_files(["docs/readme.md"], targets) == ["docs/readme.md"]


def test_cli_exits_0_for_valid_contract(tmp_path: Path) -> None:
    target_path = _write_targets(tmp_path / "targets.toml", VALID_TOML)
    run = subprocess.run(
        [sys.executable, "tools/coverage/validate_coverage_targets.py", "--targets", str(target_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_cli_exits_1_for_invalid_contract(tmp_path: Path) -> None:
    bad = VALID_TOML.replace("mid_term = 90", "mid_term = -1", 1)
    target_path = _write_targets(tmp_path / "bad.toml", bad)
    run = subprocess.run(
        [sys.executable, "tools/coverage/validate_coverage_targets.py", "--targets", str(target_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1


def test_json_output_written_when_requested(tmp_path: Path) -> None:
    target_path = _write_targets(tmp_path / "targets.toml", VALID_TOML)
    files_path = tmp_path / "files.txt"
    files_path.write_text("core/a.py\ndocs/a.md\n", encoding="utf-8")
    out_path = tmp_path / "result.json"
    run = subprocess.run(
        [
            sys.executable,
            "tools/coverage/validate_coverage_targets.py",
            "--targets",
            str(target_path),
            "--files",
            str(files_path),
            "--json-out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "pass"



def test_weights_must_sum_to_one(tmp_path: Path) -> None:
    bad = VALID_TOML + "\n\n[weights]\ndivergence = 0.7\nconvergence = 0.4\n"
    targets = load_coverage_targets(_write_targets(tmp_path / "bad_weights.toml", bad))
    errors = validate_target_contract(targets)
    assert any("weights must sum to 1.0" in e for e in errors)


def test_cli_json_includes_weights(tmp_path: Path) -> None:
    weighted = VALID_TOML + "\n\n[weights]\ndivergence = 0.5\nconvergence = 0.5\n"
    target_path = _write_targets(tmp_path / "targets_weighted.toml", weighted)
    out_path = tmp_path / "result_weights.json"
    run = subprocess.run([sys.executable, "tools/coverage/validate_coverage_targets.py", "--targets", str(target_path), "--json-out", str(out_path)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["weights"] == {"divergence": 0.5, "convergence": 0.5}
