from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from cortex_service.scripts.calibrate_lyapunov_threshold import (
    evaluate_threshold,
    load_points,
    transition_steps,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "lyapunov", "regime_label"])
        writer.writeheader()
        writer.writerows(rows)


def test_load_points_valid_csv(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    _write_csv(
        path,
        [
            {"step": 1, "lyapunov": 0.4, "regime_label": "neutral"},
            {"step": 2, "lyapunov": 0.8, "regime_label": "bullish"},
        ],
    )
    points = load_points(path)
    assert len(points) == 2
    assert points[1].label == "bullish"


def test_load_points_missing_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text("step,lyapunov\n1,0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing columns"):
        load_points(path)


def load_points_from_rows(rows: list[dict[str, object]]):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as handle:
        tmp = Path(handle.name)
    _write_csv(tmp, rows)
    return load_points(tmp)


def test_transition_steps_empty_single_multiple() -> None:
    assert transition_steps([]) == []
    assert (
        transition_steps(
            load_points_from_rows([{"step": 1, "lyapunov": 0.1, "regime_label": "n"}])
        )
        == []
    )
    pts = load_points_from_rows(
        [
            {"step": 1, "lyapunov": 0.1, "regime_label": "n"},
            {"step": 2, "lyapunov": 0.2, "regime_label": "n"},
            {"step": 3, "lyapunov": 0.3, "regime_label": "b"},
            {"step": 4, "lyapunov": 0.4, "regime_label": "b"},
            {"step": 5, "lyapunov": 0.5, "regime_label": "n"},
        ]
    )
    assert transition_steps(pts) == [3, 5]


def test_evaluate_threshold_no_transitions() -> None:
    pts = load_points_from_rows(
        [
            {"step": 1, "lyapunov": 0.2, "regime_label": "n"},
            {"step": 2, "lyapunov": 0.3, "regime_label": "n"},
        ]
    )
    mean_lead, covered = evaluate_threshold(pts, 0.4)
    assert mean_lead == 0.0
    assert covered == 0


def test_evaluate_threshold_known_leadtime() -> None:
    pts = load_points_from_rows(
        [
            {"step": 1, "lyapunov": 0.1, "regime_label": "n"},
            {"step": 2, "lyapunov": 0.8, "regime_label": "n"},
            {"step": 3, "lyapunov": 0.2, "regime_label": "b"},
        ]
    )
    mean_lead, covered = evaluate_threshold(pts, 0.7)
    assert mean_lead == 1.0
    assert covered == 1


def test_main_e2e(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    _write_csv(
        path,
        [
            {"step": 1, "lyapunov": 0.1, "regime_label": "n"},
            {"step": 2, "lyapunov": 0.9, "regime_label": "n"},
            {"step": 3, "lyapunov": 0.2, "regime_label": "b"},
        ],
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cortex_service.scripts.calibrate_lyapunov_threshold",
            "--csv",
            str(path),
            "--min",
            "0.5",
            "--max",
            "0.9",
            "--step",
            "0.1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "best_threshold=" in result.stdout
