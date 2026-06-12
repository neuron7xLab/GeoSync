# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from geosync.mfn.cli import build_parser, main


def test_points_argument_requires_number(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--points", "abc", "run"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "points must be an integer" in captured.err


def test_points_argument_rejects_zero(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--points", "0", "run"])

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "points must be >= 4" in captured.err


def test_points_argument_accepts_minimum() -> None:
    args = build_parser().parse_args(["--points", "4", "run"])
    assert args.points == 4
