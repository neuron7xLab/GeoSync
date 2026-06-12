# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""MFN stage numeric contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from geosync.mfn.contract import MFNContract
from geosync.mfn.pipeline import compare, detect, forecast, write_json


def test_detect_rejects_missing_mean_return(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    write_json(
        out / "extract.json",
        {
            "schema_version": "mfn.integration.v1",
            "stage": "extract",
            "features": {"volatility": 0.1},
        },
    )

    with pytest.raises(ValueError, match="mean_return"):
        detect(out, contract=MFNContract())


def test_forecast_rejects_missing_last_price(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    write_json(
        out / "extract.json",
        {
            "schema_version": "mfn.integration.v1",
            "stage": "extract",
            "features": {"mean_return": 0.1},
        },
    )
    write_json(out / "detect.json", {"schema_version": "mfn.integration.v1", "stage": "detect"})

    with pytest.raises(ValueError, match="last_price"):
        forecast(out, contract=MFNContract())


def test_compare_rejects_bad_prediction_value(tmp_path: Path) -> None:
    out = tmp_path / "bundle"
    write_json(
        out / "extract.json",
        {
            "schema_version": "mfn.integration.v1",
            "stage": "extract",
            "features": {"last_price": 100.0},
        },
    )
    write_json(
        out / "forecast.json",
        {
            "schema_version": "mfn.integration.v1",
            "stage": "forecast",
            "predicted_next_price": "not-a-number",
        },
    )

    with pytest.raises(ValueError, match="predicted_next_price"):
        compare(out, contract=MFNContract())
