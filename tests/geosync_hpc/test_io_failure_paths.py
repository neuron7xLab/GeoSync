# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Failure-path coverage for GeoSync HPC data and artifact I/O."""

from __future__ import annotations

from pathlib import Path

import pytest

from geosync_hpc.data import read_ticks_csv


def test_read_ticks_csv_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_ticks_csv(tmp_path / "missing_ticks.csv")


def test_read_ticks_csv_corrupted_timestamp_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "corrupted.csv"
    csv_path.write_text(
        "timestamp,mid,bid,ask,bid_size,ask_size,last,last_size\n"
        "not-a-date,100,99,101,10,10,100,1\n",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, TypeError)):
        read_ticks_csv(csv_path)
