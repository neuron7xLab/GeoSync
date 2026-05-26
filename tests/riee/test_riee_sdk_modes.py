from __future__ import annotations

import os

from runtime.riee.engine import KernelPanic
from runtime.riee.sdk import riee_guard
from scripts.riee.operational_modes import supported_modes


def test_supported_modes_declared() -> None:
    modes = {m.name for m in supported_modes()}
    assert modes == {'cloud_native', 'local_edge', 'application_sdk'}


def test_sdk_passthrough_when_disabled() -> None:
    os.environ['RIEE_ENABLE'] = '0'

    @riee_guard()
    def f() -> float:
        return 1.1

    assert f() == 1.1


def test_sdk_panics_when_enabled_and_drift() -> None:
    os.environ['RIEE_ENABLE'] = '1'

    @riee_guard()
    def f() -> float:
        return 1.1

    try:
        f()
    except KernelPanic:
        pass
    else:
        raise AssertionError('KernelPanic expected when RIEE enabled')
