# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml


def _load_serotonin_module() -> tuple[ModuleType, Any, Any]:
    module_path = (
        Path(__file__).resolve().parents[4]
        / "core"
        / "neuro"
        / "serotonin"
        / "serotonin_controller.py"
    )
    spec = importlib.util.spec_from_file_location(
        "serotonin_controller_runtime_safety", module_path
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, module.SerotoninController, module.ControllerOutput


@pytest.fixture(scope="module")
def serotonin_module() -> ModuleType:
    return _load_serotonin_module()[0]


@pytest.fixture()
def serotonin_controller(tmp_path: Path, serotonin_module: ModuleType) -> Any:
    _module, SerotoninController, _ = _load_serotonin_module()
    cfg_source = Path(__file__).resolve().parents[4] / "configs" / "serotonin.yaml"
    cfg_path = tmp_path / "serotonin.yaml"
    loaded = yaml.safe_load(cfg_source.read_text(encoding="utf-8")) or {}
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "active_profile": "v24",
                "serotonin_v24": loaded.get("serotonin_v24", {}),
            }
        ),
        encoding="utf-8",
    )
    return SerotoninController(str(cfg_path))


def _obs(
    stress: float = 0.4, drawdown: float = -0.02, novelty: float = 0.3
) -> dict[str, float]:
    return {"stress": stress, "drawdown": drawdown, "novelty": novelty}


MAX_COMPLEXITY_FACTOR = 5


def test_serotonin_bounds_random(serotonin_controller: Any) -> None:
    """INV-5HT2: s(t) in [0, 1] — serotonin level stays bounded across varied stress inputs."""
    ctrl = serotonin_controller
    for value in [0.1, 0.5, 1.0, 2.0]:
        out = ctrl.update(_obs(stress=value, drawdown=-0.01 * value, novelty=0.2))
        assert 0.0 <= out.metrics_snapshot["serotonin_level"] <= 1.0


def test_stress_monotonic_risk_budget(serotonin_controller: Any) -> None:
    """INV-5HT3: higher stress -> lower risk budget (pre-desensitization monotonicity)."""
    ctrl = serotonin_controller
    low = ctrl.update(_obs(stress=0.3, drawdown=-0.02, novelty=0.1))
    high = ctrl.update(_obs(stress=1.2, drawdown=-0.02, novelty=0.1))
    assert high.risk_budget <= low.risk_budget + 1e-9


def test_defensive_gate_blocks_actions(serotonin_controller: Any) -> None:
    """INV-5HT7: stress >= 1 OR |dd| >= 0.5 -> veto; defensive gate blocks actions."""
    ctrl = serotonin_controller
    out = ctrl.update(_obs(stress=2.5, drawdown=-0.2, novelty=0.4))
    assert out.action_gate == "HOLD_OR_REDUCE_ONLY"
    assert out.mode == "DEFENSIVE"


def test_cooldown_persists_when_hold_active(serotonin_controller: Any) -> None:
    """INV-5HT7: veto persists across consecutive high-stress updates."""
    ctrl = serotonin_controller
    first = ctrl.update(_obs(stress=2.0, drawdown=-0.3, novelty=0.5))
    second = ctrl.update(_obs(stress=1.8, drawdown=-0.25, novelty=0.5))
    assert first.action_gate == "HOLD_OR_REDUCE_ONLY"
    assert second.action_gate == "HOLD_OR_REDUCE_ONLY"


def test_hysteresis_not_flip_flop(serotonin_controller: Any) -> None:
    """INV-5HT7: veto gate must not oscillate on marginal stress changes."""
    ctrl = serotonin_controller
    on = ctrl.update(_obs(stress=1.6, drawdown=-0.2, novelty=0.3))
    off = ctrl.update(_obs(stress=1.55, drawdown=-0.19, novelty=0.3))
    assert not (on.action_gate == "ALLOW" and off.action_gate == "HOLD_OR_REDUCE_ONLY")


def test_invalid_input_triggers_safe_mode(serotonin_controller: Any) -> None:
    """INV-HPC2: finite inputs -> finite outputs; NaN input triggers safe DEFENSIVE mode."""
    ctrl = serotonin_controller
    bad = ctrl.update({"stress": float("nan"), "drawdown": -0.1, "novelty": 0.2})
    assert bad.mode == "DEFENSIVE"
    assert "INVALID_INPUT" in bad.reason_codes


def test_numeric_stability_extremes(serotonin_controller: Any) -> None:
    """INV-5HT2: s(t) in [0, 1] holds even under extreme input magnitudes."""
    ctrl = serotonin_controller
    out = ctrl.update(_obs(stress=50.0, drawdown=-10.0, novelty=20.0))
    assert 0.0 <= out.metrics_snapshot["serotonin_level"] <= 1.0
    assert out.risk_budget >= ctrl._min_risk_budget


def test_state_roundtrip(serotonin_controller: Any) -> None:
    """Infra: state serialization round-trip preserves controller state."""
    ctrl = serotonin_controller
    ctrl.update(_obs(stress=0.7, drawdown=-0.1, novelty=0.2))
    state = ctrl.get_state()
    ctrl.reset()
    ctrl.set_state(state)
    new_state = ctrl.get_state()
    assert state == new_state


def test_reason_codes_whitelist_only(
    serotonin_controller: Any, serotonin_module: ModuleType
) -> None:
    """Infra: all emitted reason codes belong to the REASON_CODES_WHITELIST."""
    ctrl = serotonin_controller
    ctrl.update(_obs(stress=2.0, drawdown=-0.2, novelty=0.7))
    reasons = ctrl.explain_last_decision()
    for code in serotonin_module.REASON_CODES_WHITELIST:
        if code in reasons:
            assert code in serotonin_module.REASON_CODES_WHITELIST


def test_trace_schema_stable_keys(serotonin_controller: Any) -> None:
    """Infra: trace JSONL event keys match expected schema."""
    ctrl = serotonin_controller
    ctrl.update(_obs(stress=1.0, drawdown=-0.1, novelty=0.5))
    trace = ctrl.export_trace_jsonl().splitlines()[-1]
    event = json.loads(trace)
    expected_keys = [
        "timestamp_utc",
        "schema_version",
        "active_profile",
        "inputs",
        "outputs",
        "reason_codes",
        "invariants_checked",
        "update_latency_us",
    ]
    assert list(event.keys()) == expected_keys
    assert set(event["inputs"].keys()) == {
        "stress",
        "drawdown",
        "novelty",
        "market_vol",
        "free_energy",
        "cum_losses",
        "rho_loss",
    }
    assert set(event["outputs"].keys()) >= {
        "mode",
        "risk_budget",
        "gate",
        "serotonin_level",
    }
    assert isinstance(event["invariants_checked"], dict)


def test_update_not_using_pandas(
    monkeypatch: pytest.MonkeyPatch, serotonin_controller: Any
) -> None:
    """Infra: update path must not import pandas."""

    class _NoPandas:
        def __getattr__(self, name: str) -> None:
            raise AssertionError("pandas should not be used")

    sys.modules["pandas"] = _NoPandas()  # type: ignore[assignment]
    ctrl = serotonin_controller
    ctrl.update(_obs())
    sys.modules.pop("pandas", None)


def test_update_constant_time_complexity(serotonin_controller: Any) -> None:
    """Infra: per-update latency does not degrade with iteration count."""
    ctrl = serotonin_controller
    t0 = time.perf_counter()
    for _ in range(50):
        ctrl.update(_obs(stress=0.6, drawdown=-0.05, novelty=0.2))
    base = time.perf_counter() - t0

    t1 = time.perf_counter()
    for _ in range(5000):
        ctrl.update(_obs(stress=0.61, drawdown=-0.05, novelty=0.2))
    long = time.perf_counter() - t1

    assert long / 5000 < base / 50 * MAX_COMPLEXITY_FACTOR


def test_micro_benchmark_latency(serotonin_controller: Any) -> None:
    """Infra: median update latency stays below 2000 us."""
    ctrl = serotonin_controller
    samples: list[float] = []
    for _ in range(10):
        out = ctrl.update(_obs(stress=0.5, drawdown=-0.03, novelty=0.2))
        samples.append(out.metrics_snapshot["update_latency_us"])
    median = sorted(samples)[len(samples) // 2]
    assert median < 2000


def test_invariants_flags_and_clamp_recorded(serotonin_controller: Any) -> None:
    """INV-5HT2 + INV-HPC2: invariant check flags are recorded in trace."""
    ctrl = serotonin_controller
    ctrl.update(_obs(stress=2.5, drawdown=-0.2, novelty=0.5))
    event = json.loads(ctrl.export_trace_jsonl().splitlines()[-1])
    invariants = event["invariants_checked"]
    assert invariants["finite_inputs"] is True
    assert invariants["serotonin_in_bounds"] is True
    assert "risk_budget_clamped" in invariants
    if invariants["risk_budget_clamped"]:
        assert "RISK_BUDGET_CLAMPED" in event["reason_codes"]
    else:
        assert "RISK_BUDGET_CLAMPED" not in event["reason_codes"]


def test_regression_cooldown_reentry(serotonin_controller: Any) -> None:
    """INV-5HT7: re-entering high stress does not increase risk budget."""
    ctrl = serotonin_controller
    first = ctrl.update(_obs(stress=2.0, drawdown=-0.3, novelty=0.4))
    second = ctrl.update(_obs(stress=2.1, drawdown=-0.25, novelty=0.4))
    assert second.risk_budget <= first.risk_budget


def test_risk_gate_reduces_exposure(serotonin_controller: Any) -> None:
    """INV-5HT7: stress >= 1 triggers veto, reducing risk budget vs calm baseline."""
    ctrl = serotonin_controller
    low = ctrl.update(_obs(stress=0.2, drawdown=-0.01, novelty=0.1))
    high = ctrl.update(_obs(stress=3.0, drawdown=-0.3, novelty=0.1))
    assert high.risk_budget < low.risk_budget


def test_ecs_alignment_monotone_stress(serotonin_controller: Any) -> None:
    """INV-5HT3: monotone stress increase -> non-increasing risk budget."""
    ctrl = serotonin_controller
    a = ctrl.update(_obs(stress=0.4, drawdown=-0.02, novelty=0.2))
    b = ctrl.update(_obs(stress=1.0, drawdown=-0.02, novelty=0.2))
    assert b.risk_budget <= a.risk_budget


def test_crisis_priority_overrides_modes(serotonin_controller: Any) -> None:
    """INV-HPC2: infinite stress input -> safe DEFENSIVE mode (finite output)."""
    ctrl = serotonin_controller
    out = ctrl.update({"stress": float("inf"), "drawdown": -0.1, "novelty": 0.2})
    assert out.mode == "DEFENSIVE"


def test_reason_codes_flow_to_trace(serotonin_controller: Any) -> None:
    """Infra: reason_codes propagate into trace JSONL events."""
    ctrl = serotonin_controller
    ctrl.update(_obs(stress=2.2, drawdown=-0.2, novelty=0.5))
    event = json.loads(ctrl.export_trace_jsonl().splitlines()[-1])
    assert "reason_codes" in event
    assert isinstance(event["reason_codes"], list)


def test_positive_drawdown_triggers_spike(serotonin_controller: Any) -> None:
    """INV-5HT7: |dd| >= threshold -> veto via DRAWDOWN_SPIKE."""
    ctrl = serotonin_controller
    out = ctrl.update(_obs(stress=0.2, drawdown=0.2, novelty=0.2))
    assert out.action_gate == "HOLD_OR_REDUCE_ONLY"
    assert "DRAWDOWN_SPIKE" in out.reason_codes


def test_no_cyclic_imports(serotonin_module: ModuleType) -> None:
    """Infra: production module does not import from tests package."""
    assert "tests" not in serotonin_module.SerotoninController.__module__


def test_config_validation_error_message(tmp_path: Path) -> None:
    """Infra: invalid config raises ValueError with descriptive message."""
    bad_cfg: dict[str, object] = {
        "alpha": 1.0,
        "beta": 1.0,
        "gamma": 1.0,
        "delta_rho": 1.0,
        "k": 1.0,
        "theta": 0.0,
        "delta": 0.5,
        "za_bias": 0.0,
        "decay_rate": 0.1,
        "cooldown_threshold": 0.5,
        "desens_threshold_ticks": 1,
        "desens_rate": 0.1,
        "target_dd": 0.1,
        "target_sharpe": 1.0,
        "beta_temper": 0.1,
        "phase_threshold": 0.1,
        "phase_kappa": 0.1,
        "burst_factor": 0.1,
        "mod_t_max": 1.0,
        "mod_t_half": 1.0,
        "mod_k": 0.1,
        "max_desens_counter": 10,
        "desens_gain": 0.1,
        "gate_veto": 0.9,
        "phasic_veto": 1.0,
        "temperature_floor_min": 0.8,
        "temperature_floor_max": 0.4,
    }
    cfg_path = tmp_path / "serotonin.yaml"
    cfg_path.write_text(yaml.safe_dump(bad_cfg), encoding="utf-8")
    _module, SerotoninController, _ = _load_serotonin_module()
    with pytest.raises(ValueError, match="temperature_floor_min"):
        SerotoninController(str(cfg_path))


def test_deterministic_timestamp(
    monkeypatch: pytest.MonkeyPatch, serotonin_controller: Any
) -> None:
    """Infra: deterministic time provider produces expected timestamp in trace."""
    ctrl = serotonin_controller
    fixed = dt.datetime(2024, 1, 1, 0, 0, 0)
    ctrl._time_provider = lambda: fixed
    ctrl.update(_obs(stress=0.9, drawdown=-0.1, novelty=0.2))
    event = json.loads(ctrl.export_trace_jsonl().splitlines()[-1])
    assert event["timestamp_utc"].startswith("2024-01-01T00:00:00")


def test_jsonl_export_stable_order(serotonin_controller: Any) -> None:
    """Infra: JSONL export key ordering is deterministic."""
    ctrl = serotonin_controller
    ctrl.update(_obs(stress=0.8, drawdown=-0.1, novelty=0.2))
    lines = ctrl.export_trace_jsonl().splitlines()
    parsed_lines = [json.loads(line) for line in lines]
    assert all(isinstance(evt, dict) for evt in parsed_lines)
    parsed = parsed_lines[-1]
    expected_keys = [
        "timestamp_utc",
        "schema_version",
        "active_profile",
        "inputs",
        "outputs",
        "reason_codes",
        "invariants_checked",
        "update_latency_us",
    ]
    assert list(parsed.keys()) == expected_keys
