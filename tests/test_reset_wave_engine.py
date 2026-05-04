from __future__ import annotations

from geosync.neuroeconomics.reset_wave_engine import (
    ResetWaveConfig,
    latent_interpretive_forecast_layer,
    run_reset_wave,
)


def test_reset_wave_engine_converges_and_decreases_potential() -> None:
    cfg = ResetWaveConfig(coupling_gain=1.0, dt=0.1, steps=64, convergence_tol=0.02)
    out = run_reset_wave([0.4, -0.3, 0.2], [0.0, 0.0, 0.0], cfg)
    assert not out.locked
    assert out.final_potential <= out.initial_potential
    assert out.converged


def test_reset_wave_engine_lock_on_extreme_drift() -> None:
    cfg = ResetWaveConfig(max_phase_error=0.5)
    out = run_reset_wave([2.0, -2.0], [0.0, 0.0], cfg)
    assert out.locked
    assert not out.converged
    assert out.final_potential == out.initial_potential


def test_reset_wave_engine_deterministic_trajectory() -> None:
    cfg = ResetWaveConfig(coupling_gain=0.8, dt=0.05, steps=20)
    a = run_reset_wave([0.1, -0.2], [0.0, 0.0], cfg)
    b = run_reset_wave([0.1, -0.2], [0.0, 0.0], cfg)
    assert a == b


def test_reset_wave_engine_rk4_fixed_and_euler_are_stable() -> None:
    base = [0.0, 0.0, 0.0]
    node = [0.4, -0.3, 0.2]
    rk = run_reset_wave(node, base, ResetWaveConfig(integrator="rk4_fixed", steps=50, dt=0.05))
    eu = run_reset_wave(node, base, ResetWaveConfig(integrator="euler", steps=50, dt=0.05))
    assert rk.final_potential <= rk.initial_potential
    assert eu.final_potential <= eu.initial_potential


def test_forecast_layer_outputs_supported_class() -> None:
    out = run_reset_wave([0.3, -0.2], [0.0, 0.0], ResetWaveConfig())
    cls, conf = latent_interpretive_forecast_layer(out)
    assert cls in {"CONVERGING", "LOCKED", "OSCILLATORY", "DIVERGING", "UNSTABLE", "UNKNOWN"}
    assert 0.0 <= conf <= 1.0
