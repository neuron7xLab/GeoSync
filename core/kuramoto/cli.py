# SPDX-License-Identifier: LicenseRef-TradePulse-Proprietary
"""Command-line interface for the Kuramoto simulation engine.

Entry point: ``tp-kuramoto`` (registered in ``pyproject.toml``).

Usage examples
--------------
# Run with defaults (N=10, K=1.0, dt=0.01, steps=1000)
tp-kuramoto simulate

# Strongly-coupled network, reproducible seed
tp-kuramoto simulate --N 50 --K 3.0 --steps 2000 --seed 42

# Pipe JSON output to another tool
tp-kuramoto simulate --N 20 --K 2.0 --quiet | jq .summary

# Save full result to file
tp-kuramoto simulate --N 30 --K 1.5 --output result.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click
import numpy as np

from .config import KuramotoConfig
from .engine import run_simulation


# ──────────────────────────────────────────────────────────────────────────────
# CLI group
# ──────────────────────────────────────────────────────────────────────────────


@click.group()
@click.version_option(version="1.0.0", prog_name="tp-kuramoto")
def cli() -> None:
    """TradePulse Kuramoto Simulation Engine.

    Simulate coupled phase oscillators using the Kuramoto model:

    \b
        dθᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(θⱼ − θᵢ)

    The order parameter R ∈ [0, 1] measures global synchronisation:
    R ≈ 0 → desynchronised,  R ≈ 1 → fully synchronised.

    \b
    Examples:
        tp-kuramoto simulate --N 50 --K 2.0 --steps 1000 --seed 0
        tp-kuramoto simulate --N 100 --K 0.5 --dt 0.005 --output out.json
    """


# ──────────────────────────────────────────────────────────────────────────────
# simulate sub-command
# ──────────────────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--N",
    "n_oscillators",
    type=int,
    default=10,
    show_default=True,
    help="Number of coupled oscillators (≥ 2).",
)
@click.option(
    "--K",
    "coupling",
    type=float,
    default=1.0,
    show_default=True,
    help="Global coupling strength.",
)
@click.option(
    "--dt",
    type=float,
    default=0.01,
    show_default=True,
    help="Integration time-step (> 0).",
)
@click.option(
    "--steps",
    type=int,
    default=1000,
    show_default=True,
    help="Number of RK4 integration steps (≥ 1).",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (omit for random).",
)
@click.option(
    "--omega",
    "omega_str",
    type=str,
    default=None,
    help=(
        "Comma-separated natural frequencies, e.g. '--omega 0.5,1.0,1.5'. "
        "Length must equal N. Overrides random draw."
    ),
)
@click.option(
    "--theta0",
    "theta0_str",
    type=str,
    default=None,
    help=(
        "Comma-separated initial phases in radians, e.g. '--theta0 0,1.57,3.14'. "
        "Length must equal N. Overrides random draw."
    ),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write JSON result to this file (optional).",
)
@click.option(
    "--quiet",
    is_flag=True,
    help="Suppress human-readable summary; emit only JSON.",
)
def simulate(
    n_oscillators: int,
    coupling: float,
    dt: float,
    steps: int,
    seed: int | None,
    omega_str: str | None,
    theta0_str: str | None,
    output_path: Path | None,
    quiet: bool,
) -> None:
    """Run a Kuramoto simulation and print summary statistics.

    The simulation integrates the Kuramoto ODE over ``steps`` steps with
    time-step ``dt`` using 4th-order Runge-Kutta.

    \b
    Output includes:
        - Final order parameter R (synchrony at end of simulation)
        - Mean / max R over entire trajectory
        - Phase trajectory shape confirmation
        - Optional full JSON dump to --output file

    \b
    Examples:
        tp-kuramoto simulate --N 50 --K 2.0 --steps 1000 --seed 42
        tp-kuramoto simulate --N 100 --K 3.0 --quiet | jq .summary
    """
    # ── parse optional array arguments ──────────────────────────────────────
    omega: np.ndarray | None = None
    theta0: np.ndarray | None = None

    if omega_str is not None:
        try:
            omega = np.array([float(v) for v in omega_str.split(",")], dtype=np.float64)
        except ValueError as exc:
            click.echo(f"Error parsing --omega: {exc}", err=True)
            sys.exit(1)

    if theta0_str is not None:
        try:
            theta0 = np.array(
                [float(v) for v in theta0_str.split(",")], dtype=np.float64
            )
        except ValueError as exc:
            click.echo(f"Error parsing --theta0: {exc}", err=True)
            sys.exit(1)

    # ── build and validate config ────────────────────────────────────────────
    try:
        cfg = KuramotoConfig(
            N=n_oscillators,
            K=coupling,
            dt=dt,
            steps=steps,
            seed=seed,
            omega=omega,
            theta0=theta0,
        )
    except Exception as exc:  # pydantic ValidationError or ValueError
        click.echo(f"Configuration error: {exc}", err=True)
        sys.exit(1)

    # ── run simulation ───────────────────────────────────────────────────────
    try:
        result = run_simulation(cfg)
    except Exception as exc:
        click.echo(f"Simulation error: {exc}", err=True)
        sys.exit(1)

    # ── build output payload ─────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "summary": result.summary,
        "config": cfg.to_dict(),
    }

    if not quiet:
        _print_summary(result.summary)

    # ── save to file ─────────────────────────────────────────────────────────
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Include full trajectories only when writing to file
        payload["order_parameter"] = result.order_parameter.tolist()
        payload["time"] = result.time.tolist()
        payload["phases_shape"] = list(result.phases.shape)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if not quiet:
            click.echo(f"\nFull result saved to: {output_path}")

    if quiet:
        click.echo(json.dumps(payload, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _print_summary(summary: dict[str, Any]) -> None:
    """Render a human-readable summary table to stdout."""
    click.echo("")
    click.echo("━" * 50)
    click.echo("  Kuramoto Simulation — Summary")
    click.echo("━" * 50)
    click.echo(f"  Oscillators (N)  : {summary['N']}")
    click.echo(f"  Coupling (K)     : {summary['K']:.4f}")
    click.echo(f"  Time-step (dt)   : {summary['dt']:.6f}")
    click.echo(f"  Steps            : {summary['steps']}")
    click.echo(f"  Total time       : {summary['total_time']:.4f}")
    click.echo("─" * 50)
    click.echo(f"  Final R          : {summary['final_R']:.6f}")
    click.echo(f"  Mean R           : {summary['mean_R']:.6f}")
    click.echo(f"  Max  R           : {summary['max_R']:.6f}")
    click.echo(f"  Min  R           : {summary['min_R']:.6f}")
    click.echo(f"  Std  R           : {summary['std_R']:.6f}")
    click.echo("━" * 50)
    click.echo("")


def main() -> None:
    """Registered entry point for ``tp-kuramoto``."""
    cli()


if __name__ == "__main__":
    main()
