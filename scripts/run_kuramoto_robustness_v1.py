#!/usr/bin/env python3
# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Kuramoto robustness-framework v1 CLI runner.

Read-only on every frozen input. Writes strictly under
``results/cross_asset_kuramoto/robustness_v1/``:

- ``verdict.json``          — terminal decision + suite scalars
- ``cpcv_summary.json``     — PBO, PSR, fold sharpes
- ``null_summary.json``     — two null families + p-values
- ``jitter_summary.json``   — placeholder evaluator output
- ``ROBUSTNESS_v1.md``      — 1-page human-readable report

A hash-mismatch on any frozen artifact exits 2 with a ``FAIL`` verdict
file already written to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from backtest.robustness_gates import DecisionLabel, evaluate_robustness_gates  # noqa: E402
from research.robustness.protocols.kuramoto_contract import (  # noqa: E402
    FrozenArtifactMismatch,
    KuramotoRobustnessContract,
)
from research.robustness.protocols.kuramoto_gate_runner import (  # noqa: E402
    run_kuramoto_gate_runner,
)

OUT_DIR = REPO / "results" / "cross_asset_kuramoto" / "robustness_v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _render_markdown(
    verdict_label: str,
    cpcv_dict: dict[str, Any],
    null_dict: dict[str, Any],
    jitter_dict: dict[str, Any],
    reasons: tuple[str, ...],
) -> str:
    lines = [
        "# Cross-asset Kuramoto · Robustness v1 report",
        "",
        f"Terminal decision: **{verdict_label}**",
        "",
        "## Suite summary",
        "",
        "| Suite | Metric | Value | Pass |",
        "|---|---|---:|:-:|",
        f"| CPCV | PBO (fold mirror) | {cpcv_dict['pbo']:.4f} | {'✓' if cpcv_dict['pbo_pass'] else '✗'} |",
        f"| CPCV | PSR (daily) | {cpcv_dict['psr_daily']:.4f} | {'✓' if cpcv_dict['psr_pass'] else '✗'} |",
        f"| CPCV | Annualised Sharpe (daily) | {cpcv_dict['annualised_sharpe']:.4f} | n/a |",
    ]
    loo_pbo = cpcv_dict.get("loo_pbo")
    if loo_pbo is not None:
        lines.append(
            f"| CPCV | PBO (LOO grid, n={cpcv_dict['loo_n_strategies']}) | "
            f"{loo_pbo:.4f} | "
            f"{'✓' if cpcv_dict['loo_pbo_pass'] else '✗'} |"
        )
    for family in null_dict["families"]:
        lines.append(
            f"| Null | {family['family']} p-value | "
            f"{family['p_value']:.4f} | "
            f"{'✓' if family['p_value_pass'] else '✗'} |"
        )
    lines.extend(
        [
            f"| Jitter | fraction_within_tol | "
            f"{jitter_dict['stability']['fraction_within_tol']:.4f} | "
            f"{'✓' if jitter_dict['fraction_within_tol_pass'] else '✗'} |",
            f"| Jitter | evaluator_mode | `{jitter_dict['evaluator_mode']}` | n/a |",
            "",
            "## Reasons",
            "",
        ]
    )
    if reasons:
        lines.extend(f"- {r}" for r in reasons)
    else:
        lines.append("- (none — all gates green)")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Evidence is derived from the frozen `offline_robustness/"
            "SOURCE_HASHES.json` bundle; 28 artifacts hash-verified.",
            "- Null suite uses cumulative-return pct_change as a return proxy;"
            " raw `net_ret` is not in the frozen demo bundle, which limits"
            " statistical power relative to the published headline Sharpe"
            " (`risk_metrics.csv::sharpe = 1.2619`).",
            "- Jitter evaluator is PLACEHOLDER_APPROXIMATION: rebuild under"
            " perturbed parameters requires the raw asset panel.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="null resamples per family (default: 1000)",
    )
    parser.add_argument(
        "--n-jitter-candidates",
        type=int,
        default=64,
        help="jitter candidates (default: 64)",
    )
    parser.add_argument(
        "--require-live-jitter",
        action="store_true",
        help="demote to INSUFFICIENT_EVIDENCE if jitter is placeholder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"output directory (default: {OUT_DIR})",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        contract = KuramotoRobustnessContract.from_frozen_artifacts()
    except FrozenArtifactMismatch as exc:
        verdict = {
            "label": DecisionLabel.FAIL.value,
            "reasons": (f"frozen-artifact integrity failure: {exc}",),
        }
        _write_json(args.out_dir / "verdict.json", verdict)
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    evidence = run_kuramoto_gate_runner(
        contract,
        null_kwargs={"n_bootstrap": args.n_bootstrap},
        jitter_kwargs={"n_candidates": args.n_jitter_candidates},
    )
    decision = evaluate_robustness_gates(
        evidence,
        require_live_jitter=args.require_live_jitter,
    )

    cpcv_dict = asdict(evidence.cpcv)
    null_dict = {
        "all_families_pass": evidence.null.all_families_pass,
        "families": [asdict(f) for f in evidence.null.families],
    }
    jitter_dict = {
        "evaluator_mode": evidence.jitter.evaluator_mode,
        "fraction_within_tol_pass": evidence.jitter.fraction_within_tol_pass,
        "pass_threshold": evidence.jitter.pass_threshold,
        "stability": asdict(evidence.jitter.stability),
    }

    _write_json(args.out_dir / "cpcv_summary.json", cpcv_dict)
    _write_json(args.out_dir / "null_summary.json", null_dict)
    _write_json(args.out_dir / "jitter_summary.json", jitter_dict)
    _write_json(
        args.out_dir / "verdict.json",
        {
            "label": decision.label.value,
            "cpcv_pass": decision.cpcv_pass,
            "null_pass": decision.null_pass,
            "jitter_pass": decision.jitter_pass,
            "jitter_is_placeholder": decision.jitter_is_placeholder,
            "reasons": list(decision.reasons),
            "contract_manifest_generated_utc": contract.manifest.generated_utc,
            "contract_manifest_regenerated_utc": contract.manifest.regenerated_utc,
        },
    )
    (args.out_dir / "ROBUSTNESS_v1.md").write_text(
        _render_markdown(
            verdict_label=decision.label.value,
            cpcv_dict=cpcv_dict,
            null_dict=null_dict,
            jitter_dict=jitter_dict,
            reasons=decision.reasons,
        ),
        encoding="utf-8",
    )

    print(f"verdict: {decision.label.value}")
    for r in decision.reasons:
        print(f"  - {r}")
    return 0 if decision.label is DecisionLabel.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
