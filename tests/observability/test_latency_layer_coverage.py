# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""IERD-Q6 four-layer latency coverage gate (Phase-4 progress).

IERD-PAI-FPS-UX-001 §5 / issue #531 require all four latency layers
(client_render, network_TTFB, server_compute, db_io) covered by
regression-gated CI. The Phase-4 ENTRY gate (#550) covered only
server_compute.

This module scores the layers the honest way (the Q7 device): every
covered layer cites a **real resolvable test**, statically verified to
exist (cross-language: a pytest ``path::func`` or a frontend Jest
spec). Layers with no genuine regression gate are explicit
``UNCOVERED`` with a reason — never assumed.

Genuinely met today and asserted **fail-closed**:

* ``server_compute`` — the existing p95 budget gate.
* ``client_render``  — the Web Vitals reporter shipped in this PR
  (``apps/web``) wired via ``next/web-vitals``, gated by the
  frontend-gate Jest run.
* ``db_io``           — ``observe_database_query`` collector-level
  regression test.
* The Grafana latency dashboard is structurally validated against the
  §5 budgets and the real Prometheus series.

Honest gap (``LayerCoverage`` = 0.75, informational at Phase-4):
``network_TTFB`` has no regression-gated budget assertion — Web Vitals
TTFB is instrumented client-side but a real-network/real-Lighthouse
budget run needs infra not exercised in unit CI. The "all 4 layers
regression-gated" criterion (#531) is therefore red by design and
env-gated (``GEOSYNC_LATENCY_LAYER_GATE``) so the global lanes stay
green; Phase-4 EXIT lands the Lighthouse-CI budget run + OTel trace
propagation and flips it fail-closed (claim ANCHORED).

Tracks claim ``e2e-latency-budget-compliance`` and issue IERD-Q6
(#531).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# §5 budgets per layer, in milliseconds (interactive folds into
# server_compute; db_io rolls into the server p95).
LAYER_BUDGETS_MS: Final[dict[str, int]] = {
    "client_render": 1000,
    "network_TTFB": 300,
    "server_compute": 200,
    "db_io": 100,
}

# layer → genuine resolvable regression target, or None ⇒ UNCOVERED.
_LAYER_COVERAGE: Final[dict[str, str | None]] = {
    "server_compute": (
        "tests/api/test_latency_budget_server_compute.py::test_endpoint_p95_within_budget"
    ),
    "client_render": "apps/web/app/__tests__/web-vitals-reporter.test.tsx",
    "db_io": (
        "tests/unit/test_metrics_collector.py::test_database_metrics_track_size_growth_and_latency"
    ),
    # network_TTFB → UNCOVERED: no regression-gated budget run; Web
    # Vitals TTFB is instrumented but not CI-asserted against a real
    # network. Phase-4 EXIT (Lighthouse CI + OTel).
    "network_TTFB": None,
}

_DASHBOARD: Final[Path] = _REPO_ROOT / "observability" / "grafana" / "latency_budget_dashboard.json"

_LAYER_GATE_ENV: Final[str] = "GEOSYNC_LATENCY_LAYER_GATE"


def _resolve(target: str) -> tuple[Path, str | None]:
    if "::" in target:
        rel, func = target.split("::", 1)
    else:
        rel, func = target, None
    return _REPO_ROOT / rel, func


def _target_exists(target: str) -> bool:
    """Cross-language static existence check (real falsification)."""
    path, func = _resolve(target)
    if not path.is_file():
        return False
    source = path.read_text(encoding="utf-8")
    if func is not None:
        return re.search(rf"^\s*def {re.escape(func)}\b", source, re.MULTILINE) is not None
    if path.suffix in {".tsx", ".ts"}:
        # A Jest/Playwright spec: must contain at least one test/it case.
        return re.search(r"\b(test|it)\s*\(", source) is not None
    return re.search(r"^\s*def test_\w+", source, re.MULTILINE) is not None


def _covered_layers() -> list[str]:
    return [layer for layer, t in _LAYER_COVERAGE.items() if t is not None]


def test_every_cited_layer_target_resolves() -> None:
    """Each cited regression target still exists (rename/delete = fail)."""
    missing = sorted(t for t in _LAYER_COVERAGE.values() if t is not None and not _target_exists(t))
    assert not missing, (
        f"IERD-Q6 layer matrix cites target(s) that no longer resolve: "
        f"{missing}. A covered layer whose gate vanished is not coverage."
    )


def test_server_compute_layer_gated_fail_closed() -> None:
    """server_compute is genuinely gated — asserted strictly."""
    target = _LAYER_COVERAGE["server_compute"]
    assert target is not None and _target_exists(target), (
        "IERD-Q6: the server_compute p95 budget gate "
        f"({target}) must exist — it is the one ANCHORED-grade layer."
    )


def test_client_render_instrumented_fail_closed() -> None:
    """client_render Web Vitals instrumentation is genuinely present."""
    target = _LAYER_COVERAGE["client_render"]
    assert target is not None and _target_exists(target), (
        "IERD-Q6: the Web Vitals reporter test must exist; client_render "
        "instrumentation is the genuine increment of this PR."
    )
    reporter = _REPO_ROOT / "apps" / "web" / "app" / "_components" / "web-vitals-reporter.tsx"
    assert reporter.is_file(), f"missing Web Vitals reporter at {reporter}"
    src = reporter.read_text(encoding="utf-8")
    assert (
        "useReportWebVitals" in src and "WEB_VITAL_BUDGETS_MS" in src
    ), "Web Vitals reporter must wire next/web-vitals and pin the §5 budget table."


def test_grafana_latency_dashboard_well_formed() -> None:
    """The latency dashboard covers all 4 layers at the §5 budgets."""
    assert _DASHBOARD.is_file(), f"missing dashboard at {_DASHBOARD}"
    spec = json.loads(_DASHBOARD.read_text(encoding="utf-8"))
    panels = spec.get("panels", [])
    by_layer = {p.get("layer"): p for p in panels}
    assert set(by_layer) == set(
        LAYER_BUDGETS_MS
    ), f"dashboard must have one panel per §5 layer; got {sorted(by_layer)}"
    for layer, budget in LAYER_BUDGETS_MS.items():
        panel = by_layer[layer]
        assert (
            panel.get("budget_ms") == budget
        ), f"{layer} panel budget_ms={panel.get('budget_ms')} ≠ §5 {budget}"
        targets = panel.get("targets", [])
        assert targets and all(
            t.get("expr") for t in targets
        ), f"{layer} panel must carry a non-empty Prometheus expr"


_layer_gate_only = pytest.mark.skipif(
    os.environ.get(_LAYER_GATE_ENV) != "1",
    reason=(
        "IERD-Q6 Phase-4 'all 4 layers regression-gated' check; runs only "
        f"in the dedicated latency-layer-coverage workflow ({_LAYER_GATE_ENV}"
        "=1, continue-on-error). Phase-4 EXIT lifts this when network_TTFB "
        "lands a regression-gated budget run."
    ),
)


@_layer_gate_only
def test_all_four_layers_regression_gated() -> None:
    """LayerCoverage == 1.0. Informational at Phase-4 (workflow continue-on-error)."""
    covered = _covered_layers()
    total = len(_LAYER_COVERAGE)
    score = len(covered) / total
    for layer, target in _LAYER_COVERAGE.items():
        status = "covered" if target is not None else "UNCOVERED"
        print(f"\n[layer] {layer:16s} {status:9s} {target or '-'}")
    print(f"\n[layer] AGGREGATE covered={len(covered)}/{total} LayerCoverage={score:.4f}")
    assert score >= 1.0, (
        f"IERD-Q6 §531: not all four latency layers are regression-gated "
        f"(LayerCoverage={score:.4f}; covered={sorted(covered)}). "
        f"network_TTFB needs a real-network/Lighthouse-CI budget run — "
        f"Phase-4 EXIT. Do NOT cite a non-asserting test to inflate this."
    )
