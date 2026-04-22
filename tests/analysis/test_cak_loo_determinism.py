"""T5 · one LOO configuration run twice produces identical numeric output."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPT = REPO / "scripts" / "analysis_cak_leave_one_out.py"


def _load():
    spec = importlib.util.spec_from_file_location("loo_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_baseline_full_is_deterministic() -> None:
    """Run the frozen baseline twice; all scalars must be bit-equal."""
    mod = _load()
    _, a = mod._run(
        mod.PARAMS.regime_assets,
        mod.PARAMS.strategy_assets,
        mod.PARAMS.regime_buckets,
    )
    _, b = mod._run(
        mod.PARAMS.regime_assets,
        mod.PARAMS.strategy_assets,
        mod.PARAMS.regime_buckets,
    )
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and not np.isfinite(va):
            assert not np.isfinite(vb)
        else:
            assert va == vb, f"LOO baseline non-deterministic on {k}: {va} != {vb}"


def test_regime_loo_single_asset_deterministic() -> None:
    """Regime LOO omitting BTC twice produces identical metrics."""
    mod = _load()
    regime = tuple(a for a in mod.PARAMS.regime_assets if a != "BTC")
    _, a = mod._run(regime, mod.PARAMS.strategy_assets, mod.PARAMS.regime_buckets)
    _, b = mod._run(regime, mod.PARAMS.strategy_assets, mod.PARAMS.regime_buckets)
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and not np.isfinite(va):
            assert not np.isfinite(vb)
        else:
            assert va == vb, f"regime LOO omit=BTC non-deterministic on {k}"
