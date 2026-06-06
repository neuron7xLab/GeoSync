# ruff: noqa: I001

from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import (
    canonical,
    load_yaml,
    sha256_text,
)
from BBB_NVU_Cognitive_Noise_Gate_2026.src.operational_kernel import (
    OperationalKernel,
    RequestMapping,
    verify_operational_envelope,
)
from BBB_NVU_Cognitive_Noise_Gate_2026.tests.requirements import requirement

ROOT = Path(__file__).resolve().parents[1]
RULES = load_yaml(ROOT / "config" / "risk_rules.yaml")
FIXED_TIME = "2026-06-03T00:00:00Z"
FIXED_ENGINE_HASH = "engine-hash-for-operational-kernel-tests"

STABLE_INPUT: dict[str, Any] = {
    "subject_id": "S-ST