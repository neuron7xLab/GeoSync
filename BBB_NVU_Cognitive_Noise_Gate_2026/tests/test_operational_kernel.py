# ruff: noqa: I001

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import load_yaml
from BBB_NVU_Cognitive_Noise_Gate_2026.src.operational_kernel import (
    OperationalKernel,
    RequestMapping,
    verify_operational_envelope,
)
from BBB_NVU_Cognitive_Noise_Gate_2026.tests.requirements import requirement

ROOT = Path(__file__).resolve().parents[1]
RULES = load_yaml(ROOT / "config" / "risk_rules.yaml")