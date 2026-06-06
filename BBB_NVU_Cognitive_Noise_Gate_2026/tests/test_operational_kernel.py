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
    verify_operational_envelope,
)
from BBB_NVU_Cognitive_Noise_Gate_2026.tests.requirements import requirement

ROOT = Path(__file__).resolve().parents[1]
RULES = load_yaml(ROOT / "config" / "risk_rules.yaml")
TIME = "2026-06-03T00:00:00Z"
ENGINE_HASH = "engine-hash-for-operational-kernel-tests"

def doc(subject: str, bsi: float = 20.0, invalid: bool = False) -> dict[str, Any]:
    return {
        "subject_id": subject,
        "critical_data_invalid": invalid,
        "confidence": 1.0 if invalid else 0.95,
        "domain_indices": {"BSI": bsi, "NRI": 20, "VML": 20, "GRS": 80, "CNI": 20},
        "degradations": [],
    }

def req(payload: dict[str, Any], source: str, **extra