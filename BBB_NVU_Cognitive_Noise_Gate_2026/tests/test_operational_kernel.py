from pathlib import Path
from typing import Any

from pydantic import ValidationError

from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import load_yaml
from BBB_NVU_Cognitive_Noise_Gate_2026.src.operational_kernel import (
    OperationalKernel,
    verify_operational_envelope,
)
from BBB_NV