from pathlib import Path
from pydantic import ValidationError
from BBB_NVU_Cognitive_Noise_Gate_2026.src.deterministic_engine import load_yaml
from BBB_NVU_Cognitive_Noise_Gate_2026.src.operational_kernel import OperationalKernel, verify_operational_envelope
from BBB_NVU_Cognitive_Noise_Gate_2026.tests.requirements import requirement
R = load_yaml(Path(__file__).resolve().parents[1] / "config" / "risk_rules