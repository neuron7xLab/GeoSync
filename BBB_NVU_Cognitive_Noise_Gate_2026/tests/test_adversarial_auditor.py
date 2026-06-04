# ruff: noqa: I001

from BBB_NVU_Cognitive_Noise_Gate_2026.scripts.adversarial_auditor import AdversarialAuditor
from BBB_NVU_Cognitive_Noise_Gate_2026.tests.requirements import requirement


@requirement("R002")
def test_adversarial_sandbox_campaign_has_zero_bypasses():
    vectors = AdversarialAuditor().run_campaign(iterations=60)
    assert vectors
    assert all(
        vector["risk_state"] != "GREEN_STABLE"
        for vector in vectors
        if vector["input"].get("degradations")
    )
    assert all(
        vector["risk_state"] == "BLACK_INVALID"
        for vector in vectors
        if vector["input"].get("critical_data_invalid")
        or any(str(value) == "nan" for value in vector["input"].get("domain_indices", {}).values())
    )
