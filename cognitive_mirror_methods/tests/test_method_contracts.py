import json
from pathlib import Path


def test_method_schema_has_required_contract_fields() -> None:
    schema = json.loads(Path("cognitive_mirror_methods/schemas/method.schema.json").read_text())
    required = set(schema["required"])
    assert {"id", "definition", "input", "process", "output", "validation", "failure_modes", "example"} <= required


def test_prompt_registry_has_versioned_prompt() -> None:
    text = Path("cognitive_mirror_methods/prompts/registry.yaml").read_text()
    assert "version:" in text
    assert "finalizer_mirror" in text
