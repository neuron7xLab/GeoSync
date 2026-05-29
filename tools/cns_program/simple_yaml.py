from __future__ import annotations

from pathlib import Path
from typing import Any


class SimpleYAMLError(ValueError):
    """Raised when CNS YAML loading cannot continue."""


def load_yaml_text(text: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise SimpleYAMLError("PyYAML is required for CNS YAML loading") from exc
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SimpleYAMLError("top-level YAML document must be a mapping")
    return data


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    return load_yaml_text(Path(path).read_text(encoding="utf-8"))
