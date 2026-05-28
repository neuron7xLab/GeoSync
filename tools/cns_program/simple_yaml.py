from __future__ import annotations

from pathlib import Path
from typing import Any


class SimpleYAMLError(ValueError):
    pass


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"\'')


def load_yaml_text(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_at_indent: dict[int, str] = {}
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            item = _parse_scalar(line[2:])
            if not isinstance(parent, list):
                raise SimpleYAMLError(f"list item without list parent: {raw}")
            parent.append(item)
            continue
        if ":" not in line:
            raise SimpleYAMLError(f"invalid line: {raw}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue
        next_container: Any = {}
        parent[key] = next_container
        last_key_at_indent[indent] = key
        stack.append((indent, next_container))
    # second pass to convert known empty dict sequence parents into lists
    return _normalize_sequences(root)


def _normalize_sequences(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _normalize_sequences(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_normalize_sequences(v) for v in node]
    return node


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    # This intentionally supports the small subset used by CNS configs via PyYAML when present,
    # and falls back to a tiny parser in dependency-minimal CI.
    try:
        import yaml  # type: ignore
    except Exception:
        return load_yaml_text(Path(path).read_text(encoding="utf-8"))
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}
