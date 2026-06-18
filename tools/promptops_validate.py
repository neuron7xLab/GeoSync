#!/usr/bin/env python3
"""Fail-closed PromptOps artifact validator.

This is intentionally small, deterministic, and CI-friendly. It validates the
first vertical slice of a prompt-as-code contract without making LLM calls.
Shadow inference remains a recorded contract here; later PRs can bind it to a
real runner behind an explicit secret-gated job.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:  # YAML support is optional locally, installed in CI.
    import yaml  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None

ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{2,127}$")
REF_RE = re.compile(r"<ref:([A-Za-z0-9_.-]+)>")
SEMVER_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
URL_RE = re.compile(r"https?://[^\s)\]>'\"]+")
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SECRET_PATTERNS = {
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "generic_token": re.compile(r"(?i)(api[_-]?key|token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"),
}
INJECTION_PATTERNS = {
    "ignore_previous": re.compile(r"(?i)ignore\s+(all\s+)?previous\s+instructions?"),
    "reveal_prompt": re.compile(r"(?i)reveal\s+(the\s+)?(system\s+)?prompt"),
    "developer_mode": re.compile(r"(?i)you\s+are\s+now\s+(in\s+)?developer\s+mode"),
}
ALLOWED_ROLES = {"system", "developer", "user", "assistant", "tool"}


@dataclass
class ValidationIssue:
    path: str
    code: str
    message: str


@dataclass
class ValidationReport:
    artifact: str
    ok: bool = True
    issues: list[ValidationIssue] = field(default_factory=list)

    def fail(self, code: str, message: str, path: str = "$." ) -> None:
        self.ok = False
        self.issues.append(ValidationIssue(path=path, code=code, message=message))

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "ok": self.ok,
            "issues": [issue.__dict__ for issue in self.issues],
        }


def load_artifact(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to validate YAML artifacts")
        return yaml.safe_load(text)
    raise RuntimeError(f"Unsupported artifact extension: {path.suffix}")


def valid_url(raw: str) -> bool:
    parsed = urlparse(raw)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def all_text_nodes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(all_text_nodes(item))
        return out
    if isinstance(value, dict):
        out = []
        for item in value.values():
            out.extend(all_text_nodes(item))
        return out
    return []


def require_keys(report: ValidationReport, data: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in data:
            report.fail("missing_required_key", f"Missing required key: {key}", f"$.{key}")


def validate_artifact(path: Path) -> ValidationReport:
    report = ValidationReport(artifact=str(path))
    try:
        data = load_artifact(path)
    except Exception as exc:
        report.fail("parse_error", str(exc))
        return report

    if not isinstance(data, dict):
        report.fail("invalid_root", "Artifact root must be an object")
        return report

    require_keys(report, data, ("artifact_type", "id", "version", "messages", "unit_tests", "shadow_tests"))
    if not report.ok:
        return report

    if data.get("artifact_type") != "system_prompt_contract":
        report.fail("invalid_artifact_type", "artifact_type must be system_prompt_contract", "$.artifact_type")

    artifact_id = data.get("id")
    if not isinstance(artifact_id, str) or not ID_RE.fullmatch(artifact_id):
        report.fail("invalid_id", "id must match ^[A-Za-z][A-Za-z0-9_.-]{2,127}$", "$.id")

    version = data.get("version")
    if not isinstance(version, str) or not SEMVER_RE.fullmatch(version):
        report.fail("invalid_version", "version must be strict semver: MAJOR.MINOR.PATCH", "$.version")

    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        report.fail("invalid_messages", "messages must be a non-empty array", "$.messages")
        messages = []

    roles: list[str] = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            report.fail("invalid_message", "message must be an object", f"$.messages[{i}]")
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in ALLOWED_ROLES:
            report.fail("invalid_role", f"role must be one of {sorted(ALLOWED_ROLES)}", f"$.messages[{i}].role")
        else:
            roles.append(role)
        if not isinstance(content, str) or not content.strip():
            report.fail("empty_content", "message.content must be a non-empty string", f"$.messages[{i}].content")

    if "system" not in roles and "developer" not in roles:
        report.fail("missing_authoritative_instruction", "messages must include at least one system or developer instruction", "$.messages")

    references = data.get("references", [])
    declared_refs: set[str] = set()
    if references is not None:
        if not isinstance(references, list):
            report.fail("invalid_references", "references must be an array", "$.references")
        else:
            for i, ref in enumerate(references):
                if not isinstance(ref, dict):
                    report.fail("invalid_reference", "reference must be an object", f"$.references[{i}]")
                    continue
                ref_id, href = ref.get("id"), ref.get("href")
                if not isinstance(ref_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", ref_id):
                    report.fail("invalid_reference_id", "reference id is invalid", f"$.references[{i}].id")
                else:
                    declared_refs.add(ref_id)
                if not isinstance(href, str) or not valid_url(href):
                    report.fail("invalid_reference_href", "reference href must be http(s) URI", f"$.references[{i}].href")

    text_nodes = all_text_nodes(data)
    mentioned_refs = {match.group(1) for text in text_nodes for match in REF_RE.finditer(text)}
    missing_refs = mentioned_refs - declared_refs
    if missing_refs:
        report.fail("missing_reference_declaration", f"Referenced IDs are not declared: {sorted(missing_refs)}", "$.references")

    urls = set()
    for text in text_nodes:
        urls.update(URL_RE.findall(text))
        urls.update(match.group(1) for match in MARKDOWN_LINK_RE.finditer(text))
    for raw in sorted(urls):
        if raw.startswith("mailto:"):
            continue
        if not valid_url(raw):
            report.fail("invalid_url", f"Invalid URL: {raw}", "$.text")

    for label, pattern in SECRET_PATTERNS.items():
        for text in text_nodes:
            if pattern.search(text):
                report.fail("secret_like_value", f"Potential secret detected: {label}", "$.text")
                break

    for label, pattern in INJECTION_PATTERNS.items():
        for text in text_nodes:
            if pattern.search(text):
                report.fail("prompt_injection_phrase", f"Potential prompt-injection phrase detected: {label}", "$.text")
                break

    for collection_name in ("unit_tests", "shadow_tests"):
        collection = data.get(collection_name)
        if not isinstance(collection, list) or not collection:
            report.fail("invalid_test_collection", f"{collection_name} must be a non-empty array", f"$.{collection_name}")
            continue
        seen: set[str] = set()
        for i, case in enumerate(collection):
            if not isinstance(case, dict):
                report.fail("invalid_test_case", "test case must be an object", f"$.{collection_name}[{i}]")
                continue
            case_id = case.get("id")
            if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
                report.fail("invalid_test_id", "test id is invalid", f"$.{collection_name}[{i}].id")
                continue
            if case_id in seen:
                report.fail("duplicate_test_id", f"Duplicate test id: {case_id}", f"$.{collection_name}[{i}].id")
            seen.add(case_id)
            required = ("input", "assertions") if collection_name == "unit_tests" else ("input", "baseline_expectations", "quality_gates")
            for key in required:
                if key not in case:
                    report.fail("missing_test_key", f"Missing {key} in {collection_name} case", f"$.{collection_name}[{i}].{key}")

    return report


def discover(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in {".json", ".yaml", ".yml"}))
        elif path.exists():
            out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate PromptOps system prompt contracts")
    parser.add_argument("paths", nargs="+", help="Artifact files or directories")
    parser.add_argument("--report", default="", help="Optional JSON report path")
    args = parser.parse_args(argv)

    artifacts = discover(args.paths)
    if not artifacts:
        print("PROMPTOPS_VALIDATION: FAIL no artifacts found", file=sys.stderr)
        return 2

    reports = [validate_artifact(path) for path in artifacts]
    payload = {"ok": all(r.ok for r in reports), "reports": [r.as_dict() for r in reports]}
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
