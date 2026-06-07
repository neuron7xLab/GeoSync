# mypy: ignore-errors
# ruff: noqa
# fmt: off
# SPDX-License-Identifier: MIT
"""Fail-closed validator for the GeoSync Epistemic Rhombus manifest."""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from typing import Any, Iterator
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
DEFAULT_MANIFEST=Path("governance/epistemic_rhombus_manifest.jsonl")
DEFAULT_SCHEMA=Path("schemas/governance/epistemic_rhombus_manifest.schema.json")
WORKFLOW_ROOT=Path(".github/workflows")
WORKFLOW_SUFFIXES={".yml",".yaml"}
FORBIDDEN_PR_TARGET_TRIGGER="pull_request"+"_target"
class ManifestReadError(ValueError): pass
def iter_jsonl(path:Path)->Iterator[tuple[int,dict[str,Any]]]:
    with path.open("r",encoding="utf-8") as s:
        for n,line in enumerate(s,1):
            if not line.strip(): continue
            try: v=json