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
DEFAULT_MANIFEST=Path("governance/epistemic