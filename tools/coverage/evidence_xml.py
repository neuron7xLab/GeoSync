# SPDX-License-Identifier: MIT
"""XML evidence readers for the coverage gate."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def read_xml(path: str) -> ET.Element:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise ValueError("missing-or-empty-xml")
    try:
        return ET.parse(p).getroot()
    except ET.ParseError as exc:
        raise ValueError("malformed-xml") from exc
