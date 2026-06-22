# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Tests for the package-boundary ratchet gate.

The ratchet freezes the set of non-geosync top-level packages on the wheel
surface and enforces monotone shrink: a NEW leak fails the build; a re-homed
package leaves a stale ledger that also fails until tightened with --write. It
must pass on the committed baseline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "ci" / "check_package_boundary.py"
)
spec = importlib.util.spec_from_file_location("check_package_boundary", MODULE_PATH)
assert spec is not None and spec.loader is not None
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

_PYPROJECT = """\
[tool.setuptools.packages.find]
include = [{includes}]
"""


def _make_tree(root: Path, packages: list[str]) -> None:
    includes = ", ".join(f'"{p}", "{p}.*"' for p in packages)
    (root / "pyproject.toml").write_text(_PYPROJECT.format(includes=includes))
    for p in packages:
        (root / p).mkdir(parents=True, exist_ok=True)


class PackageBoundaryRatchetTest(unittest.TestCase):
    def test_committed_baseline_holds(self) -> None:
        """The real repo's wheel surface must match its frozen baseline."""
        self.assertEqual(gate.main([]), 0)

    def _run_in(self, packages: list[str], baseline: list[str]) -> int:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_tree(root, packages)
            base_path = root / ".github" / "package_boundary_baseline.json"
            base_path.parent.mkdir(parents=True, exist_ok=True)
            base_path.write_text(
                json.dumps({"version": 1, "non_geosync_packages": baseline})
            )
            orig_root, orig_py, orig_base = gate.ROOT, gate.PYPROJECT, gate.BASELINE_PATH
            gate.ROOT = root
            gate.PYPROJECT = root / "pyproject.toml"
            gate.BASELINE_PATH = base_path
            try:
                return gate.main([])
            finally:
                gate.ROOT, gate.PYPROJECT, gate.BASELINE_PATH = (
                    orig_root,
                    orig_py,
                    orig_base,
                )

    def test_leak_equal_to_baseline_passes(self) -> None:
        rc = self._run_in(["geosync", "core", "tools"], ["core", "tools"])
        self.assertEqual(rc, 0)

    def test_new_leak_fails_closed(self) -> None:
        rc = self._run_in(["geosync", "core", "tools", "evil"], ["core", "tools"])
        self.assertEqual(rc, 1)

    def test_paydown_leaves_stale_ledger_failing(self) -> None:
        # 'tools' re-homed away but baseline still lists it -> must fail until --write
        rc = self._run_in(["geosync", "core"], ["core", "tools"])
        self.assertEqual(rc, 1)

    def test_geosync_subpackages_never_count_as_leak(self) -> None:
        rc = self._run_in(["geosync", "geosync_pro", "geosync_agent"], [])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
