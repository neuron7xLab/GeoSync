from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "ci" / "emit_pyo3_env.py"
spec = importlib.util.spec_from_file_location("emit_pyo3_env", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class EmitPyo3EnvTests(unittest.TestCase):
    def test_candidate_library_names_strip_shared_library_suffixes(self) -> None:
        self.assertEqual(module._candidate_library_names("libpython3.13.so.1.0")[0], "python3.13")
        self.assertEqual(module._candidate_library_names("libpython3.13.dylib")[0], "python3.13")

    def test_pyo3_env_shell_exports_are_quoted(self) -> None:
        env = module.Pyo3Env(
            pyo3_python="/tmp/python with space/bin/python",
            rustflags="-L native=/tmp/lib path -l python3.13",
            ld_library_path="/tmp/lib path",
        )
        exports = env.shell_exports()
        self.assertIn("export PYO3_PYTHON=", exports)
        self.assertIn("'/tmp/python with space/bin/python'", exports)

    def test_resolve_env_preserves_existing_rustflags(self) -> None:
        env = module.resolve_env(existing_rustflags="-C target-cpu=native")
        self.assertTrue(env.rustflags.startswith("-C target-cpu=native "))
        self.assertIn(" -l ", env.rustflags)

    def test_env_lines_reject_multiline_values(self) -> None:
        env = module.Pyo3Env(
            pyo3_python="/tmp/python",
            rustflags="-L native=/tmp/lib\nINJECTED=1",
            ld_library_path="/tmp/lib",
        )
        with self.assertRaises(ValueError):
            env.lines()


if __name__ == "__main__":
    unittest.main()
