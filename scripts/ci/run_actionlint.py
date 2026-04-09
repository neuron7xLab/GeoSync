#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run actionlint with local binary first, docker fallback, soft-skip if unavailable."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def main() -> int:
    actionlint = shutil.which("actionlint")
    if actionlint:
        return _run([actionlint, "-color"])

    docker = shutil.which("docker")
    if docker:
        return _run(
            [
                docker,
                "run",
                "--rm",
                "-v",
                f"{ROOT}:/repo",
                "-w",
                "/repo",
                "rhysd/actionlint:1.7.8",
                "-color",
            ]
        )

    print("SKIP: actionlint unavailable (no binary and no docker).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
