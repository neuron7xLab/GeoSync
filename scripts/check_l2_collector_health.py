#!/usr/bin/env python3
"""Long-running L2 collector health check — CLI over longrun_monitor.

Intended invocation: cron every ~60 s, or systemd watchdog sidecar.

Exit codes:
    0 — HEALTHY
    1 — DEGRADED
    2 — STALE
    3 — UNREACHABLE (log missing or no flush ever)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from research.microstructure.longrun_monitor import (
    parse_log_tail,
    read_log_tail,
)

_EXIT = {"HEALTHY": 0, "DEGRADED": 1, "STALE": 2, "UNREACHABLE": 3}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/collector_longrun.log"),
        help="Path to collector log file",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=256 * 1024,
        help="Max bytes to read from tail of log (default 256 KiB)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional append-only JSONL history path",
    )
    args = parser.parse_args()

    text = read_log_tail(Path(args.log), max_bytes=int(args.max_bytes))
    now = datetime.now(timezone.utc)
    report = parse_log_tail(text, now_utc=now)
    payload = asdict(report)
    payload["log_path"] = str(args.log)

    print(json.dumps(payload, indent=2, sort_keys=True, default=str))

    if args.output is not None:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as fh:
            fh.write(
                json.dumps({**payload, "checked_at_utc": now.isoformat(timespec="seconds")}) + "\n"
            )

    return _EXIT.get(report.verdict, 3)


if __name__ == "__main__":
    sys.exit(main())
