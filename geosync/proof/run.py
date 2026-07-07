# Copyright (c) 2023-2026 Yaroslav Vasylenko (neuron7xLab)
# SPDX-License-Identifier: MIT
"""Five-minute GeoSync proof pipeline: D -> H -> T -> F -> V.

Stdlib-only: tests one explicit hypothesis on one deterministic fixture and
writes one verdict artifact.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "geosync" / "proof" / "fixtures" / "market_fixture.csv"
ARTIFACT = ROOT / "artifacts" / "geosync_proof" / "verdict.json"
THRESHOLD = 0.60
TRAIN_RETURNS = 8
MECHANISM = "lag1_directional_persistence_fixture_test"


@dataclass(frozen=True)
class MarketBar:
    timestamp: str
    close: float


@dataclass(frozen=True)
class Hypothesis:
    rule: str
    train_hit_rate: float
    train_samples: int
    text: str


@dataclass(frozen=True)
class TestResult:
    hit_rate: float | None
    hits: int
    samples: int


def _relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_timestamp() -> str:
    raw = os.environ.get("SOURCE_DATE_EPOCH", "1700000000")
    try:
        epoch = int(raw)
    except ValueError:
        epoch = 1700000000
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _code_version() -> str:
    override = os.environ.get("GEOSYNC_PROOF_CODE_VERSION")
    if override:
        return override
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    version = proc.stdout.strip()
    return version if proc.returncode == 0 and version else "unknown"


def load_market_data(path: Path = FIXTURE) -> list[MarketBar]:
    rows: list[MarketBar] = []
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != ["timestamp", "close"]:
            raise ValueError("fixture schema must be timestamp,close")
        for row in reader:
            timestamp = str(row.get("timestamp", "")).strip()
            close_raw = str(row.get("close", "")).strip()
            if not timestamp or not close_raw:
                raise ValueError("fixture row has empty timestamp or close")
            close = float(close_raw)
            if close <= 0:
                raise ValueError("close must be positive")
            rows.append(MarketBar(timestamp=timestamp, close=close))
    if len(rows) < TRAIN_RETURNS + 4:
        raise ValueError("fixture must contain enough bars for train and test windows")
    return rows


def return_signs(bars: Iterable[MarketBar]) -> list[int]:
    values = [bar.close for bar in bars]
    signs: list[int] = []
    for prev, cur in zip(values, values[1:]):
        diff = cur - prev
        if diff > 0:
            signs.append(1)
        elif diff < 0:
            signs.append(-1)
        else:
            signs.append(0)
    if any(s == 0 for s in signs):
        raise ValueError("zero-return bars are unsupported in this proof fixture")
    return signs


def _score_rule(signs: list[int], start: int, stop: int, rule: str) -> TestResult:
    hits = 0
    samples = 0
    for idx in range(start, stop):
        prev = signs[idx - 1]
        actual = signs[idx]
        predicted = prev if rule == "continuation" else -prev
        hits += int(predicted == actual)
        samples += 1
    return TestResult(hit_rate=(hits / samples if samples else None), hits=hits, samples=samples)


def generate_hypothesis(signs: list[int]) -> Hypothesis:
    if len(signs) <= TRAIN_RETURNS:
        raise ValueError("not enough returns to generate hypothesis")
    cont = _score_rule(signs, 1, TRAIN_RETURNS, "continuation")
    rev = _score_rule(signs, 1, TRAIN_RETURNS, "reversal")
    if cont.hit_rate is None or rev.hit_rate is None:
        raise ValueError("training window produced no samples")
    rule = "continuation" if cont.hit_rate >= rev.hit_rate else "reversal"
    chosen = cont if rule == "continuation" else rev
    text = (
        "On the training slice of the fixture, the prior return sign is hypothesized "
        f"to predict the next return sign by {rule}."
    )
    return Hypothesis(rule=rule, train_hit_rate=chosen.hit_rate, train_samples=chosen.samples, text=text)


def run_test(signs: list[int], hypothesis: Hypothesis) -> TestResult:
    return _score_rule(signs, TRAIN_RETURNS, len(signs), hypothesis.rule)


def build_verdict(dataset: Path = FIXTURE) -> dict[str, object]:
    bars = load_market_data(dataset)
    signs = return_signs(bars)
    hypothesis = generate_hypothesis(signs)
    result = run_test(signs, hypothesis)
    if result.hit_rate is None or result.samples < 3:
        verdict = "INCONCLUSIVE"
    elif result.hit_rate >= THRESHOLD:
        verdict = "ACCEPT"
    else:
        verdict = "REJECT"
    return {
        "mechanism": MECHANISM,
        "input_dataset": _relpath(dataset),
        "hypothesis": hypothesis.text,
        "metric": "holdout_hit_rate",
        "threshold": f">= {THRESHOLD:.2f}",
        "result": {
            "holdout_hit_rate": result.hit_rate,
            "hits": result.hits,
            "samples": result.samples,
            "training_rule": hypothesis.rule,
            "training_hit_rate": hypothesis.train_hit_rate,
            "training_samples": hypothesis.train_samples,
        },
        "verdict": verdict,
        "failure_condition": "Reject if holdout_hit_rate is below 0.60; inconclusive if fewer than 3 holdout samples.",
        "repro_command": "python -m geosync.proof.run",
        "timestamp_utc": _utc_timestamp(),
        "code_version": _code_version(),
    }


def write_verdict(verdict: dict[str, object], path: Path = ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(verdict, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> int:
    verdict = build_verdict()
    write_verdict(verdict)
    result = verdict["result"]
    if not isinstance(result, dict):
        raise TypeError("internal result must be a dict")
    line = (
        "GEOSYNC_PROOF "
        f"verdict={verdict['verdict']} "
        f"mechanism={verdict['mechanism']} "
        f"metric={verdict['metric']} "
        f"value={result['holdout_hit_rate']} "
        f"threshold='{verdict['threshold']}' "
        f"artifact={_relpath(ARTIFACT)}\n"
    )
    sys.stdout.write(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
