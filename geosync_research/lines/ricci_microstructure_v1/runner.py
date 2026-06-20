"""End-to-end Ricci microstructure research runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .artifact import write_artifact, write_evidence_bundle
from .determinism import git_sha, sha256_path, timestamp_utc
from .ingest import validate_l2_frame
from .synthetic_controls import NULL_MODELS, evaluate_null_gate

DEFAULT_CONFIG: dict[str, Any] = {
    "alpha": 0.5,
    "depth": 5,
    "n_min_edges": 2,
    "n_surrogates": 1000,
    "null_model_type": "iaaft",
}


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config root must be an object")
    merged = {**DEFAULT_CONFIG, **payload}
    if int(merged["n_surrogates"]) < 1000:
        raise ValueError("n_surrogates must be >= 1000")
    if merged["null_model_type"] not in NULL_MODELS:
        raise ValueError(f"unsupported null_model_type: {merged['null_model_type']}")
    return merged


def _read_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"unsupported data file type: {path.suffix}")


def run_pipeline(
    *,
    config_path: Path,
    data_path: Path,
    out_root: Path,
    seed: int,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    depth = int(config["depth"])
    frame = validate_l2_frame(_read_data(data_path), depth=depth)
    run_id = f"ricci_microstructure_v1_{sha256_path(config_path)[:8]}_{sha256_path(data_path)[:8]}_{seed}"
    run_dir = out_root / run_id
    replay_command = (
        "geosync-research run --line ricci_microstructure_v1 "
        f"--config {config_path.as_posix()} --data {data_path.as_posix()} "
        f"--out {out_root.as_posix()} --seed {seed}"
    )
    if dry_run:
        return {"run_id": run_id, "dry_run": True, "replay_command": replay_command}

    stats = evaluate_null_gate(
        frame,
        depth=depth,
        alpha=float(config["alpha"]),
        n_min_edges=int(config["n_min_edges"]),
        n_surrogates=int(config["n_surrogates"]),
        null_model=str(config["null_model_type"]),
        seed=seed,
    )
    data_manifest_path = data_path.parent / "data_manifest.json"
    if not data_manifest_path.is_file():
        raise RuntimeError("data_manifest.json missing; run ingest before run")
    payload = {
        "run_id": run_id,
        "schema_version": "ricci_microstructure_artifact.v1",
        "config_hash": sha256_path(config_path),
        "data_sha256": sha256_path(data_path),
        "git_sha": git_sha(),
        "mean_curvature": float(stats["mean_curvature"]),
        "p_value": float(stats["p_value"]),
        "cliffs_delta": float(stats["cliffs_delta"]),
        "effect_size_class": str(stats["effect_size_class"]),
        "null_model_type": str(config["null_model_type"]),
        "n_surrogates": int(config["n_surrogates"]),
        "falsification_status": str(stats["falsification_status"]),
        "replay_command": replay_command,
        "timestamp_utc": timestamp_utc(),
        "claim_tier": "T3/NOVEL/EXPLORATORY/FALSIFIABLE",
    }
    artifact_path = run_dir / "artifact.json"
    artifact_sha = write_artifact(artifact_path, payload)
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "run_log.json").write_text(
        json.dumps({"artifact_sha256": artifact_sha, "stats": stats}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    evidence_bundle = write_evidence_bundle(
        run_id=run_id,
        artifact_path=artifact_path,
        data_manifest_path=data_manifest_path,
        replay_command=replay_command,
        out_dir=run_dir,
    )
    return {
        "run_id": run_id,
        "artifact": artifact_path.as_posix(),
        "artifact_sha256": artifact_sha,
        "evidence_bundle": evidence_bundle.as_posix(),
        "falsification_status": payload["falsification_status"],
    }
