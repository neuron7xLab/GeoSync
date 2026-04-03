# SPDX-License-Identifier: LicenseRef-TradePulse-Proprietary
"""CLI tests for the Kuramoto simulation entrypoint."""

from __future__ import annotations

import json

from click.testing import CliRunner

from core.kuramoto.cli import cli
from core.kuramoto.io import SCHEMA_VERSION


def test_cli_quiet_summary_mode_contract() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["simulate", "--N", "4", "--steps", "5", "--quiet", "--export", "summary", "--seed", "7"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload == {
        "schema_version": SCHEMA_VERSION,
        "summary": payload["summary"],
        "config": payload["config"],
    }


def test_cli_full_export_file_round_trip() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        out = "run.json"
        result = runner.invoke(
            cli,
            ["simulate", "--N", "3", "--steps", "8", "--seed", "3", "--output", out, "--quiet", "--export", "full"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["schema_version"] == SCHEMA_VERSION
        assert len(payload["order_parameter"]) == 9
        assert len(payload["time"]) == 9
        assert len(payload["phases"]) == 9

        with open(out, "r", encoding="utf-8") as handle:
            file_payload = json.load(handle)
        assert file_payload == payload


def test_cli_human_readable_mode_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["simulate", "--N", "3", "--steps", "2", "--seed", "4"])
    assert result.exit_code == 0
    assert "Kuramoto Simulation" in result.output
    assert "Final R" in result.output


def test_cli_adjacency_matrix_file() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("adj.json", "w", encoding="utf-8") as handle:
            json.dump([[0.0, 1.0], [1.0, 0.0]], handle)

        result = runner.invoke(
            cli,
            ["simulate", "--N", "2", "--steps", "5", "--adjacency-file", "adj.json", "--quiet", "--seed", "1"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["summary"]["coupling_mode"] == "adjacency"


def test_cli_edge_list_file() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("edges.json", "w", encoding="utf-8") as handle:
            json.dump({"edges": [{"source": 0, "target": 1, "weight": 0.5}]}, handle)

        result = runner.invoke(
            cli,
            ["simulate", "--N", "2", "--steps", "3", "--edge-list-file", "edges.json", "--quiet"],
        )
        assert result.exit_code == 0


def test_cli_rejects_multiple_topologies() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("adj.json", "w", encoding="utf-8") as handle:
            json.dump([[0.0, 1.0], [1.0, 0.0]], handle)
        with open("edges.json", "w", encoding="utf-8") as handle:
            json.dump({"edges": []}, handle)

        result = runner.invoke(
            cli,
            [
                "simulate",
                "--N",
                "2",
                "--adjacency-file",
                "adj.json",
                "--edge-list-file",
                "edges.json",
            ],
        )
        assert result.exit_code != 0
        assert "Use only one topology source" in result.output


def test_cli_bad_omega_and_theta0_fail() -> None:
    runner = CliRunner()
    omega_result = runner.invoke(cli, ["simulate", "--omega", "1.0,abc"])
    theta_result = runner.invoke(cli, ["simulate", "--theta0", "0.0,nan"])
    assert omega_result.exit_code != 0
    assert theta_result.exit_code != 0
    assert "Failed to parse --omega" in omega_result.output
    assert "contains non-finite" in theta_result.output


def test_cli_adjacency_file_shape_mismatch_fails() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("bad.csv", "w", encoding="utf-8") as handle:
            handle.write("1,2,3\n")
        result = runner.invoke(cli, ["simulate", "--adjacency-file", "bad.csv", "--N", "3", "--quiet"])
        assert result.exit_code != 0
        assert "Adjacency matrix must be 2-dimensional" in result.output


def test_cli_adjacency_file_unsupported_extension_fails() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("adj.bin", "wb") as handle:
            handle.write(b"binary")
        result = runner.invoke(cli, ["simulate", "--adjacency-file", "adj.bin", "--quiet"])
        assert result.exit_code != 0
        assert "Unsupported adjacency file extension" in result.output


def test_cli_edge_list_malformed_schema_and_weight_failures() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        with open("bad_schema.json", "w", encoding="utf-8") as handle:
            json.dump({"not_edges": []}, handle)
        with open("bad_weight.json", "w", encoding="utf-8") as handle:
            json.dump({"edges": [{"source": 0, "target": 1, "weight": "inf"}]}, handle)

        schema_result = runner.invoke(
            cli,
            ["simulate", "--N", "2", "--edge-list-file", "bad_schema.json", "--quiet"],
        )
        weight_result = runner.invoke(
            cli,
            ["simulate", "--N", "2", "--edge-list-file", "bad_weight.json", "--quiet"],
        )
        assert schema_result.exit_code != 0
        assert "must contain an 'edges' array" in schema_result.output
        assert weight_result.exit_code != 0
        assert "non-finite weight" in weight_result.output
