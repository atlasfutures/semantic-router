# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

artifact_fixture = importlib.import_module("artifact_fixture")
stager = importlib.import_module("rayline_development_artifact")
config_builder = importlib.import_module("rayline_parity_arc_config")


def _staged_manifest(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source"
    output = tmp_path / "staged"
    artifact_fixture.generate(source)
    stager.stage_artifact(
        source,
        output,
        artifact_id="private-development-worker-double-v1",
        created_at="2026-08-01T00:00:00Z",
        exporter_revision="semantic-router@test",
        worker_base_url="http://worker-double:8081/v1",
        api_key_env="RAYLINE_PARITY_WORKER_KEY",
    )
    return json.loads((output / "manifest.json").read_text())


def test_generated_config_matches_every_artifact_worker(tmp_path: Path) -> None:
    manifest = _staged_manifest(tmp_path)
    template = yaml.safe_load(
        (REPO_ROOT / "deploy/compose/rayline-arc/config.yaml").read_text()
    )

    config = config_builder.build_config(
        template,
        manifest,
        artifact_mount_path="/private/runtime",
        encoder_base_url="https://encoder.example.invalid",
        encoder_build_id="vllm@test",
        encoder_plugin_version="rayline-arc-io@test",
        worker_endpoint="worker-double:8081",
    )

    workers = manifest["workers"]
    models = config["providers"]["models"]
    decision = config["routing"]["decisions"][0]
    assert [model["name"] for model in models] == [worker["id"] for worker in workers]
    assert [reference["model"] for reference in decision["modelRefs"]] == [
        worker["id"] for worker in workers
    ]
    assert [reference["use_reasoning"] for reference in decision["modelRefs"]] == [
        worker["thinking_mode"] == "on" for worker in workers
    ]
    for model, worker in zip(models, workers, strict=True):
        assert model["provider_model_id"] == worker["model"]
        assert model["backend_refs"][0]["base_url"] == worker["provider_base_url"]
        assert model["pricing"]["prompt_per_1m"] == pytest.approx(
            worker["estimated_input_cost_per_token"] * 1_000_000
        )
    arc = decision["algorithm"]["rayline_arc"]
    assert arc["artifact_revision"] == manifest["artifact_id"]
    assert arc["encoder"]["serving_rung"] == "B"
    assert arc["episode"]["backend"] == "redis"


def test_config_rejects_unstaged_openrouter_runtime(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifact_fixture.generate(source)
    manifest = json.loads((source / "manifest.json").read_text())
    template = yaml.safe_load(
        (REPO_ROOT / "deploy/compose/rayline-arc/config.yaml").read_text()
    )

    with pytest.raises(config_builder.ConfigError, match="not staged"):
        config_builder.build_config(
            template,
            manifest,
            artifact_mount_path="/private/runtime",
            encoder_base_url="https://encoder.example.invalid",
            encoder_build_id="vllm@test",
            encoder_plugin_version="rayline-arc-io@test",
            worker_endpoint="worker-double:8081",
        )
