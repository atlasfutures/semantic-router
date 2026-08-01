# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

artifact_fixture = importlib.import_module("artifact_fixture")
stager = importlib.import_module("rayline_development_artifact")


def _stage(source: Path, output: Path) -> dict[str, object]:
    return stager.stage_artifact(
        source,
        output,
        artifact_id="private-development-worker-double-v1",
        created_at="2026-08-01T00:00:00Z",
        exporter_revision="semantic-router@test",
        worker_base_url="http://worker-double:8080/v1",
        api_key_env="RAYLINE_PARITY_WORKER_KEY",
    )


def test_stage_preserves_numerical_files_and_rewrites_only_dispatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    artifact_fixture.generate(source)
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    source_manifest = json.loads(source_manifest_bytes)

    receipt = _stage(source, output)
    staged = json.loads((output / "manifest.json").read_text())

    assert receipt["worker_count"] == len(source_manifest["workers"])
    assert (
        receipt["source_checkpoint_sha256"]
        == source_manifest["source"]["checkpoint"]["sha256"]
    )
    assert (source / "manifest.json").read_bytes() == source_manifest_bytes
    assert (output / "head.safetensors").read_bytes() == (
        source / "head.safetensors"
    ).read_bytes()
    assert (output / "head_golden.json").read_bytes() == (
        source / "head_golden.json"
    ).read_bytes()
    assert staged["policy"] == source_manifest["policy"]
    assert staged["source"] == source_manifest["source"]
    assert staged["weights"] == source_manifest["weights"]
    assert staged["golden"] == source_manifest["golden"]
    for original, worker in zip(
        source_manifest["workers"], staged["workers"], strict=True
    ):
        assert worker["id"] == original["id"]
        assert worker["model"] == original["model"]
        assert (
            worker["estimated_input_cost_per_token"]
            == original["estimated_input_cost_per_token"]
        )
        assert worker["dispatch_backend"] == "openai_compatible"
        assert worker["provider_base_url"] == "http://worker-double:8080/v1"
        assert worker["openrouter_provider_order"] == []
        assert worker["extra_body"]["chat_template_kwargs"]["enable_thinking"] == (
            worker["thinking_mode"] == "on"
        )


def test_stage_rejects_registered_file_digest_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    artifact_fixture.generate(source)
    (source / "head.safetensors").write_bytes(b"drift")

    with pytest.raises(stager.ArtifactError, match="digest differs"):
        _stage(source, tmp_path / "output")


def test_stage_normalizes_private_runtime_thinking_modes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    artifact_fixture.generate(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["workers"][0]["thinking_mode"] = "disabled"
    manifest["workers"][1]["thinking_mode"] = "high"
    manifest_path.write_text(json.dumps(manifest))

    _stage(source, output)
    staged = json.loads((output / "manifest.json").read_text())

    assert [worker["thinking_mode"] for worker in staged["workers"]] == [
        "off",
        "on",
    ]


def test_stage_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    artifact_fixture.generate(source)
    output.mkdir()

    with pytest.raises(stager.ArtifactError, match="already exists"):
        _stage(source, output)
