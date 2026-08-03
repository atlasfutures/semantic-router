# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
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
budget = importlib.import_module("rayline_three_arm_budget")
config_builder = importlib.import_module("rayline_dynamic_arc_config")
contract = importlib.import_module("rayline_dynamic_stop_contract")
launcher = importlib.import_module("rayline_dynamic_stop_launcher")
stager = importlib.import_module("rayline_development_artifact")


def _staged_manifest(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "source"
    output = tmp_path / "staged"
    artifact_fixture.generate(source)
    stager.stage_artifact(
        source,
        output,
        artifact_id="private-development-worker-double-v1",
        created_at="2026-08-03T00:00:00Z",
        exporter_revision="semantic-router@test",
        worker_base_url="http://worker-double:8081/v1",
        api_key_env="RAYLINE_PARITY_WORKER_KEY",
    )
    return json.loads((output / "manifest.json").read_text())


def _rendezvous_owner(raw_episode_id: str, replicas: tuple[str, ...]) -> str:
    episode_hash = hashlib.sha256(raw_episode_id.encode()).hexdigest()
    return max(
        replicas,
        key=lambda replica_id: hashlib.sha256(
            episode_hash.encode() + b"\x00" + replica_id.encode()
        ).digest(),
    )


def test_dyn006_freezes_three_replica_budget_and_closed_authority() -> None:
    receipt = budget.budget_receipt(contract.DYN006.budget)

    assert contract.PATHFINDER_AUTHORIZATION_COMMIT == (
        "06fb91b47f2652ee31e538d860f92947b42e3a6d"
    )
    assert contract.ENCODER_REPLICA_IDS == ("encoder-a", "encoder-b", "encoder-c")
    assert contract.EXPECTED_PRE_BOUNDARY_OWNERS == (2, 3, 3)
    assert contract.EXPECTED_POST_STOP_OWNERS == (0, 4, 4)
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(11.1273264)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(
        84.76783001447986
    )
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        49.54499400552014
    )
    assert contract.LAUNCHABLE_CONTRACT is None
    with pytest.raises(ValueError, match="no Rayline dynamic-stop experiment"):
        contract.resolve_launch_contract(contract.DYN006_RUN_ID)


def test_dyn006_namespace_freezes_balanced_measured_placement() -> None:
    prefix = f"{contract.DYN006_RUN_ID}:r030:{contract.SESSION_NAMESPACE}"
    before = [
        _rendezvous_owner(f"{prefix}:{episode_id}", contract.ENCODER_REPLICA_IDS)
        for episode_id in contract.MEASURED_EPISODE_IDS
    ]
    after = [
        (
            _rendezvous_owner(
                f"{prefix}:{episode_id}", contract.ENCODER_REPLICA_IDS[1:]
            )
            if owner == contract.UNAVAILABLE_REPLICA_ID
            else owner
        )
        for episode_id, owner in zip(contract.MEASURED_EPISODE_IDS, before, strict=True)
    ]

    def vector(owners: list[str]) -> tuple[int, ...]:
        return tuple(
            owners.count(replica_id) for replica_id in contract.ENCODER_REPLICA_IDS
        )

    assert vector(before) == contract.EXPECTED_PRE_BOUNDARY_OWNERS
    assert vector(after) == contract.EXPECTED_POST_STOP_OWNERS
    assert (
        _rendezvous_owner(
            f"{prefix}:{contract.WARMUP_EPISODE_ID}", contract.ENCODER_REPLICA_IDS
        )
        == "encoder-b"
    )


def test_generated_dynamic_config_uses_membership_and_close_contract(
    tmp_path: Path,
) -> None:
    manifest = _staged_manifest(tmp_path)
    template = yaml.safe_load(
        (REPO_ROOT / "deploy/compose/rayline-arc/config.yaml").read_text()
    )

    config = config_builder.build_dynamic_config(
        template,
        manifest,
        artifact_mount_path="/private/runtime",
        encoder_build_id="vllm@test",
        encoder_plugin_version="rayline-arc-io@test",
        worker_endpoint="worker-double:8081",
    )
    arc = config["routing"]["decisions"][0]["algorithm"]["rayline_arc"]

    assert "base_url" not in arc["encoder"]
    assert arc["encoder"]["membership"] == {
        "schema_version": "rayline.arc.encoder-membership.v1",
        "source": "redis",
        "refresh_seconds": 1,
    }
    assert arc["encoder"]["failover"]["max_remaps"] == 1
    assert arc["episode"]["close_header"] == "x-rayline-episode-close"
    assert arc["episode"]["key_prefix"] == config_builder.DYNAMIC_KEY_PREFIX
    assert arc["episode"]["idle_ttl_seconds"] == config_builder.DYNAMIC_IDLE_TTL_SECONDS


def test_unregistered_dyn006_stops_before_side_effects(tmp_path: Path) -> None:
    args = launcher.argparse.Namespace(
        run_id=contract.DYN006_RUN_ID,
        pathfinder_root=tmp_path,
        packet_dir=tmp_path / "packet",
        runtime_dir=tmp_path / "runtime",
        router_image="unused",
    )

    with pytest.raises(ValueError, match="no Rayline dynamic-stop experiment"):
        launcher._preflight(args)

    assert list(tmp_path.iterdir()) == []
