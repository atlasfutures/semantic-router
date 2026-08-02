# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

launcher = importlib.import_module("rayline_three_arm_launcher")
budget = importlib.import_module("rayline_three_arm_budget")
contract = importlib.import_module("rayline_three_arm_contract")


def _base() -> dict:
    prices = {
        "estimated_input_cost_per_token": 0.000001,
        "estimated_cache_read_cost_per_token": 0.0000005,
        "estimated_cache_write_cost_per_token": 0.0000015,
        "estimated_output_cost_per_token": 0.000002,
    }
    return {
        "router": {"policy": "mtrouter"},
        "workers": [
            {"id": "worker-a", **prices},
            {"id": "worker-b", **prices},
        ],
    }


def test_budget_preserves_packet_ceiling_and_cleanup_reserve() -> None:
    receipt = budget.budget_receipt(contract.PERF016.budget)

    assert (
        receipt["maximum_resource_envelope_usd"]
        < contract.PERF016.budget.packet_ceiling_usd
    )
    assert (
        receipt["reserve_after_full_envelope_usd"]
        >= contract.PERF016.budget.required_reserve_usd
    )
    assert receipt["provider_spend_usd"] == 0
    assert receipt["maximum_resource_seconds"] > receipt["maximum_paid_wall_seconds"]


def test_perf016_contract_is_exact_and_all_completed_runs_are_closed() -> None:
    receipt = budget.budget_receipt(contract.PERF016.budget)
    expected_resource_seconds = (
        contract.PERF016.budget.maximum_paid_wall_seconds
        + contract.PERF016.budget.maximum_orphan_request_seconds
        + contract.PERF016.budget.maximum_scaledown_seconds
    )

    assert contract.LAUNCHABLE_CONTRACT is None
    assert (
        receipt["maximum_paid_wall_seconds"]
        == contract.PERF016.budget.maximum_paid_wall_seconds
    )
    assert receipt["maximum_resource_seconds"] == expected_resource_seconds
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(6.1280928)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(55.60064962)
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(3.7121744)
    with pytest.raises(ValueError, match="currently launchable"):
        contract.resolve_launch_contract(contract.PERF015_RUN_ID)
    with pytest.raises(ValueError, match="currently launchable"):
        contract.resolve_launch_contract(contract.PERF016_RUN_ID)


def test_pathfinder_config_uses_same_workers_and_protected_encoder(
    tmp_path: Path,
) -> None:
    config = launcher.derive_pathfinder_config(
        _base(),
        checkpoint=tmp_path / "checkpoint.pt",
        decision_log=tmp_path / "decisions.jsonl",
        worker_ids=["worker-a", "worker-b"],
    )

    router = config["router"]
    assert router["mtrouter_encoder_backend"] == "vllm"
    assert router["mtrouter_vllm_base_url"] == contract.IDENTITY.encoder_url
    assert (
        router["mtrouter_vllm_expected_build_id"] == contract.IDENTITY.engine_build_id
    )
    assert router["mtrouter_incremental_encode"] is False
    assert [worker["id"] for worker in config["workers"]] == [
        "worker-a",
        "worker-b",
    ]
    assert all(worker["backend"] == "mock" for worker in config["workers"])
    assert all(worker["api_key_env"] == "" for worker in config["workers"])


def test_pathfinder_worker_order_mismatch_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(launcher.LaunchError, match="frozen topology"):
        launcher.derive_pathfinder_config(
            _base(),
            checkpoint=tmp_path / "checkpoint.pt",
            decision_log=tmp_path / "decisions.jsonl",
            worker_ids=["worker-b", "worker-a"],
        )


def test_launcher_has_no_release_qualification_switch() -> None:
    source = (SCRIPT_DIR / "rayline_three_arm_launcher.py").read_text()

    assert "execute-paid-1000" not in source
    assert '"release_qualification_1000_executed": False' in source
