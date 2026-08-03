# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

budget = importlib.import_module("rayline_three_arm_budget")
contract = importlib.import_module("rayline_failover_contract")


def test_perf025_freezes_one_r030_cell_and_bounded_two_replica_budget() -> None:
    assert contract.FAILOVER_ARMS == (
        "arc_dual_sticky",
        "arc_dual_forced_failover",
    )
    assert contract.FAILOVER_AFTER_POOLING == contract.TURNS_PER_EPISODE // 2
    assert [cell.label for cell in contract.PERF025.cells] == ["r030"]
    assert [cell.offered_rate_rps for cell in contract.PERF025.cells] == [0.30]
    receipt = budget.budget_receipt(contract.PERF025.budget)
    assert receipt["encoder_replicas"] == len(contract.ENCODER_APP_NAMES)
    assert receipt["maximum_resource_seconds"] == contract.MAXIMUM_RESOURCE_SECONDS
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(7.4182176)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(
        75.761437231823516
    )
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        58.551386788176484
    )

    successor_receipt = budget.budget_receipt(contract.PERF026.budget)
    assert [cell.label for cell in contract.PERF026.cells] == ["r030"]
    assert successor_receipt["previous_conservative_usd"] == pytest.approx(
        70.1005119398672
    )
    assert successor_receipt["cumulative_if_full_envelope_usd"] == pytest.approx(
        77.5187295398672
    )
    assert successor_receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        56.7940944801328
    )


def test_all_failover_launch_authority_is_closed() -> None:
    assert contract.PATHFINDER_AUTHORIZATION_COMMIT == (
        "c7aaca5bdfcee0c398569b1019e5fd8985461b84"
    )
    assert contract.LAUNCHABLE_CONTRACT is None
    for run_id in (contract.PERF025_RUN_ID, contract.PERF026_RUN_ID):
        with pytest.raises(ValueError, match="no Rayline failover experiment"):
            contract.resolve_launch_contract(run_id)
