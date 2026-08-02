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


def test_only_perf025_launch_authority_is_open() -> None:
    assert contract.PATHFINDER_AUTHORIZATION_COMMIT == (
        "c1b080f4a12127985745ff22480d206fc40dd9da"
    )
    assert contract.resolve_launch_contract(contract.PERF025_RUN_ID) is contract.PERF025
    with pytest.raises(ValueError, match="launcher only permits preregistered"):
        contract.resolve_launch_contract("different-run")
