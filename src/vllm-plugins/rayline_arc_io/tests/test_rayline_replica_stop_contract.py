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
contract = importlib.import_module("rayline_replica_stop_contract")


def test_perf027_freezes_balanced_real_stop_and_budget() -> None:
    assert contract.STOP_ARMS == (
        "arc_dual_staged_control",
        "arc_dual_replica_stop",
    )
    assert contract.UNAVAILABLE_REPLICA == 0
    assert contract.EXPECTED_MEASURED_PRIMARY_SESSIONS == (4, 4)
    assert contract.EXPECTED_ALL_PRIMARY_SESSIONS == (5, 4)
    assert [cell.label for cell in contract.PERF027.cells] == ["r030"]
    receipt = budget.budget_receipt(contract.PERF027.budget)
    assert receipt["previous_conservative_usd"] == pytest.approx(71.9354755968929)
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(7.4182176)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(79.3536931968929)
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(54.9591308231071)


def test_perf027_source_authority_is_closed_after_execution() -> None:
    assert contract.PATHFINDER_AUTHORIZATION_COMMIT == (
        "afb5aa1be2fb9416422ac3adeb5bccefa360e401"
    )
    assert contract.LAUNCHABLE_CONTRACT is None
    with pytest.raises(ValueError, match="no Rayline replica-stop experiment"):
        contract.resolve_launch_contract(contract.PERF027_RUN_ID)
