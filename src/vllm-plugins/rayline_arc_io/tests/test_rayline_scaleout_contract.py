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
contract = importlib.import_module("rayline_scaleout_contract")


def test_perf022_freezes_overloaded_cells_two_apps_and_budget() -> None:
    assert contract.SCALEOUT_ARMS == ("arc_single", "arc_dual_affinity")
    assert contract.ENCODER_APP_NAMES == (
        "rayline-arc-session-encoder-a",
        "rayline-arc-session-encoder-b",
    )
    assert [cell.label for cell in contract.PERF022.cells] == ["r030", "r045"]
    assert [cell.offered_rate_rps for cell in contract.PERF022.cells] == [0.30, 0.45]
    receipt = budget.budget_receipt(contract.PERF022.budget)
    assert receipt["encoder_replicas"] == len(contract.ENCODER_APP_NAMES)
    assert receipt["maximum_resource_seconds"] == contract.MAXIMUM_RESOURCE_SECONDS
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(13.8688416)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(
        75.67812892218463
    )
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        58.63469509781537
    )


def test_only_perf022_launch_authority_is_open() -> None:
    assert contract.PATHFINDER_AUTHORIZATION_COMMIT == (
        "24b4a3d6a548e5b96589432a5f5d32f572575165"
    )
    assert contract.resolve_launch_contract(contract.PERF022_RUN_ID) is contract.PERF022
    with pytest.raises(ValueError, match="launcher only permits preregistered"):
        contract.resolve_launch_contract("unregistered-run")
