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
contract = importlib.import_module("rayline_open_loop_contract")


def test_perf020_contract_freezes_rates_and_budget() -> None:
    assert [cell.label for cell in contract.PERF020.cells] == [
        "r015",
        "r030",
        "r045",
    ]
    assert [cell.offered_rate_rps for cell in contract.PERF020.cells] == [
        0.15,
        0.30,
        0.45,
    ]
    receipt = budget.budget_receipt(contract.PERF020.budget)
    assert receipt["maximum_resource_seconds"] == contract.MAXIMUM_RESOURCE_SECONDS
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(6.9344208)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(
        65.03408153955587
    )
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        19.27874248044413
    )


def test_perf020_launch_authority_is_closed_after_execution() -> None:
    with pytest.raises(ValueError, match="no Rayline open-loop sweep"):
        contract.resolve_launch_contract(contract.PERF020_RUN_ID)
