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
contract = importlib.import_module("rayline_concurrency_contract")


def test_concurrency_packet_and_cell_contract_is_exact() -> None:
    expected_measured_cases = 32
    expected_warmup_cases = 4
    assert [cell.concurrency for cell in contract.PERF017.cells] == [1, 4, 8]
    assert [cell.profile for cell in contract.PERF017.cells] == [
        "sweep-32-c1",
        "sweep-32-c4",
        "sweep-32-c8",
    ]
    assert expected_measured_cases == contract.MEASURED_CASES
    assert expected_warmup_cases == contract.WARMUP_CASES
    assert contract.SWEEP_ARMS == ("rayline_remote", "rayline_arc")
    assert contract.PERF018.cells == contract.PERF017.cells
    assert contract.PERF019.cells == contract.PERF017.cells


def test_perf019_budget_is_preregistered_but_needs_authority() -> None:
    with pytest.raises(budget.BudgetError, match="exceeds budget authority"):
        budget.budget_receipt(contract.PERF019.budget)
    resource_seconds = (
        contract.PERF019.budget.maximum_paid_wall_seconds
        + contract.PERF019.budget.maximum_orphan_request_seconds
        + contract.PERF019.budget.maximum_scaledown_seconds
    )
    envelope = resource_seconds * budget.resource_rate_usd_per_second()
    required_authority = (
        contract.PERF019.budget.previous_conservative_usd
        + envelope
        + contract.PERF019.budget.required_reserve_usd
    )
    cumulative = contract.PERF019.budget.previous_conservative_usd + envelope
    reserve = contract.PERF019.budget.authorized_cumulative_usd - cumulative

    assert envelope == pytest.approx(5.3217648)
    assert required_authority == pytest.approx(
        contract.PERF019_REQUIRED_CUMULATIVE_AUTHORITY_USD
    )
    assert cumulative == pytest.approx(61.79459254)
    assert reserve == pytest.approx(2.51823148)
    assert pytest.approx(5.0) == contract.ADDITIONAL_AUTHORITY_GRANTED_USD
    assert pytest.approx(0.48176852) == contract.ADDITIONAL_AUTHORITY_REQUIRED_USD
    with pytest.raises(ValueError, match="no Rayline concurrency sweep"):
        contract.resolve_launch_contract(contract.PERF019_RUN_ID)
