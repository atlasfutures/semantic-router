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
capacity = importlib.import_module("rayline_saturation_capacity_contract")
l4 = importlib.import_module("rayline_l4_capacity_contract")
rtx = importlib.import_module("rayline_rtx6000_capacity_contract")

# The standing authority and the conservative position PERF036 closed at:
# `$188.841229466383` plus its full `$5.6186208` envelope, charged whole
# regardless of the `$0.9561` it actually used.
STANDING_AUTHORIZED_USD = 197.459850266383
STANDING_CONSERVATIVE_USD = 194.459850266383


def test_every_granted_packet_needs_exactly_no_further_raise() -> None:
    """The family grants the minimum, so a granted packet's shortfall is zero.

    Each of these three ran under a ceiling raised by exactly its own
    envelope, which is why every one of them lands on the `$3.00` reserve
    floor to the last digit. A non-zero result here would mean some closed
    packet was granted more than it needed, or that the pricing changed under
    a recorded run.
    """

    for contract in (capacity.PERF034, l4.PERF035, rtx.PERF036):
        assert budget.minimum_viable_grant_usd(contract.budget) == 0.0


def test_the_computed_grant_agrees_with_the_receipt_it_does_not_share() -> None:
    """The duplicated envelope arithmetic may not drift from the receipt's.

    `budget_receipt` needs the parts as well as the total, so the envelope is
    computed twice rather than shared. This is the pin that keeps the two
    copies equal, on a granted packet where the receipt is allowed to exist.
    """

    contract = rtx.PERF036.budget
    receipt = budget.budget_receipt(contract)
    envelope = receipt["maximum_resource_envelope_usd"]
    shortfall = (
        contract.previous_conservative_usd
        + envelope
        + contract.required_reserve_usd
        - contract.authorized_cumulative_usd
    )
    assert shortfall == pytest.approx(0.0, abs=1e-9)
    assert budget.minimum_viable_grant_usd(contract) == max(0.0, shortfall)


def test_a_prepared_rtx6000_packet_needs_its_whole_envelope_granted() -> None:
    """Nothing is left to spend, so the raise equals the envelope exactly.

    PERF036 closed on the reserve floor: `$194.459850266383` conservative
    against a `$197.459850266383` ceiling is exactly `$3.00`, all of it
    reserve. A successor on the same card and the same 40-minute wall
    therefore needs its entire `$5.6186208` envelope granted -- there is no
    partial headroom to draw on first.
    """

    prepared = budget.BudgetContract(
        run_id="rayline-prepared-envelope-probe",
        previous_conservative_usd=STANDING_CONSERVATIVE_USD,
        authorized_cumulative_usd=STANDING_AUTHORIZED_USD,
        packet_ceiling_usd=6.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=rtx.PERF036.budget.maximum_paid_wall_seconds,
        maximum_orphan_request_seconds=(
            rtx.PERF036.budget.maximum_orphan_request_seconds
        ),
        maximum_scaledown_seconds=rtx.PERF036.budget.maximum_scaledown_seconds,
        encoder_gpu="RTX-PRO-6000",
    )

    envelope = 5160 * budget.resource_rate_usd_per_second("RTX-PRO-6000")
    assert envelope == pytest.approx(5.6186208)
    assert budget.minimum_viable_grant_usd(prepared) == pytest.approx(5.6186208)

    # Fail-closed until that grant lands: the reserve is negative by the whole
    # envelope, so no receipt exists for this packet at all.
    with pytest.raises(budget.BudgetError):
        budget.budget_receipt(prepared)

    # And granted minimally, the reserve returns to exactly the floor.
    granted = budget.BudgetContract(
        **{
            **prepared.__dict__,
            "authorized_cumulative_usd": STANDING_AUTHORIZED_USD + envelope,
        }
    )
    assert budget.minimum_viable_grant_usd(granted) == 0.0
    receipt = budget.budget_receipt(granted)
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(3.0)


def test_the_standing_position_is_the_one_perf036_closed_at() -> None:
    """The raise is only right if the position it starts from is right."""

    assert rtx.AUTHORIZED_CUMULATIVE_USD == pytest.approx(STANDING_AUTHORIZED_USD)
    assert STANDING_CONSERVATIVE_USD == pytest.approx(
        rtx.PREVIOUS_CONSERVATIVE_USD + rtx.MINIMUM_VIABLE_GRANT_USD
    )
    assert STANDING_AUTHORIZED_USD - STANDING_CONSERVATIVE_USD == pytest.approx(3.0)


def test_an_unpriced_gpu_class_cannot_be_granted_a_raise() -> None:
    """Adding a class is a pricing act, and pricing is where it fails closed."""

    unpriced = budget.BudgetContract(
        run_id="rayline-unpriced-probe",
        previous_conservative_usd=0.0,
        authorized_cumulative_usd=1000.0,
        packet_ceiling_usd=10.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=2400,
        encoder_gpu="B200",
    )
    with pytest.raises(budget.BudgetError):
        budget.minimum_viable_grant_usd(unpriced)
