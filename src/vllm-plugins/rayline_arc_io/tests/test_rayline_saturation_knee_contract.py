# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

budget = importlib.import_module("rayline_three_arm_budget")
open_loop = importlib.import_module("rayline_open_loop_contract")
ladder = importlib.import_module("rayline_saturation_ladder_contract")
knee = importlib.import_module("rayline_saturation_knee_contract")

ENVELOPE_USD = 6.9344208
PERF031_CLOSING_POSITION_USD = 165.618546266383


def test_the_packet_is_a_single_flashinfer_arm() -> None:
    """No `torch_reference` arm: the control's knee is already recorded.

    PERF031A reproduced PERF021's `r030` knee exactly, so re-running the
    control above `r045` would measure nothing new while doubling the spend.
    PERF032's comparison is against PERF031's closed numbers.
    """

    assert knee.SATURATION_KNEE_ARMS == (knee.PERF032,)
    assert knee.PERF032.encoder_gdn_prefill_backend == "flashinfer"
    assert knee.PERF032.encoder_app_name == (
        "rayline-arc-session-encoder-flashinfer-perf031"
    )
    assert knee.PERF032.encoder_build_id == (
        "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca+gdn-flashinfer-eager"
    )
    # Reusing PERF031B's app is what makes no new profile and no allowlist
    # change necessary, and keeps the engine identity byte-identical to the
    # unsaturated run PERF032 extends.
    assert knee.PERF032.encoder_app_name == ladder.PERF031B.encoder_app_name
    assert knee.PERF032.encoder_build_id == ladder.PERF031B.encoder_build_id


def test_the_ladder_is_exactly_the_four_preregistered_rungs() -> None:
    labels = [cell.label for cell in knee.PERF032.cells]
    rates = [cell.offered_rate_rps for cell in knee.PERF032.cells]

    assert labels == ["r045", "r060", "r090", "r120"]
    assert rates == [0.45, 0.60, 0.90, 1.20]
    assert rates == sorted(rates)


def test_r045_is_the_same_cell_perf031b_ran_unsaturated() -> None:
    """The anchor is byte-identical to PERF021's top rung, not merely equal-rate.

    If PERF032 does not reproduce an unsaturated `r045`, the packet is
    measuring a different system and `r060`/`r090`/`r120` are uninterpretable.
    That check is only meaningful if the cell itself is the same document.
    """

    anchor = knee.PERF032.cells[0]
    perf021_top = open_loop.PERF021.cells[-1]

    assert anchor.label == knee.ANCHOR_CELL == perf021_top.label
    assert anchor.offered_rate_rps == knee.ANCHOR_OFFERED_RATE_RPS
    assert anchor.offered_rate_rps == perf021_top.offered_rate_rps
    assert anchor.workload_sha256 == perf021_top.workload_sha256
    assert anchor.identity_sha256 == perf021_top.identity_sha256
    # PERF031B's measured result at that rung, which the anchor must repeat.
    assert knee.ANCHOR_REALIZED_ARRIVAL_RATE_RPS == 0.5586011607274736
    assert knee.ANCHOR_COMPLETION_THROUGHPUT_DPS == 0.5518306368768308
    assert knee.ANCHOR_COMPLETION_THROUGHPUT_DPS < knee.ANCHOR_REALIZED_ARRIVAL_RATE_RPS


def test_the_predicted_knee_sits_between_r090_and_r120() -> None:
    """The capacity model is preregistered as falsifiable, not as an outcome."""

    offered = {cell.label: cell.offered_rate_rps for cell in knee.PERF032.cells}
    assert knee.PREDICTED_KNEE_DECISIONS_PER_SECOND == 1.143
    assert offered["r090"] < knee.PREDICTED_KNEE_DECISIONS_PER_SECOND < offered["r120"]
    # And the whole point of the new packet: the prediction is above every rung
    # PERF031B could offer, which is why that run could not test it.
    assert knee.PREDICTED_KNEE_DECISIONS_PER_SECOND > max(
        cell.offered_rate_rps for cell in ladder.PERF031B.cells
    )


def test_the_workload_is_unchanged_and_only_the_rungs_are_new() -> None:
    assert knee.PERF032.corpus_sha256 == open_loop.PERF020.corpus_sha256
    assert knee.PERF032.topology_sha256 == open_loop.PERF020.topology_sha256
    # A new rung set is a new packet, so the manifest digest must move.
    assert knee.PERF032.packet_manifest_sha256 == (
        "eeb1c69f57ae964b238c7763ff87abf2dc727ba94b757c45e24aa2e013b08fed"
    )
    assert (
        knee.PERF032.packet_manifest_sha256 != open_loop.PERF020.packet_manifest_sha256
    )
    for cell in knee.PERF032.cells:
        assert len(cell.workload_sha256) == 64
        assert len(cell.identity_sha256) == 64
        assert cell.concurrency == 8
    digests = {cell.workload_sha256 for cell in knee.PERF032.cells}
    assert len(digests) == len(knee.PERF032.cells)


def test_the_budget_fails_closed_until_a_human_raises_the_ceiling() -> None:
    """Preparation may size the envelope; it may not grant the money."""

    assert knee.PERF032.budget.previous_conservative_usd == pytest.approx(
        PERF031_CLOSING_POSITION_USD
    )
    assert knee.PERF032.budget.authorized_cumulative_usd == (
        knee.AUTHORIZED_CUMULATIVE_USD
    )
    # A ceiling only ever moves up, and only a human moves it. Assert that
    # invariant rather than the ungranted snapshot: this guard must keep
    # working after a grant, or it fails the moment it is needed least and
    # forces an edit to the very test that protects the money.
    assert knee.AUTHORIZED_CUMULATIVE_USD >= (
        ladder.PERF031B.budget.authorized_cumulative_usd
    )
    if knee.AUTHORIZED_CUMULATIVE_USD < (
        PERF031_CLOSING_POSITION_USD
        + ENVELOPE_USD
        + knee.PERF032.budget.required_reserve_usd
    ):
        # Ungranted, or granted too little: it must refuse, never proceed.
        with pytest.raises(budget.BudgetError, match=knee.PERF032_RUN_ID):
            budget.budget_receipt(knee.PERF032.budget)
    else:
        # Granted: the floor must still be intact after a full envelope.
        live = budget.budget_receipt(knee.PERF032.budget)
        assert live["reserve_after_full_envelope_usd"] >= (
            knee.PERF032.budget.required_reserve_usd
        )

    # `MINIMUM_VIABLE_GRANT_USD` was computed from PERF031's closing ceiling,
    # so it must be checked against that ceiling and not against whatever the
    # ceiling is today. Anchoring it to the live value would make the stated
    # minimum look wrong the moment a real grant lands.
    granted = dataclasses.replace(
        knee.PERF032.budget,
        authorized_cumulative_usd=(
            ladder.PERF031B.budget.authorized_cumulative_usd
            + knee.MINIMUM_VIABLE_GRANT_USD
        ),
    )
    receipt = budget.budget_receipt(granted)

    assert receipt["maximum_resource_seconds"] == open_loop.MAXIMUM_RESOURCE_SECONDS
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(ENVELOPE_USD)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(172.552967066383)
    # The stated minimum grant is exactly minimal: it lands on the floor.
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        knee.PERF032.budget.required_reserve_usd
    )
    assert knee.PERF032.budget.required_reserve_usd == 3.0
    assert knee.PERF032.budget.encoder_replicas == 1
    assert receipt["provider_spend_usd"] == 0.0


def test_launch_authority_opens_at_most_this_run_and_fails_closed() -> None:
    """The packet has three legitimate states; assert what spans all of them.

    Prepared is unbound against the `PENDING` pin that no commit can equal.
    Open is this one run bound against a real pushed Pathfinder head. Closed
    is unbound again, keeping the real pin as the record of what was measured,
    exactly as the closed PERF020/PERF021 and PERF031 contracts do. Asserting
    the prepared snapshot would break this guard on the day it is opened.
    """

    pin = knee.PATHFINDER_AUTHORIZATION_COMMIT
    # The pin is a property of the packet, never of the bound run.
    assert knee.PERF032.pathfinder_authorization_commit == pin
    real_head = len(pin) == 40 and set(pin) <= set("0123456789abcdef")
    assert pin == "PENDING" or real_head

    if knee.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError, match="no Rayline saturation knee"):
            knee.resolve_launch_contract(knee.PERF032.run_id)
        return

    # Bound: exactly this run, and never against the placeholder, because
    # `_assert_pushed` forces the Pathfinder HEAD to equal the pin.
    assert knee.LAUNCHABLE_CONTRACT is knee.PERF032
    assert real_head
    assert knee.resolve_launch_contract(knee.PERF032.run_id) is knee.PERF032
    with pytest.raises(ValueError, match="only permits preregistered run id"):
        knee.resolve_launch_contract("rayline-not-a-preregistered-run")


def test_the_run_owns_a_namespace_no_closed_run_can_collide_with() -> None:
    assert knee.PERF032.run_id == "rayline-saturation-knee-perf032-20260810"
    assert knee.PERF032.compose_project_prefix == "rayline-saturation-knee-perf032"
    assert knee.PERF032.temporary_prefix == "rayline-perf032-"

    closed = (
        open_loop.PERF020,
        open_loop.PERF021,
        ladder.PERF031A,
        ladder.PERF031B,
    )
    for prior in closed:
        assert knee.PERF032.run_id != prior.run_id
        assert knee.PERF032.compose_project_prefix != prior.compose_project_prefix
        assert knee.PERF032.temporary_prefix != prior.temporary_prefix
        assert not knee.PERF032.compose_project_prefix.startswith(
            prior.compose_project_prefix
        )
        assert not prior.compose_project_prefix.startswith(
            knee.PERF032.compose_project_prefix
        )
