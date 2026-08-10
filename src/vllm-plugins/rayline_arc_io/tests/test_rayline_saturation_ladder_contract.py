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
open_loop = importlib.import_module("rayline_open_loop_contract")
ladder = importlib.import_module("rayline_saturation_ladder_contract")

PER_ARM_ENVELOPE_USD = 6.9344208


def test_both_arms_reuse_the_frozen_perf021_packet_verbatim() -> None:
    for arm in ladder.SATURATION_LADDER_ARMS:
        assert arm.packet_manifest_sha256 == open_loop.PERF021.packet_manifest_sha256
        assert arm.corpus_sha256 == open_loop.PERF021.corpus_sha256
        assert arm.topology_sha256 == open_loop.PERF021.topology_sha256
        assert arm.cells == open_loop.PERF021.cells
        assert [cell.label for cell in arm.cells] == ["r015", "r030", "r045"]
        assert [cell.offered_rate_rps for cell in arm.cells] == [0.15, 0.30, 0.45]


def test_the_only_variable_between_the_arms_is_the_gdn_backend() -> None:
    control, treatment = ladder.SATURATION_LADDER_ARMS

    # The control must be identity-matched to PERF021, which means the default
    # app and its bare engine build id — not a `-reference-perf031` profile.
    assert control.encoder_app_name == open_loop.PERF021.encoder_app_name
    assert control.encoder_build_id == open_loop.PERF021.encoder_build_id
    assert control.encoder_build_id == ("vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca")
    assert control.encoder_gdn_prefill_backend == "torch_reference"

    assert treatment.encoder_app_name == (
        "rayline-arc-session-encoder-flashinfer-perf031"
    )
    assert treatment.encoder_build_id == (
        "vllm@9f5ea81ca0aa570aea46baf82311a1139c1267ca+gdn-flashinfer-eager"
    )
    assert treatment.encoder_gdn_prefill_backend == "flashinfer"


def test_the_control_pins_the_exact_perf021_knee_it_must_reproduce() -> None:
    assert ladder.CONTROL_FIRST_OVERLOADED_CELL == "r030"
    assert ladder.CONTROL_NOT_OVERLOADED_RATE_RPS == 0.1862
    assert ladder.CONTROL_OVERLOADED_RATE_RPS == 0.3724


def test_the_sequential_two_arm_envelope_fits_the_authorized_ceiling() -> None:
    control = budget.budget_receipt(ladder.PERF031A.budget)
    treatment = budget.budget_receipt(ladder.PERF031B.budget)

    assert control["maximum_resource_seconds"] == open_loop.MAXIMUM_RESOURCE_SECONDS
    assert control["maximum_resource_envelope_usd"] == pytest.approx(
        PER_ARM_ENVELOPE_USD
    )
    assert treatment["maximum_resource_envelope_usd"] == pytest.approx(
        PER_ARM_ENVELOPE_USD
    )
    # Arm 1 charges arm 0's complete envelope first: the arms are sequential
    # runs, so their envelopes accumulate rather than overlap.
    assert ladder.PERF031B.budget.previous_conservative_usd == pytest.approx(
        control["cumulative_if_full_envelope_usd"]
    )
    assert control["cumulative_if_full_envelope_usd"] == pytest.approx(158.684125466383)
    assert treatment["cumulative_if_full_envelope_usd"] == pytest.approx(
        165.618546266383
    )
    assert treatment["reserve_after_full_envelope_usd"] == pytest.approx(8.694277753617)
    for arm in ladder.SATURATION_LADDER_ARMS:
        assert arm.budget.authorized_cumulative_usd == 174.31282402
        assert arm.budget.required_reserve_usd == 3.0
        assert arm.budget.encoder_replicas == 1
    assert treatment["provider_spend_usd"] == 0.0


def test_launch_authority_opens_at_most_one_arm_and_fails_closed() -> None:
    """Binding is deliberate and authorized; the invariant is that it stays narrow.

    Preparation leaves the ladder unbound against a `PENDING` pin that no
    commit can equal. An authorization checkpoint may open exactly one run id
    against a real pushed Pathfinder head, and every other arm must still
    refuse. Both states are legitimate, so this asserts the invariant rather
    than either snapshot -- otherwise binding would require editing the very
    test that guards it.
    """

    bound = ladder.LAUNCHABLE_CONTRACT
    pin = ladder.PATHFINDER_AUTHORIZATION_COMMIT

    if bound is None:
        assert pin == "PENDING"
        for arm in ladder.SATURATION_LADDER_ARMS:
            assert arm.pathfinder_authorization_commit == "PENDING"
            with pytest.raises(ValueError, match="no Rayline saturation ladder"):
                ladder.resolve_launch_contract(arm.run_id)
        return

    assert bound in ladder.SATURATION_LADDER_ARMS
    # A bound ladder may never carry the placeholder: `_assert_pushed` forces
    # the Pathfinder HEAD to equal this, so it must be a real 40-char head.
    assert pin != "PENDING"
    assert len(pin) == 40
    assert set(pin) <= set("0123456789abcdef")
    assert ladder.resolve_launch_contract(bound.run_id) is bound

    for arm in ladder.SATURATION_LADDER_ARMS:
        assert arm.pathfinder_authorization_commit == pin
        if arm is not bound:
            with pytest.raises(ValueError, match="only permits preregistered run id"):
                ladder.resolve_launch_contract(arm.run_id)


def test_each_arm_owns_a_distinct_run_and_resource_namespace() -> None:
    control, treatment = ladder.SATURATION_LADDER_ARMS

    assert control.run_id == "rayline-saturation-ladder-perf031a-20260810"
    assert treatment.run_id == "rayline-saturation-ladder-perf031b-20260810"
    identities = {
        (arm.run_id, arm.compose_project_prefix, arm.temporary_prefix)
        for arm in ladder.SATURATION_LADDER_ARMS
    }
    assert len(identities) == len(ladder.SATURATION_LADDER_ARMS)
    # No PERF031 namespace may collide with the closed PERF020/PERF021 runs.
    closed = {open_loop.PERF020.run_id, open_loop.PERF021.run_id}
    assert not closed & {arm.run_id for arm in ladder.SATURATION_LADDER_ARMS}
