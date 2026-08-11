# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

budget = importlib.import_module("rayline_three_arm_budget")
capacity = importlib.import_module("rayline_saturation_capacity_contract")
knee = importlib.import_module("rayline_saturation_knee_contract")
knee_v2 = importlib.import_module("rayline_saturation_knee_v2_contract")
launcher = importlib.import_module("rayline_open_loop_launcher")
open_loop = importlib.import_module("rayline_open_loop_contract")

PACKET_DIR = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf034"
PERF033_RUN = REPO_ROOT / ".agent-harness/rayline-parity/rayline-saturation-knee-perf033-20260810"
# The session service imports `modal` at module scope, so every assertion
# about it is made against its source text, never by importing it.
SESSION_SERVICE = Path(__file__).resolve().parents[1] / "modal_session_service.py"


def test_perf034_opens_at_most_itself_and_fails_closed() -> None:
    """The packet has three legitimate states; assert what spans all of them.

    Prepared is unbound against the literal `PENDING`, which no commit can
    equal and which `_assert_pushed` compares HEAD against. Open is this one
    run bound against a real pushed Pathfinder head. Closed is unbound again,
    keeping the real pin as the record of what was measured, exactly as every
    closed contract in this packet family does.
    """

    pin = capacity.PATHFINDER_AUTHORIZATION_COMMIT
    # The pin belongs to the packet, never to the bound run.
    assert capacity.PERF034.pathfinder_authorization_commit == pin
    real_head = len(pin) == 40 and set(pin) <= set("0123456789abcdef")
    assert pin == "PENDING" or real_head

    if capacity.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            capacity.resolve_launch_contract(capacity.PERF034_RUN_ID)
        return

    # Bound: this run only, and never against the placeholder.
    assert capacity.LAUNCHABLE_CONTRACT is capacity.PERF034
    assert real_head
    assert (
        capacity.resolve_launch_contract(capacity.PERF034_RUN_ID)
        is capacity.PERF034
    )
    with pytest.raises(ValueError):
        capacity.resolve_launch_contract("rayline-not-a-preregistered-run")


def test_perf034_is_registered_with_the_launcher_it_will_run_under() -> None:
    """A registry the launcher does not consult is a packet that cannot run."""

    assert (
        capacity.resolve_launch_contract
        in launcher._resolve_contract.__globals__.values()
    )
    if capacity.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            launcher._resolve_contract(capacity.PERF034_RUN_ID)
    else:
        assert (
            launcher._resolve_contract(capacity.PERF034_RUN_ID)
            is capacity.PERF034
        )


def test_perf034_budget_fits_exactly_the_confirmed_grant() -> None:
    """The 2026-08-11 grant is the minimum viable amount, and no more.

    The user confirmed the spend without naming a figure, so authorize
    commit e282f16c records the contract's own minimum: the ceiling is the
    old authority plus exactly `MINIMUM_VIABLE_GRANT_USD`, the envelope is
    unchanged from PERF033 because the cap raise moves session-service
    constants rather than container resources, and the reserve after a full
    envelope sits at exactly the `$3.00` floor. Any regression that widens
    the envelope or shrinks the grant turns this back into a raise.
    """

    assert capacity.AUTHORIZED_CUMULATIVE_USD == pytest.approx(
        184.31282402 + capacity.MINIMUM_VIABLE_GRANT_USD
    )

    receipt = budget.budget_receipt(capacity.PERF034.budget)

    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(6.9344208)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(
        186.421808666383
    )
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        3.0, abs=1e-9
    )
    assert (
        receipt["reserve_after_full_envelope_usd"]
        >= capacity.PERF034.budget.required_reserve_usd
    )


def test_perf034_widens_the_corpus_on_the_frozen_topology() -> None:
    """32 lanes is the whole corpus, not a tunable.

    The 128-case directional corpus carries 32 episodes of four decisions
    each and the probe runs one thread per episode, so no packet over this
    corpus can ever be wider. The topology stays PERF020's; the corpus digest
    necessarily differs from the 32-case selection every prior packet ran.
    """

    assert capacity.EPISODE_LANES == 32
    assert capacity.PERF034.measured_cases == 128
    assert capacity.PERF034.warmup_cases == 8
    assert capacity.PERF034.measured_episodes == 32
    assert capacity.PERF034.warmup_episodes == 2
    assert capacity.PERF034.topology_sha256 == open_loop.PERF020.topology_sha256
    assert capacity.PERF034.corpus_sha256 != open_loop.PERF020.corpus_sha256


def test_perf034_anchor_reoffers_perf033s_rate_with_its_own_documents() -> None:
    """The anchor is the same rate but deliberately not the same document.

    A rung's workload document carries the lane count and the case counts,
    so the 32-lane `r120` cannot be byte-identical to PERF033's 8-lane one.
    The anchor property is therefore a measured quantity -- at 1.20 offered,
    an unconstrained encoder must reproduce PERF033's completion throughput
    -- and the digests are pinned to differ so nobody mistakes it for the
    byte-identity anchors of the prior packets.
    """

    anchor = capacity.PERF034.cells[0]
    perf033_anchor = knee_v2.PERF033.cells[0]

    assert anchor.label == capacity.ANCHOR_CELL == perf033_anchor.label
    assert anchor.offered_rate_rps == perf033_anchor.offered_rate_rps == 1.20
    assert anchor.workload_sha256 != perf033_anchor.workload_sha256
    assert anchor.identity_sha256 != perf033_anchor.identity_sha256
    assert capacity.PERF033_ANCHOR_COMPLETION_THROUGHPUT_DPS == pytest.approx(
        1.1547543726851863
    )


def test_perf034_rungs_ascend_to_the_predicted_knee() -> None:
    labels = [cell.label for cell in capacity.PERF034.cells]
    rates = [cell.offered_rate_rps for cell in capacity.PERF034.cells]

    assert labels == ["r120", "r240", "r400", "r645"]
    assert rates == sorted(rates)
    # Realized targets of roughly 1.5 / 3 / 5 / 8 decisions per second at the
    # measured realized-per-offered ratio.
    scale = capacity.ANCHOR_REALIZED_PER_OFFERED
    assert [rate * scale for rate in rates] == pytest.approx(
        [1.48956, 2.97912, 4.9652, 8.006385]
    )


def test_perf034_arms_both_firing_points() -> None:
    criterion = capacity.PERF034.saturation

    assert isinstance(criterion, open_loop.SaturationCriterion)
    assert criterion.episode_lanes == capacity.EPISODE_LANES == 32
    assert criterion.occupancy_ratio == 1.0
    assert criterion.throughput_plateau_gain == 1 / 3
    # The closed runs keep their frozen report shapes: PERF033 stays v3.
    assert knee_v2.PERF033.saturation.throughput_plateau_gain is None
    assert open_loop.PERF020.saturation is None
    assert open_loop.PERF021.saturation is None
    assert knee.PERF032.saturation is None


def test_perf034_owns_its_cap_raised_encoder_app() -> None:
    """Same engine build, distinct app: the caps are part of the deployment.

    The closed runs' evidence names the perf031 app, whose session-service
    caps must stay 8/32 forever. The cap raise therefore lives in a distinct
    app name that the session service maps to the same flashinfer profile.
    """

    assert capacity.PERF034.encoder_app_name == capacity.PERF034_APP_NAME
    assert capacity.PERF034_APP_NAME != knee.FLASHINFER_APP_NAME
    assert capacity.PERF034.encoder_build_id == knee.FLASHINFER_BUILD_ID
    assert capacity.PERF034.encoder_build_id.endswith("+gdn-flashinfer-eager")
    assert capacity.PERF034.encoder_gdn_prefill_backend == "flashinfer"
    assert capacity.PERF034.encoder_gpu == "H100"

    source = SESSION_SERVICE.read_text()
    assert f'"{capacity.PERF034_APP_NAME}": "flashinfer"' in source
    assert "32 if APP_NAME in PERF034_APP_PROFILES else 8" in source
    assert "64 if APP_NAME in PERF034_APP_PROFILES else 32" in source


def test_perf034_trace_prefix_is_the_recorded_perf020_trace() -> None:
    """The continuity constant is a recorded value, and this pins it to the record.

    Every 32-case closed run wrote the same `selected_worker_trace_sha256`.
    The 128-case corpus routes a superset, so PERF034's check is a prefix
    property against this digest -- which must therefore equal what the
    closed receipts actually recorded, not a plausible-looking string.
    """

    receipt_path = PERF033_RUN / "r120" / "rayline_arc.json"
    if not receipt_path.is_file():
        pytest.skip(f"{PERF033_RUN.name} receipts are not present")
    recorded = json.loads(receipt_path.read_text())

    assert (
        recorded["results"]["selected_worker_trace_sha256"]
        == capacity.PERF020_TRACE_PREFIX_SHA256
    )


def test_perf034_digests_match_the_generated_packet() -> None:
    """The contract's digests are the packet's, not plausible-looking strings."""

    if not PACKET_DIR.is_dir():
        pytest.skip("packet-perf034 is not present")
    from rayline_three_arm_launcher import _sha256

    assert (
        _sha256(PACKET_DIR / "manifest.json")
        == capacity.PERF034.packet_manifest_sha256
    )
    assert _sha256(PACKET_DIR / "corpus.json") == capacity.PERF034.corpus_sha256
    assert (
        _sha256(PACKET_DIR / "topology.json") == capacity.PERF034.topology_sha256
    )
    for cell in capacity.PERF034.cells:
        cell_dir = PACKET_DIR / "cells" / cell.label
        assert _sha256(cell_dir / "workload.json") == cell.workload_sha256
        assert _sha256(cell_dir / "identity.json") == cell.identity_sha256
