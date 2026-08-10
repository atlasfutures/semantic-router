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
knee = importlib.import_module("rayline_saturation_knee_contract")
knee_v2 = importlib.import_module("rayline_saturation_knee_v2_contract")
launcher = importlib.import_module("rayline_open_loop_launcher")
open_loop = importlib.import_module("rayline_open_loop_contract")

PACKET_DIR = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf033"


def test_perf033_is_not_launchable_and_pins_no_authorization_head() -> None:
    """Preparation may open neither gate.

    `LAUNCHABLE_CONTRACT` refuses the arm outright, and the Pathfinder pin is
    the literal `PENDING`, which no commit can equal and which `_assert_pushed`
    compares HEAD against. Either alone is sufficient.
    """

    assert knee_v2.LAUNCHABLE_CONTRACT is None
    assert knee_v2.PATHFINDER_AUTHORIZATION_COMMIT == "PENDING"
    assert knee_v2.PERF033.pathfinder_authorization_commit == "PENDING"
    with pytest.raises(ValueError):
        knee_v2.resolve_launch_contract(knee_v2.PERF033_RUN_ID)


def test_perf033_is_registered_with_the_launcher_it_will_run_under() -> None:
    """A registry the launcher does not consult is a packet that cannot run.

    PERF032 failed preflight the first time for exactly this reason, so the
    registration is pinned rather than assumed. The resolver still fails closed
    here, because nothing in any registry is launchable.
    """

    assert (
        knee_v2.resolve_launch_contract
        in launcher._resolve_contract.__globals__.values()
    )
    with pytest.raises(ValueError):
        launcher._resolve_contract(knee_v2.PERF033_RUN_ID)


def test_perf033_budget_fits_the_existing_authority() -> None:
    """PERF033 needs no fresh grant, and the arithmetic says so rather than a comment."""

    receipt = budget.budget_receipt(knee_v2.PERF033.budget)

    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(6.9344208)
    assert receipt["cumulative_if_full_envelope_usd"] == pytest.approx(
        179.487387866383
    )
    assert receipt["reserve_after_full_envelope_usd"] == pytest.approx(
        4.825436153617, abs=1e-9
    )
    assert (
        receipt["reserve_after_full_envelope_usd"]
        >= knee_v2.PERF033.budget.required_reserve_usd
    )


def test_perf033_anchor_rung_is_perf032s_top_rung_verbatim() -> None:
    """The anchor is the same document, not merely the same rate.

    A rung's workload and identity derive only from the offered rate, the seed
    and the frozen constants, so re-offering `1.20` reproduces PERF032's `r120`
    byte for byte. If it did not, the anchor would compare two different
    workloads and no higher rung would be interpretable.
    """

    anchor = knee_v2.PERF033.cells[0]
    perf032_top = knee.PERF032.cells[-1]

    assert anchor.label == knee_v2.ANCHOR_CELL == perf032_top.label
    assert anchor.offered_rate_rps == perf032_top.offered_rate_rps
    assert anchor.workload_sha256 == perf032_top.workload_sha256
    assert anchor.identity_sha256 == perf032_top.identity_sha256


def test_perf033_rungs_ascend_past_the_measured_bound() -> None:
    labels = [cell.label for cell in knee_v2.PERF033.cells]
    rates = [cell.offered_rate_rps for cell in knee_v2.PERF033.cells]

    assert labels == ["r120", "r160", "r220", "r320"]
    assert rates == sorted(rates)
    # PERF032's top rung realized `1.4896` arrivals and peaked at seven of
    # eight lanes. Every successor rung offers more than that.
    scale = knee_v2.ANCHOR_REALIZED_ARRIVAL_RATE_RPS / knee_v2.ANCHOR_OFFERED_RATE_RPS
    assert [rate * scale for rate in rates[1:]] == pytest.approx(
        [1.9861374603643505, 2.730939008000982, 3.972274920728701]
    )


def test_perf033_carries_the_saturation_criterion_the_comparator_will_apply() -> None:
    criterion = knee_v2.PERF033.saturation

    assert isinstance(criterion, open_loop.SaturationCriterion)
    assert criterion.episode_lanes == knee_v2.EPISODE_LANES == 8
    assert criterion.occupancy_ratio == 1.0
    # The closed runs keep the frozen report shape.
    assert open_loop.PERF020.saturation is None
    assert open_loop.PERF021.saturation is None
    assert knee.PERF032.saturation is None


def test_perf033_reuses_the_measured_flashinfer_engine() -> None:
    assert knee_v2.PERF033.encoder_app_name == knee.FLASHINFER_APP_NAME
    assert knee_v2.PERF033.encoder_build_id == knee.FLASHINFER_BUILD_ID
    assert knee_v2.PERF033.encoder_build_id.endswith("+gdn-flashinfer-eager")
    assert knee_v2.PERF033.encoder_gdn_prefill_backend == "flashinfer"
    assert knee_v2.PERF033.encoder_gpu == "H100"


def test_perf033_digests_match_the_generated_packet() -> None:
    """The contract's digests are the packet's, not plausible-looking strings."""

    if not PACKET_DIR.is_dir():
        pytest.skip("packet-perf033 is not present")
    from rayline_three_arm_launcher import _sha256

    assert (
        _sha256(PACKET_DIR / "manifest.json")
        == knee_v2.PERF033.packet_manifest_sha256
    )
    assert _sha256(PACKET_DIR / "corpus.json") == knee_v2.PERF033.corpus_sha256
    assert _sha256(PACKET_DIR / "topology.json") == knee_v2.PERF033.topology_sha256
    for cell in knee_v2.PERF033.cells:
        cell_dir = PACKET_DIR / "cells" / cell.label
        assert _sha256(cell_dir / "workload.json") == cell.workload_sha256
        assert _sha256(cell_dir / "identity.json") == cell.identity_sha256
