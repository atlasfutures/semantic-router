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
l4 = importlib.import_module("rayline_l4_capacity_contract")
capacity = importlib.import_module("rayline_saturation_capacity_contract")
knee = importlib.import_module("rayline_saturation_knee_contract")
knee_v2 = importlib.import_module("rayline_saturation_knee_v2_contract")
launcher = importlib.import_module("rayline_open_loop_launcher")
open_loop = importlib.import_module("rayline_open_loop_contract")

PACKET_DIR = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf035"
PERF033_RUN = REPO_ROOT / ".agent-harness/rayline-parity/rayline-saturation-knee-perf033-20260810"
# The session service imports `modal` at module scope, so every assertion
# about it is made against its source text, never by importing it.
SESSION_SERVICE = Path(__file__).resolve().parents[1] / "modal_session_service.py"


def test_perf035_opens_at_most_itself_and_fails_closed() -> None:
    """Prepared means unbound against a literal no commit can equal.

    `PENDING` is what `_assert_pushed` compares the Pathfinder HEAD against,
    so it cannot pass; the launchable contract is separately `None`. Both gates
    must move for a launch to exist, and preparation may move neither.
    """

    pin = l4.PATHFINDER_AUTHORIZATION_COMMIT
    # The pin belongs to the packet, never to the bound run.
    assert l4.PERF035.pathfinder_authorization_commit == pin
    real_head = len(pin) == 40 and set(pin) <= set("0123456789abcdef")
    assert pin == "PENDING" or real_head

    if l4.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            l4.resolve_launch_contract(l4.PERF035_RUN_ID)
        return

    # Bound: this run only, and never against the placeholder.
    assert l4.LAUNCHABLE_CONTRACT is l4.PERF035
    assert real_head
    assert l4.resolve_launch_contract(l4.PERF035_RUN_ID) is l4.PERF035
    with pytest.raises(ValueError):
        l4.resolve_launch_contract("rayline-not-a-preregistered-run")


def test_perf035_is_registered_with_the_launcher_it_will_run_under() -> None:
    """A registry the launcher does not consult is a packet that cannot run."""

    assert l4.resolve_launch_contract in launcher._resolve_contract.__globals__.values()
    if l4.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            launcher._resolve_contract(l4.PERF035_RUN_ID)
    else:
        assert launcher._resolve_contract(l4.PERF035_RUN_ID) is l4.PERF035


def test_perf035_budget_is_priced_on_l4_seconds_and_granted_minimally() -> None:
    """The envelope is real, and the authority for it moved by exactly that.

    PERF034's grant left the reserve at exactly the `$3.00` floor, so the
    minimum viable grant equals the envelope. The 2026-08-11 authorization
    moved the ceiling by precisely that amount -- not by the `$10` bound the
    approval named -- so the reserve after a full envelope lands on the floor
    again, exactly.
    """

    contract = l4.PERF035.budget
    assert contract.encoder_gpu == "L4"

    rate = budget.resource_rate_usd_per_second("L4")
    # Modal on-demand L4 seconds plus the unchanged 8-core / 64 GiB container.
    assert rate == pytest.approx(0.000222 + 8 * 0.0000131 + 64 * 0.00000222)
    envelope = 5160 * rate
    assert envelope == pytest.approx(2.4194208)

    # Granted minimally: the receipt exists and the reserve after a full
    # envelope is exactly the floor, to the cent and beyond.
    receipt = budget.budget_receipt(contract)
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(2.4194208)
    reserve = contract.authorized_cumulative_usd - (
        contract.previous_conservative_usd + envelope
    )
    assert reserve == pytest.approx(contract.required_reserve_usd)

    assert l4.MINIMUM_VIABLE_GRANT_USD == pytest.approx(envelope)
    assert l4.AUTHORIZED_CUMULATIVE_USD == pytest.approx(
        capacity.AUTHORIZED_CUMULATIVE_USD + l4.MINIMUM_VIABLE_GRANT_USD
    )
    assert l4.PREVIOUS_CONSERVATIVE_USD == pytest.approx(186.421808666383)
    # The ceiling is a real bound on an L4 packet, not one inherited from a
    # card that costs five times as much.
    assert contract.packet_ceiling_usd == 3.0
    assert envelope < contract.packet_ceiling_usd


def test_perf035_leaves_every_h100_packet_priced_exactly_as_recorded() -> None:
    """Adding a GPU class may not reprice a single closed run."""

    assert budget.resource_rate_usd_per_second() == pytest.approx(0.00134388)
    assert budget.resource_rate_usd_per_second("H100") == pytest.approx(0.00134388)
    receipt = budget.budget_receipt(capacity.PERF034.budget)
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(6.9344208)
    assert receipt["pricing_snapshot"] == budget.PRICING_SNAPSHOT
    assert budget.PRICING_SNAPSHOT == "modal-on-demand-2026-07-31-h100-cpu-memory"
    # The L4 packet records a different snapshot, so a receipt can never claim
    # H100 pricing for an L4 run.
    assert budget.PRICING_SNAPSHOTS["L4"] != budget.PRICING_SNAPSHOT
    with pytest.raises(budget.BudgetError):
        budget.resource_rate_usd_per_second("A100")


def test_perf035_moves_only_the_card() -> None:
    """PERF033's shape exactly, on PERF020's corpus and topology.

    The cross-GPU ratio is only readable if nothing else moved, so the corpus
    and topology digests are asserted byte-identical to the ones every
    eight-lane packet since PERF020 has carried.
    """

    assert l4.EPISODE_LANES == 8
    assert l4.PERF035.measured_cases == 32
    assert l4.PERF035.warmup_cases == 4
    assert l4.PERF035.measured_episodes == 8
    assert l4.PERF035.warmup_episodes == 1
    assert l4.PERF035.corpus_sha256 == open_loop.PERF020.corpus_sha256
    assert l4.PERF035.topology_sha256 == open_loop.PERF020.topology_sha256
    assert l4.PERF035.corpus_sha256 == knee_v2.PERF033.corpus_sha256
    assert l4.PERF035.encoder_build_id == knee.FLASHINFER_BUILD_ID
    assert l4.PERF035.encoder_gdn_prefill_backend == "flashinfer"


def test_perf035_predicts_its_ceiling_from_a_model_perf033_already_validated() -> None:
    """The prediction is not TFLOPS hand-waving, and the test says why.

    Run the same token model on the H100 and it reproduces PERF033's
    independently recorded GPU-busy fraction of `0.53` to within 1.5%. If a
    regression breaks the derivation, this fails here rather than as a wrong
    L4 number nobody can check.
    """

    assert l4.PREDICTED_L4_TOKENS_PER_SECOND == pytest.approx(14163.24, abs=0.01)
    assert l4.PREDICTED_CEILING_DPS == pytest.approx(0.4014, abs=1e-4)

    h100_busy_seconds = l4.CORPUS_ENCODER_TOKENS / l4.H100_FLASHINFER_TOKENS_PER_SECOND
    perf033_top_wall = l4.MEASURED_CASES / l4.PERF033_TOP_COMPLETION_THROUGHPUT_DPS
    derived_fraction = h100_busy_seconds / perf033_top_wall
    assert derived_fraction == pytest.approx(
        l4.PERF033_RECORDED_GPU_BUSY_FRACTION, rel=0.02
    )

    # The band is preregistered because PERF034 had none and so could not say
    # whether its 7.15% anchor miss counted as a hit. It is a validation
    # target, not an integrity gate.
    assert l4.PREDICTED_CEILING_TOLERANCE == 0.30
    low = l4.PREDICTED_CEILING_DPS * (1 - l4.PREDICTED_CEILING_TOLERANCE)
    high = l4.PREDICTED_CEILING_DPS * (1 + l4.PREDICTED_CEILING_TOLERANCE)
    assert (low, high) == pytest.approx((0.2810, 0.5218), abs=1e-4)


def test_perf035_rungs_bracket_the_predicted_ceiling() -> None:
    labels = [cell.label for cell in l4.PERF035.cells]
    rates = [cell.offered_rate_rps for cell in l4.PERF035.cells]

    assert labels == ["r016", "r032", "r048", "r072"]
    assert labels[0] == l4.ANCHOR_CELL
    assert rates == sorted(rates)
    assert len(set(rates)) == len(rates)

    realized = [rate * l4.ANCHOR_REALIZED_PER_OFFERED for rate in rates]
    multiples = [value / l4.PREDICTED_CEILING_DPS for value in realized]
    # Anchor at half the prediction, top rung well past even its optimistic
    # edge, so the plateau has somewhere to appear.
    assert multiples == pytest.approx([0.4948, 0.9896, 1.4844, 2.2266], abs=1e-3)
    assert multiples[0] < 0.6
    band_high = l4.PREDICTED_CEILING_DPS * (1 + l4.PREDICTED_CEILING_TOLERANCE)
    assert realized[-1] > band_high

    # Every rung's schedule plus its completion time fits the paid wall, at the
    # prediction and at both edges of the band -- and still fits if the card
    # lands at half the prediction.
    cases = l4.MEASURED_CASES + l4.WARMUP_CASES
    for ceiling in (
        l4.PREDICTED_CEILING_DPS,
        l4.PREDICTED_CEILING_DPS * (1 - l4.PREDICTED_CEILING_TOLERANCE),
        l4.PREDICTED_CEILING_DPS / 2,
    ):
        both_arms = 2 * sum(max(cases / rate, cases / ceiling) + 16 for rate in rates)
        assert both_arms < l4.PERF035.budget.maximum_paid_wall_seconds


def test_perf035_arms_both_firing_points() -> None:
    criterion = l4.PERF035.saturation

    assert isinstance(criterion, open_loop.SaturationCriterion)
    assert criterion.episode_lanes == l4.EPISODE_LANES == 8
    assert criterion.occupancy_ratio == 1.0
    assert criterion.throughput_plateau_gain == 1 / 3
    # Same calibration PERF034 armed, on the same comparator.
    assert (
        criterion.throughput_plateau_gain
        == capacity.PERF034.saturation.throughput_plateau_gain
    )
    # The closed runs keep their frozen report shapes: PERF033 stays v3.
    assert knee_v2.PERF033.saturation.throughput_plateau_gain is None


def test_perf035_owns_its_l4_encoder_app() -> None:
    """The GPU class is part of the deployment, so it needs its own app name."""

    assert l4.PERF035.encoder_app_name == l4.PERF035_APP_NAME
    assert l4.PERF035.encoder_gpu == "L4"
    assert l4.PERF035_APP_NAME not in {
        knee.FLASHINFER_APP_NAME,
        capacity.PERF034_APP_NAME,
    }
    # Every other packet's evidence must keep claiming H100.
    assert capacity.PERF034.encoder_gpu == "H100"
    assert knee_v2.PERF033.encoder_gpu == "H100"
    assert open_loop.PERF020.encoder_gpu == "H100"

    source = SESSION_SERVICE.read_text()
    assert f'"{l4.PERF035_APP_NAME}": "flashinfer"' in source
    # The routing became a three-way when PERF036 added the RTX PRO 6000, but
    # the PERF035 branch is unchanged: this app name, and only this one,
    # deploys on an L4.
    assert "if APP_NAME in PERF035_APP_PROFILES:\n    GPU_TYPE = \"L4\"\n" in source
    # A 24 GB card cannot hold the 32-lane corpus, so the cap raise must not
    # follow the L4 app.
    assert "MAX_SESSIONS = 32 if APP_NAME in PERF034_APP_PROFILES else 8" in source


def test_perf035_memory_is_safe_only_by_corpus_construction() -> None:
    """The derived cap does not fit this card, and the contract says so.

    Eight lanes at `MAX_SERIALIZED_TOKENS` is 24 GiB of KV -- more than the
    whole L4. The packet is admissible only because the frozen corpus peaks far
    below that, which is exactly the bound this test pins.
    """

    kib_per_token = 12
    max_serialized_tokens = 262_144
    worst_case_gib = (
        l4.EPISODE_LANES * max_serialized_tokens * kib_per_token / 1024 / 1024
    )
    assert worst_case_gib == pytest.approx(24.0)

    peak_gib = l4.CORPUS_ENCODER_TOKENS * kib_per_token / 1024 / 1024
    assert peak_gib == pytest.approx(12.92, abs=0.01)
    # A ~19 GiB pool: 24 GB at 0.92 utilization, less ~1.6 GB of weights.
    assert peak_gib < 19.0
    assert worst_case_gib > 19.0


def test_perf035_corpus_encoder_tokens_are_the_corpus_they_claim() -> None:
    """The prediction rests on this number, so read it off the packet.

    Retained pooling prefills each turn's delta exactly once, so an episode's
    total encoder work is its final serialized length. The same arithmetic on
    the 128-case corpus reproduces the 4,261,735 PERF034 recorded.
    """

    corpus_path = PACKET_DIR / "corpus.json"
    if not corpus_path.is_file():
        pytest.skip("packet-perf035 is not present")
    corpus = json.loads(corpus_path.read_text())

    episodes: dict[str, int] = {}
    for case in corpus["measured"]:
        episode = case["episode_id"]
        episodes[episode] = max(episodes.get(episode, 0), case["input_tokens"])

    assert len(episodes) == l4.MEASURED_EPISODES
    assert sum(episodes.values()) == l4.CORPUS_ENCODER_TOKENS


def test_perf035_trace_digest_is_the_recorded_perf020_trace() -> None:
    """PERF035 runs the same 32 cases, so continuity is exact, not a prefix.

    PERF034 needed a prefix property because its corpus routed a superset.
    This packet routes PERF020's corpus itself, so the recorded worker trace
    must match outright -- a stronger check, and the reason the constant is
    pinned to what the closed receipts actually wrote.
    """

    receipt_path = PERF033_RUN / "r120" / "rayline_arc.json"
    if not receipt_path.is_file():
        pytest.skip(f"{PERF033_RUN.name} receipts are not present")
    recorded = json.loads(receipt_path.read_text())

    assert (
        recorded["results"]["selected_worker_trace_sha256"] == l4.PERF020_TRACE_SHA256
    )
    assert l4.PERF020_TRACE_SHA256 == capacity.PERF020_TRACE_PREFIX_SHA256


def test_perf035_digests_match_the_generated_packet() -> None:
    """The contract's digests are the packet's, not plausible-looking strings."""

    if not PACKET_DIR.is_dir():
        pytest.skip("packet-perf035 is not present")
    from rayline_three_arm_launcher import _sha256

    assert _sha256(PACKET_DIR / "manifest.json") == l4.PERF035.packet_manifest_sha256
    assert _sha256(PACKET_DIR / "corpus.json") == l4.PERF035.corpus_sha256
    assert _sha256(PACKET_DIR / "topology.json") == l4.PERF035.topology_sha256
    for cell in l4.PERF035.cells:
        cell_dir = PACKET_DIR / "cells" / cell.label
        assert _sha256(cell_dir / "workload.json") == cell.workload_sha256
        assert _sha256(cell_dir / "identity.json") == cell.identity_sha256
        workload = json.loads((cell_dir / "workload.json").read_text())
        assert workload["offered_rate_rps"] == cell.offered_rate_rps
        assert workload["max_episode_lanes"] == l4.EPISODE_LANES
