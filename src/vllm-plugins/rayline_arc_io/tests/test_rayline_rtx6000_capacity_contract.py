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
rtx = importlib.import_module("rayline_rtx6000_capacity_contract")
capacity = importlib.import_module("rayline_saturation_capacity_contract")
knee = importlib.import_module("rayline_saturation_knee_contract")
knee_v2 = importlib.import_module("rayline_saturation_knee_v2_contract")
launcher = importlib.import_module("rayline_open_loop_launcher")
open_loop = importlib.import_module("rayline_open_loop_contract")

PACKET_DIR = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf036"
PERF033_RUN = (
    REPO_ROOT / ".agent-harness/rayline-parity/rayline-saturation-knee-perf033-20260810"
)
# The session service imports `modal` at module scope, so every assertion
# about it is made against its source text, never by importing it.
SESSION_SERVICE = Path(__file__).resolve().parents[1] / "modal_session_service.py"


def test_perf036_opens_at_most_itself_and_fails_closed() -> None:
    """Prepared means unbound against a literal no commit can equal.

    `PENDING` is what `_assert_pushed` compares the Pathfinder HEAD against,
    so it cannot pass; the launchable contract is separately `None`. Both gates
    must move for a launch to exist, and preparation may move neither.
    """

    pin = rtx.PATHFINDER_AUTHORIZATION_COMMIT
    # The pin belongs to the packet, never to the bound run.
    assert rtx.PERF036.pathfinder_authorization_commit == pin
    real_head = len(pin) == 40 and set(pin) <= set("0123456789abcdef")
    assert pin == "PENDING" or real_head

    if rtx.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            rtx.resolve_launch_contract(rtx.PERF036_RUN_ID)
        return

    # Bound: this run only, and never against the placeholder.
    assert rtx.LAUNCHABLE_CONTRACT is rtx.PERF036
    assert real_head
    assert rtx.resolve_launch_contract(rtx.PERF036_RUN_ID) is rtx.PERF036
    with pytest.raises(ValueError):
        rtx.resolve_launch_contract("rayline-not-a-preregistered-run")


def test_perf036_is_registered_with_the_launcher_it_will_run_under() -> None:
    """A registry the launcher does not consult is a packet that cannot run."""

    assert (
        rtx.resolve_launch_contract in launcher._resolve_contract.__globals__.values()
    )
    if rtx.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            launcher._resolve_contract(rtx.PERF036_RUN_ID)
    else:
        assert launcher._resolve_contract(rtx.PERF036_RUN_ID) is rtx.PERF036


def test_perf036_budget_is_priced_on_rtx6000_seconds_and_granted_minimally() -> None:
    """The envelope is real, and the authority for it moved by exactly that.

    PERF035's grant landed the reserve at exactly the `$3.00` floor, so the
    minimum viable grant equals the envelope. The 2026-08-11 authorization
    moved the ceiling by precisely that amount, so the reserve after a full
    envelope lands on the floor again, exactly.
    """

    contract = rtx.PERF036.budget
    assert contract.encoder_gpu == "RTX-PRO-6000"

    rate = budget.resource_rate_usd_per_second("RTX-PRO-6000")
    # Modal on-demand RTX PRO 6000 seconds plus the unchanged 8-core / 64 GiB
    # container -- the same box the H100 and L4 packets ran, on other silicon.
    assert rate == pytest.approx(0.000842 + 8 * 0.0000131 + 64 * 0.00000222)
    envelope = 5160 * rate
    assert envelope == pytest.approx(5.6186208)

    # Granted minimally: the receipt exists and the reserve after a full
    # envelope is exactly the floor, to the cent and beyond.
    receipt = budget.budget_receipt(contract)
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(5.6186208)
    reserve = contract.authorized_cumulative_usd - (
        contract.previous_conservative_usd + envelope
    )
    assert reserve == pytest.approx(contract.required_reserve_usd)
    assert rtx.MINIMUM_VIABLE_GRANT_USD == pytest.approx(envelope)

    # The authority line moved by exactly the grant above PERF035's ceiling,
    # and the conservative position is PERF035's full-envelope close.
    assert rtx.AUTHORIZED_CUMULATIVE_USD == pytest.approx(
        l4.AUTHORIZED_CUMULATIVE_USD + rtx.MINIMUM_VIABLE_GRANT_USD
    )
    assert rtx.PREVIOUS_CONSERVATIVE_USD == pytest.approx(188.841229466383)
    assert rtx.PREVIOUS_CONSERVATIVE_USD == pytest.approx(
        l4.PREVIOUS_CONSERVATIVE_USD + l4.MINIMUM_VIABLE_GRANT_USD
    )
    # The ceiling is a real bound on an RTX PRO 6000 packet, not one inherited
    # from PERF034's H100 arithmetic.
    assert contract.packet_ceiling_usd == 6.0
    assert envelope < contract.packet_ceiling_usd


def test_perf036_leaves_every_recorded_packet_priced_exactly_as_recorded() -> None:
    """Adding a GPU class may not reprice a single closed run."""

    assert budget.resource_rate_usd_per_second() == pytest.approx(0.00134388)
    assert budget.resource_rate_usd_per_second("H100") == pytest.approx(0.00134388)
    receipt = budget.budget_receipt(capacity.PERF034.budget)
    assert receipt["maximum_resource_envelope_usd"] == pytest.approx(6.9344208)
    assert receipt["pricing_snapshot"] == budget.PRICING_SNAPSHOT
    l4_receipt = budget.budget_receipt(l4.PERF035.budget)
    assert l4_receipt["maximum_resource_envelope_usd"] == pytest.approx(2.4194208)
    assert l4_receipt["pricing_snapshot"] == budget.PRICING_SNAPSHOTS["L4"]
    # Three classes, three snapshots: a receipt can never claim one card's
    # pricing for another card's run.
    snapshots = {
        budget.PRICING_SNAPSHOT,
        budget.PRICING_SNAPSHOTS["L4"],
        budget.PRICING_SNAPSHOTS["RTX-PRO-6000"],
    }
    assert len(snapshots) == 3
    with pytest.raises(budget.BudgetError):
        budget.resource_rate_usd_per_second("A100")


def test_perf036_moves_only_the_card() -> None:
    """PERF033's and PERF035's shape exactly, on PERF020's corpus and topology.

    The cross-GPU curve is only readable if the silicon is the one variable,
    so the corpus and topology digests are asserted byte-identical to the
    ones every eight-lane packet since PERF020 has carried.
    """

    assert rtx.EPISODE_LANES == 8
    assert rtx.PERF036.measured_cases == 32
    assert rtx.PERF036.warmup_cases == 4
    assert rtx.PERF036.measured_episodes == 8
    assert rtx.PERF036.warmup_episodes == 1
    assert rtx.PERF036.corpus_sha256 == open_loop.PERF020.corpus_sha256
    assert rtx.PERF036.topology_sha256 == open_loop.PERF020.topology_sha256
    assert rtx.PERF036.corpus_sha256 == l4.PERF035.corpus_sha256
    assert rtx.PERF036.encoder_build_id == knee.FLASHINFER_BUILD_ID
    assert rtx.PERF036.encoder_gdn_prefill_backend == "flashinfer"


def test_perf036_predicts_its_ceiling_from_the_measured_cross_gpu_anchor() -> None:
    """The prediction scales a measured ceiling, and the test says why.

    PERF035 falsified the token-model calculator; the method that survived is
    naive dense-FP16 TFLOPS scaling from a measured anchor, which reproduces
    PERF035's L4 ceiling to within 10% when run from PERF033's H100 number.
    If a regression breaks that cross-check, this fails here rather than as a
    wrong RTX number nobody can check.
    """

    assert rtx.PREDICTED_CEILING_DPS == pytest.approx(0.7843, abs=1e-4)
    assert rtx.PREDICTED_CEILING_DPS == pytest.approx(
        rtx.PERF035_TOP_COMPLETION_THROUGHPUT_DPS * rtx.RTX6000_TFLOPS / rtx.L4_TFLOPS
    )

    # The validation the method already has: H100 -> L4, prediction vs the
    # ceiling PERF035 actually measured.
    l4_naive = (
        rtx.PERF033_TOP_COMPLETION_THROUGHPUT_DPS * rtx.L4_TFLOPS / rtx.H100_TFLOPS
    )
    assert l4_naive == pytest.approx(
        rtx.PERF035_TOP_COMPLETION_THROUGHPUT_DPS, rel=0.10
    )

    # The band is a validation target, not an integrity gate, exactly as it
    # was for PERF035 -- and wide enough that the unconfirmed Server-vs-
    # Workstation edition (480 vs 503.8 TFLOPS) cannot flip the outcome.
    assert rtx.PREDICTED_CEILING_TOLERANCE == 0.30
    low = rtx.PREDICTED_CEILING_DPS * (1 - rtx.PREDICTED_CEILING_TOLERANCE)
    high = rtx.PREDICTED_CEILING_DPS * (1 + rtx.PREDICTED_CEILING_TOLERANCE)
    assert (low, high) == pytest.approx((0.5490, 1.0195), abs=1e-4)
    workstation = rtx.PERF035_TOP_COMPLETION_THROUGHPUT_DPS * 503.8 / rtx.L4_TFLOPS
    assert low < workstation < high
    # The whole band sits below PERF033's measured rig ceiling, so a plateau
    # inside it is an encoder bound, not the rig's.
    assert high < rtx.PERF033_TOP_COMPLETION_THROUGHPUT_DPS


def test_perf036_rungs_bracket_the_predicted_ceiling() -> None:
    labels = [cell.label for cell in rtx.PERF036.cells]
    rates = [cell.offered_rate_rps for cell in rtx.PERF036.cells]

    assert labels == ["r032", "r064", "r096", "r144"]
    assert labels[0] == rtx.ANCHOR_CELL
    assert rates == sorted(rates)
    assert len(set(rates)) == len(rates)

    realized = [rate * rtx.ANCHOR_REALIZED_PER_OFFERED for rate in rates]
    multiples = [value / rtx.PREDICTED_CEILING_DPS for value in realized]
    # Anchor at half the prediction, top rung past even the H100 rig ceiling,
    # so the plateau has somewhere to appear wherever in the band the card
    # lands.
    assert multiples == pytest.approx([0.5065, 1.0130, 1.5194, 2.2792], abs=1e-3)
    assert multiples[0] < 0.6
    band_high = rtx.PREDICTED_CEILING_DPS * (1 + rtx.PREDICTED_CEILING_TOLERANCE)
    assert realized[-1] > band_high
    assert realized[-1] > rtx.PERF033_TOP_COMPLETION_THROUGHPUT_DPS

    # Every rung's schedule plus its completion time fits the paid wall, at
    # the prediction, at both edges of the band, at half the prediction --
    # and even at PERF035's L4 ceiling, the zero-speedup case. Only a card
    # slower than the L4 could breach the wall.
    cases = rtx.MEASURED_CASES + rtx.WARMUP_CASES
    for ceiling in (
        rtx.PREDICTED_CEILING_DPS,
        rtx.PREDICTED_CEILING_DPS * (1 - rtx.PREDICTED_CEILING_TOLERANCE),
        rtx.PREDICTED_CEILING_DPS / 2,
        rtx.PERF035_TOP_COMPLETION_THROUGHPUT_DPS,
    ):
        both_arms = 2 * sum(max(cases / rate, cases / ceiling) + 16 for rate in rates)
        assert both_arms < rtx.PERF036.budget.maximum_paid_wall_seconds


def test_perf036_shares_its_anchor_rung_with_perf035() -> None:
    """One rung is byte-identical across the two cards, by construction.

    The `r032` cell carries the same seeded schedule and identity PERF035
    ran, so the L4-vs-RTX ratio is readable directly off one shared rung --
    the same anchor property PERF033 carried against PERF032.
    """

    anchor = next(cell for cell in rtx.PERF036.cells if cell.label == rtx.ANCHOR_CELL)
    shared = next(cell for cell in l4.PERF035.cells if cell.label == "r032")
    assert anchor.offered_rate_rps == shared.offered_rate_rps == 0.32
    assert anchor.workload_sha256 == shared.workload_sha256
    assert anchor.identity_sha256 == shared.identity_sha256
    # Same seeded schedules, same measured realized-per-offered ratio.
    assert rtx.ANCHOR_REALIZED_PER_OFFERED == l4.ANCHOR_REALIZED_PER_OFFERED


def test_perf036_arms_both_firing_points() -> None:
    criterion = rtx.PERF036.saturation

    assert isinstance(criterion, open_loop.SaturationCriterion)
    assert criterion.episode_lanes == rtx.EPISODE_LANES == 8
    assert criterion.occupancy_ratio == 1.0
    assert criterion.throughput_plateau_gain == 1 / 3
    # Same calibration PERF034 armed and PERF035 fired, on the same comparator.
    assert (
        criterion.throughput_plateau_gain
        == capacity.PERF034.saturation.throughput_plateau_gain
    )
    assert (
        criterion.throughput_plateau_gain
        == l4.PERF035.saturation.throughput_plateau_gain
    )


def test_perf036_owns_its_rtx6000_encoder_app() -> None:
    """The GPU class is part of the deployment, so it needs its own app name."""

    assert rtx.PERF036.encoder_app_name == rtx.PERF036_APP_NAME
    assert rtx.PERF036.encoder_gpu == "RTX-PRO-6000"
    assert rtx.PERF036_APP_NAME not in {
        knee.FLASHINFER_APP_NAME,
        capacity.PERF034_APP_NAME,
        l4.PERF035_APP_NAME,
    }
    # Every other packet's evidence keeps claiming its recorded card.
    assert capacity.PERF034.encoder_gpu == "H100"
    assert knee_v2.PERF033.encoder_gpu == "H100"
    assert open_loop.PERF020.encoder_gpu == "H100"
    assert l4.PERF035.encoder_gpu == "L4"

    source = SESSION_SERVICE.read_text()
    assert f'"{rtx.PERF036_APP_NAME}": "flashinfer"' in source
    # This app name, and only this one, deploys on the RTX PRO 6000.
    assert (
        "elif APP_NAME in PERF036_APP_PROFILES:\n"
        '    GPU_TYPE = "RTX-PRO-6000"\n' in source
    )
    # The cap raise stays PERF034's: eight lanes and 32 ingress inputs here,
    # because the packet's one variable is the silicon.
    assert "MAX_SESSIONS = 32 if APP_NAME in PERF034_APP_PROFILES else 8" in source


def test_perf036_memory_is_safe_by_the_derived_cap_for_the_first_time() -> None:
    """Eight lanes fit this card outright, and the contract says by how much.

    The same worst case that overflows the L4 -- eight lanes at
    `MAX_SERIALIZED_TOKENS`, 24 GiB of KV -- is under a third of the 96 GB
    card's pool, so PERF036 is the family's first eight-lane packet whose
    admissibility does not rest on corpus construction.
    """

    kib_per_token = 12
    max_serialized_tokens = 262_144
    worst_case_gib = (
        rtx.EPISODE_LANES * max_serialized_tokens * kib_per_token / 1024 / 1024
    )
    assert worst_case_gib == pytest.approx(24.0)

    peak_gib = rtx.CORPUS_ENCODER_TOKENS * kib_per_token / 1024 / 1024
    assert peak_gib == pytest.approx(12.92, abs=0.01)
    # A ~80 GiB pool: 96 GB at 0.92 utilization, less ~1.6 GB of weights.
    pool_gib = (96 * 0.92 - 1.6) * 1e9 / 1024**3
    assert worst_case_gib < pool_gib / 3
    # The L4 bound this packet escapes, stated the way the PERF035 test pins
    # it: the same worst case overflows a ~19 GiB L4 pool.
    assert worst_case_gib > 19.0


def test_perf036_corpus_encoder_tokens_are_the_corpus_they_claim() -> None:
    """The prediction's anchor rests on this corpus, so read it off the packet.

    Retained pooling prefills each turn's delta exactly once, so an episode's
    total encoder work is its final serialized length. The sum must be the
    same 1,129,231 PERF035 carried, because the corpus is the same corpus.
    """

    corpus_path = PACKET_DIR / "corpus.json"
    if not corpus_path.is_file():
        pytest.skip("packet-perf036 is not present")
    corpus = json.loads(corpus_path.read_text())

    episodes: dict[str, int] = {}
    for case in corpus["measured"]:
        episode = case["episode_id"]
        episodes[episode] = max(episodes.get(episode, 0), case["input_tokens"])

    assert len(episodes) == rtx.MEASURED_EPISODES
    assert sum(episodes.values()) == rtx.CORPUS_ENCODER_TOKENS
    assert rtx.CORPUS_ENCODER_TOKENS == l4.CORPUS_ENCODER_TOKENS


def test_perf036_trace_digest_is_the_recorded_perf020_trace() -> None:
    """PERF036 runs the same 32 cases, so continuity is exact, not a prefix."""

    assert rtx.PERF020_TRACE_SHA256 == l4.PERF020_TRACE_SHA256
    receipt_path = PERF033_RUN / "r120" / "rayline_arc.json"
    if not receipt_path.is_file():
        pytest.skip(f"{PERF033_RUN.name} receipts are not present")
    recorded = json.loads(receipt_path.read_text())

    assert (
        recorded["results"]["selected_worker_trace_sha256"] == rtx.PERF020_TRACE_SHA256
    )


def test_perf036_digests_match_the_generated_packet() -> None:
    """The contract's digests are the packet's, not plausible-looking strings."""

    if not PACKET_DIR.is_dir():
        pytest.skip("packet-perf036 is not present")
    from rayline_three_arm_launcher import _sha256

    assert _sha256(PACKET_DIR / "manifest.json") == rtx.PERF036.packet_manifest_sha256
    assert _sha256(PACKET_DIR / "corpus.json") == rtx.PERF036.corpus_sha256
    assert _sha256(PACKET_DIR / "topology.json") == rtx.PERF036.topology_sha256
    for cell in rtx.PERF036.cells:
        cell_dir = PACKET_DIR / "cells" / cell.label
        assert _sha256(cell_dir / "workload.json") == cell.workload_sha256
        assert _sha256(cell_dir / "identity.json") == cell.identity_sha256
        workload = json.loads((cell_dir / "workload.json").read_text())
        assert workload["offered_rate_rps"] == cell.offered_rate_rps
        assert workload["max_episode_lanes"] == rtx.EPISODE_LANES
