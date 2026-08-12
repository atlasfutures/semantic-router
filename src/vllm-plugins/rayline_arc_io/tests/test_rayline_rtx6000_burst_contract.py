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
burst = importlib.import_module("rayline_rtx6000_burst_contract")
capacity = importlib.import_module("rayline_saturation_capacity_contract")
l4 = importlib.import_module("rayline_l4_capacity_contract")
rtx = importlib.import_module("rayline_rtx6000_capacity_contract")
knee = importlib.import_module("rayline_saturation_knee_contract")
launcher = importlib.import_module("rayline_open_loop_launcher")
open_loop = importlib.import_module("rayline_open_loop_contract")

PACKET_DIR = REPO_ROOT / ".agent-harness/rayline-parity/packet-perf034"
# The session service imports `modal` at module scope, so every assertion
# about it is made against its source text, never by importing it.
SESSION_SERVICE = Path(__file__).resolve().parents[1] / "modal_session_service.py"


def band() -> tuple[float, float]:
    centre = burst.PREDICTED_CEILING_DPS
    tolerance = burst.PREDICTED_CEILING_TOLERANCE
    return centre * (1 - tolerance), centre * (1 + tolerance)


def test_perf037_opens_at_most_itself_and_fails_closed() -> None:
    """Prepared means unbound against a literal no commit can equal.

    `PENDING` is what `_assert_pushed` compares the Pathfinder HEAD against,
    so it cannot pass; the launchable contract is separately `None`. Both gates
    must move for a launch to exist, and preparation may move neither.
    """

    pin = burst.PATHFINDER_AUTHORIZATION_COMMIT
    # The pin belongs to the packet, never to the bound run.
    assert burst.PERF037.pathfinder_authorization_commit == pin
    real_head = len(pin) == 40 and set(pin) <= set("0123456789abcdef")
    assert pin == "PENDING" or real_head

    if burst.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            burst.resolve_launch_contract(burst.PERF037_RUN_ID)
        return

    # Bound: this run only, and never against the placeholder.
    assert burst.LAUNCHABLE_CONTRACT is burst.PERF037
    assert real_head
    assert burst.resolve_launch_contract(burst.PERF037_RUN_ID) is burst.PERF037
    with pytest.raises(ValueError):
        burst.resolve_launch_contract("rayline-not-a-preregistered-run")


def test_perf037_is_registered_with_the_launcher_it_will_run_under() -> None:
    """A registry the launcher does not consult is a packet that cannot run."""

    assert (
        burst.resolve_launch_contract in launcher._resolve_contract.__globals__.values()
    )
    if burst.LAUNCHABLE_CONTRACT is None:
        with pytest.raises(ValueError):
            launcher._resolve_contract(burst.PERF037_RUN_ID)
    else:
        assert launcher._resolve_contract(burst.PERF037_RUN_ID) is burst.PERF037


def test_perf037_is_ungranted_and_needs_its_whole_envelope() -> None:
    """Nothing is left to spend, so the raise is the envelope, exactly.

    PERF036 closed on the reserve floor: its full-envelope cumulative is the
    conservative position here, and the standing ceiling is exactly `$3.00`
    above it, all of it reserve. So this packet cannot draw on partial
    headroom -- there is none -- and `budget_receipt` refuses until a human
    moves the ceiling by the whole `$5.6186208`.
    """

    contract = burst.PERF037.budget
    assert contract.encoder_gpu == "RTX-PRO-6000"

    rate = budget.resource_rate_usd_per_second("RTX-PRO-6000")
    # Modal on-demand RTX PRO 6000 seconds plus the unchanged 8-core / 64 GiB
    # container -- the same box PERF034 ran on an H100, on PERF036's silicon.
    assert rate == pytest.approx(0.000842 + 8 * 0.0000131 + 64 * 0.00000222)
    envelope = 5160 * rate
    assert envelope == pytest.approx(5.6186208)
    assert envelope < contract.packet_ceiling_usd == 6.0

    # Fail-closed by arithmetic, not by a flag: the reserve is short by the
    # whole envelope, so no receipt for this packet exists at all.
    with pytest.raises(budget.BudgetError):
        budget.budget_receipt(contract)
    assert budget.minimum_viable_grant_usd(contract) == pytest.approx(
        burst.MINIMUM_VIABLE_GRANT_USD
    )
    assert burst.MINIMUM_VIABLE_GRANT_USD == pytest.approx(envelope)

    # The standing position is PERF036's close, to the digit, and the grant
    # this packet needs lands the reserve back on the floor.
    assert burst.AUTHORIZED_CUMULATIVE_USD == pytest.approx(
        rtx.AUTHORIZED_CUMULATIVE_USD
    )
    assert burst.PREVIOUS_CONSERVATIVE_USD == pytest.approx(
        rtx.PREVIOUS_CONSERVATIVE_USD + rtx.MINIMUM_VIABLE_GRANT_USD
    )
    assert burst.GRANTED_CUMULATIVE_WOULD_BE_USD == pytest.approx(
        burst.AUTHORIZED_CUMULATIVE_USD + burst.MINIMUM_VIABLE_GRANT_USD
    )
    reserve_if_granted = burst.GRANTED_CUMULATIVE_WOULD_BE_USD - (
        burst.PREVIOUS_CONSERVATIVE_USD + envelope
    )
    assert reserve_if_granted == pytest.approx(contract.required_reserve_usd)


def test_perf037_leaves_every_recorded_packet_priced_exactly_as_recorded() -> None:
    """Preparing a packet may not reprice a single closed run."""

    for contract in (capacity.PERF034, l4.PERF035, rtx.PERF036):
        assert budget.minimum_viable_grant_usd(contract.budget) == 0.0
    assert budget.budget_receipt(capacity.PERF034.budget)[
        "maximum_resource_envelope_usd"
    ] == pytest.approx(6.9344208)
    assert budget.budget_receipt(rtx.PERF036.budget)[
        "maximum_resource_envelope_usd"
    ] == pytest.approx(5.6186208)


def test_perf037_runs_perf034s_packet_byte_for_byte() -> None:
    """The cross-GPU comparison is only readable if the silicon is the variable.

    Every digest is taken from the PERF034 contract object rather than
    retyped, so this is not a transcription check -- it is the statement that
    a transcription error cannot exist. What it does catch is a later edit
    that quietly gives PERF037 a packet of its own.
    """

    assert burst.PERF037.packet_manifest_sha256 == (
        capacity.PERF034.packet_manifest_sha256
    )
    assert burst.PERF037.corpus_sha256 == capacity.PERF034.corpus_sha256
    assert burst.PERF037.topology_sha256 == capacity.PERF034.topology_sha256
    assert burst.PERF037.cells == capacity.PERF034.cells
    assert burst.PERF037.topology_sha256 == open_loop.PERF020.topology_sha256
    # The 128-case corpus is a superset of the 32-case one every eight-lane
    # packet carries, so this digest must differ from PERF020's, not match it.
    assert burst.PERF037.corpus_sha256 != open_loop.PERF020.corpus_sha256

    assert burst.EPISODE_LANES == capacity.EPISODE_LANES == 32
    assert burst.PERF037.measured_cases == 128
    assert burst.PERF037.warmup_cases == 8
    assert burst.PERF037.measured_episodes == 32
    assert burst.PERF037.warmup_episodes == 2
    assert burst.PERF037.encoder_build_id == knee.FLASHINFER_BUILD_ID
    assert burst.PERF037.encoder_gdn_prefill_backend == "flashinfer"


def test_perf037_predicts_its_ceiling_from_the_measured_32_lane_anchor() -> None:
    """The prediction scales a measured ceiling, and the test says why.

    PERF036 validated naive dense-FP16 TFLOPS scaling on one hop at eight
    lanes. This is the same method at a second lane count, from the only
    32-lane ceiling the family has. The cross-check the method already carries
    on this exact card is pinned here: the same ratio applied to PERF033's
    eight-lane H100 figure reproduces PERF036's measured eight-lane RTX
    figure within 10%.
    """

    assert burst.PREDICTED_CEILING_DPS == pytest.approx(1.1193, abs=1e-4)
    assert burst.PREDICTED_CEILING_DPS == pytest.approx(
        burst.PERF034_TOP_COMPLETION_THROUGHPUT_DPS
        * burst.RTX6000_TFLOPS
        / burst.H100_TFLOPS
    )

    rtx8_naive = (
        burst.PERF033_TOP_COMPLETION_THROUGHPUT_DPS
        * burst.RTX6000_TFLOPS
        / burst.H100_TFLOPS
    )
    assert rtx8_naive == pytest.approx(
        burst.PERF036_TOP_COMPLETION_THROUGHPUT_DPS, rel=0.10
    )

    assert burst.PREDICTED_CEILING_TOLERANCE == 0.30
    low, high = band()
    assert (low, high) == pytest.approx((0.7835, 1.4550), abs=1e-4)

    # The second route to the same prediction: PERF036's measured eight-lane
    # RTX ceiling scaled by the H100's own 8-to-32-lane gain. It is not an
    # independent check -- it differs from the TFLOPS route by exactly the
    # residual above -- but it lands inside the band, so which route the
    # prediction takes cannot flip the outcome.
    lane_route = burst.PERF036_TOP_COMPLETION_THROUGHPUT_DPS * (
        burst.PERF034_TOP_COMPLETION_THROUGHPUT_DPS
        / burst.PERF033_TOP_COMPLETION_THROUGHPUT_DPS
    )
    assert lane_route == pytest.approx(1.1598, abs=1e-4)
    assert low < lane_route < high
    assert lane_route / burst.PREDICTED_CEILING_DPS == pytest.approx(
        burst.PERF036_TOP_COMPLETION_THROUGHPUT_DPS / rtx8_naive
    )


def test_perf037_predicts_the_burst_is_not_absorbed_anywhere_in_the_band() -> None:
    """The preregistered answer is no, and the whole band says so.

    The falsification is stated as a number rather than a mood: a measured
    ceiling of `2.33` or better would falsify the prediction, and that is more
    than double the point prediction and well outside the band. 32 lanes is
    every episode the corpus has, so "at what lane count" has no answer above
    it either.
    """

    _low, high = band()
    assert burst.PRODUCTION_BURST_DPS == 2.33
    assert burst.PRODUCTION_BURST_DPS > high
    assert burst.PRODUCTION_BURST_DPS / burst.PREDICTED_CEILING_DPS == pytest.approx(
        2.0817, abs=1e-4
    )
    assert burst.PRODUCTION_BURST_DPS / high == pytest.approx(1.6013, abs=1e-4)

    # It already exceeds the eight-lane figure PERF036 measured on this card,
    # which is why the packet exists at all.
    assert burst.PRODUCTION_BURST_DPS > burst.PERF036_TOP_COMPLETION_THROUGHPUT_DPS
    # And 32 lanes is the whole corpus: no wider packet over it can exist.
    assert burst.EPISODE_LANES == burst.MEASURED_EPISODES


def test_perf037_converts_a_ceiling_into_the_burst_duration_it_can_take() -> None:
    """The decision needs a duration, not a verdict, so the contract computes it.

    A burst is transient, so a deployment slower than the burst still absorbs
    a short one by queueing. Backlog accumulates at `burst - ceiling` and
    clears at `ceiling`, which gives the budget below.
    """

    low, high = band()
    at_prediction = burst.absorbable_burst_seconds(burst.PREDICTED_CEILING_DPS)
    assert at_prediction == pytest.approx(27.73, abs=0.01)
    assert burst.absorbable_burst_seconds(low) == pytest.approx(15.20, abs=0.01)
    assert burst.absorbable_burst_seconds(high) == pytest.approx(49.89, abs=0.01)

    # The identity the formula inverts: a burst of exactly this length leaves
    # a backlog that takes exactly the recovery budget to clear.
    backlog = (burst.PRODUCTION_BURST_DPS - burst.PREDICTED_CEILING_DPS) * at_prediction
    assert backlog / burst.PREDICTED_CEILING_DPS == pytest.approx(
        burst.RECOVERY_BUDGET_SECONDS
    )

    # A deployment at or above the burst rate never falls behind.
    assert burst.absorbable_burst_seconds(burst.PRODUCTION_BURST_DPS) == float("inf")
    assert burst.absorbable_burst_seconds(5.0) == float("inf")
    with pytest.raises(ValueError):
        burst.absorbable_burst_seconds(0.0)


def test_perf037_absorption_predicate_does_not_fire_on_perf034s_receipt() -> None:
    """Calibrated against a recorded run, so it is not trivially satisfiable.

    PERF034's `r240` is the one recorded rung whose realized arrivals exceed
    the production burst. On an H100 at 32 lanes it still did not absorb --
    it completed barely half its arrivals and its peak backlog went one past
    the lane count. A predicate that passed there would be measuring nothing.
    """

    assert not burst.absorbs_burst(
        burst.PERF034_BURST_REALIZED_DPS,
        burst.PERF034_BURST_COMPLETION_DPS,
        burst.PERF034_BURST_PEAK_BACKLOG,
    )
    assert burst.PERF034_BURST_REALIZED_DPS > burst.PRODUCTION_BURST_DPS
    assert burst.PERF034_BURST_PEAK_BACKLOG > burst.EPISODE_LANES

    # Nor at PERF034's top rung, which completed a third of its arrivals.
    assert not burst.absorbs_burst(
        6.7285, burst.PERF034_TOP_COMPLETION_THROUGHPUT_DPS, 31
    )

    # It is satisfiable, though: a cell that keeps up inside its lanes passes.
    assert burst.absorbs_burst(2.40, 2.40, burst.EPISODE_LANES)
    # Each condition is load-bearing on its own.
    assert not burst.absorbs_burst(2.00, 2.00, burst.EPISODE_LANES)
    assert not burst.absorbs_burst(2.40, 2.20, burst.EPISODE_LANES)
    assert not burst.absorbs_burst(2.40, 2.40, burst.EPISODE_LANES + 1)


def test_perf037_keeps_the_drain_clause_off_the_verdict_it_cannot_judge() -> None:
    """Three plateau verdicts died to this clause; the scope is preregistered.

    Marginal gain between rungs inherits the finite corpus's drain artifact,
    which is why the clause exists and why it keeps firing. Absorption and the
    ceiling are read off quantities one cell measures directly, so the clause
    may not reach them -- recorded here rather than argued at analysis time.
    """

    assert burst.DRAIN_CLAUSE_VOIDS_PLATEAU_VERDICT is True
    assert burst.DRAIN_CLAUSE_VOIDS_ABSORPTION_VERDICT is False
    assert burst.DRAIN_CLAUSE_VOIDS_CEILING_MEASUREMENT is False

    # The plateau verdict it may void is carried unchanged from PERF034, on
    # the same comparator and the same calibration, as corroboration only.
    criterion = burst.PERF037.saturation
    assert isinstance(criterion, open_loop.SaturationCriterion)
    assert criterion.episode_lanes == burst.EPISODE_LANES == 32
    assert criterion.occupancy_ratio == 1.0
    assert criterion.throughput_plateau_gain == 1 / 3
    assert criterion == capacity.PERF034.saturation


def test_perf037_rungs_bracket_the_production_burst() -> None:
    """PERF034's ladder, read against this card's prediction.

    The rungs are not redesigned -- redesigning them would cost the byte
    identity the cross-GPU comparison rests on -- so what this pins is that
    the inherited ladder still brackets the question: one rung below the
    production burst, three above it, and a top rung far past the prediction.
    """

    labels = [cell.label for cell in burst.PERF037.cells]
    rates = [cell.offered_rate_rps for cell in burst.PERF037.cells]
    assert labels == ["r120", "r240", "r400", "r645"]
    assert labels[0] == burst.ANCHOR_CELL
    assert rates == sorted(rates)

    # PERF034 measured 1.0432 realized per offered at 32 lanes, not the 1.2413
    # its ladder was designed with. Reusing its cells reuses that measurement.
    realized = [rate * burst.ANCHOR_REALIZED_PER_OFFERED for rate in rates]
    assert realized == pytest.approx([1.2518, 2.5037, 4.1728, 6.7286], abs=1e-3)
    below = [value for value in realized if value < burst.PRODUCTION_BURST_DPS]
    assert len(below) == 1
    assert realized[-1] > 5 * burst.PREDICTED_CEILING_DPS

    # The one rung whose realized arrivals PERF034 recorded against the burst.
    assert realized[1] == pytest.approx(burst.PERF034_BURST_REALIZED_DPS, abs=1e-3)


def test_perf037_fits_the_paid_wall_on_perf034s_own_calibration() -> None:
    """The wall is the schedule risk, so it is checked against a measured run.

    The shared fit formula ran about `1.19x` optimistic on PERF034 -- 845
    recorded seconds against its estimate -- so the estimate for this card is
    corrected by that ratio and charged the image deploy too. It holds at the
    prediction and at the band floor. It binds only below roughly `0.61`
    decisions per second, which is under PERF036's measured eight-lane figure
    and would mean 32 lanes ran slower than eight.
    """

    cases = burst.MEASURED_CASES + burst.WARMUP_CASES
    rates = [cell.offered_rate_rps for cell in burst.PERF037.cells]
    wall = burst.PERF037.budget.maximum_paid_wall_seconds

    def estimate(ceiling: float) -> float:
        return 2 * sum(max(cases / rate, cases / ceiling) + 16 for rate in rates)

    calibration = burst.PERF034_RECEIPT_SPAN_SECONDS / estimate(
        burst.PERF034_TOP_COMPLETION_THROUGHPUT_DPS
    )
    assert calibration == pytest.approx(1.1927, abs=1e-4)

    def charged(ceiling: float) -> float:
        return estimate(ceiling) * calibration + burst.PERF034_IMAGE_DEPLOY_SECONDS

    low, _high = band()
    assert charged(burst.PREDICTED_CEILING_DPS) == pytest.approx(1445.0, abs=1.0)
    assert charged(low) == pytest.approx(1942.0, abs=1.0)
    for ceiling in (
        burst.PREDICTED_CEILING_DPS,
        low,
        burst.PERF036_TOP_COMPLETION_THROUGHPUT_DPS,
    ):
        assert charged(ceiling) < wall

    # Where it would bind, stated as the ceiling rather than as a margin.
    assert charged(0.61) > wall
    assert charged(0.62) < wall
    assert 0.62 < burst.PERF036_TOP_COMPLETION_THROUGHPUT_DPS


def test_perf037_owns_its_rtx6000_32_lane_encoder_app() -> None:
    """Two deviations at once, so the app must carry both and own the name."""

    assert burst.PERF037.encoder_app_name == burst.PERF037_APP_NAME
    assert burst.PERF037.encoder_gpu == "RTX-PRO-6000"
    assert burst.PERF037_APP_NAME not in {
        knee.FLASHINFER_APP_NAME,
        capacity.PERF034_APP_NAME,
        l4.PERF035_APP_NAME,
        rtx.PERF036_APP_NAME,
    }
    # Every other packet's evidence keeps claiming its recorded card.
    assert capacity.PERF034.encoder_gpu == "H100"
    assert open_loop.PERF020.encoder_gpu == "H100"
    assert l4.PERF035.encoder_gpu == "L4"
    assert rtx.PERF036.encoder_gpu == "RTX-PRO-6000"

    source = SESSION_SERVICE.read_text()
    assert f'"{burst.PERF037_APP_NAME}": "flashinfer"' in source
    card_set = source.split("RTX6000_APP_PROFILES = ")[1].split("\n")[0]
    cap_set = source.split("CAP_RAISED_APP_PROFILES = ")[1].split("\n")[0]
    # This app is the only member of both sets, which is what makes it the
    # only app that deploys on the card at 32 lanes.
    assert "PERF037_APP_PROFILES" in card_set
    assert "PERF037_APP_PROFILES" in cap_set
    assert "PERF036_APP_PROFILES" in card_set and "PERF036" not in cap_set
    assert "PERF034_APP_PROFILES" in cap_set and "PERF034" not in card_set
    assert "MAX_SESSIONS = 32 if APP_NAME in CAP_RAISED_APP_PROFILES else 8" in source
    assert (
        "MAX_CONCURRENT_INPUTS = 64 if APP_NAME in CAP_RAISED_APP_PROFILES else 32"
        in source
    )


def test_perf037_memory_is_safe_only_by_corpus_construction() -> None:
    """The derived cap does not fit this card either, and the contract says so.

    32 lanes at `MAX_SERIALIZED_TOKENS` is 96 GiB of KV against a ~81 GiB
    pool. What makes the packet admissible is the frozen corpus's own peak,
    which is a smaller fraction of this pool than of the H100 pool that
    already ran this exact corpus clean.
    """

    kib_per_token = 12
    max_serialized_tokens = 262_144
    worst_case_gib = (
        burst.EPISODE_LANES * max_serialized_tokens * kib_per_token / 1024 / 1024
    )
    assert worst_case_gib == pytest.approx(96.0)

    peak_gib = burst.CORPUS_ENCODER_TOKENS * kib_per_token / 1024 / 1024
    assert peak_gib == pytest.approx(48.77, abs=0.01)

    rtx_pool_gib = (96 * 0.92 - 1.6) * 1e9 / 1024**3
    h100_pool_gib = (80 * 0.92 - 1.6) * 1e9 / 1024**3
    # Not safe by the derived cap -- that is the honest statement PERF036
    # could make at eight lanes and this packet cannot.
    assert worst_case_gib > rtx_pool_gib
    # Safe by corpus construction, with more room than the run that produced
    # this packet's anchor had.
    assert peak_gib < rtx_pool_gib
    assert peak_gib / rtx_pool_gib < peak_gib / h100_pool_gib
    assert peak_gib / rtx_pool_gib == pytest.approx(0.604, abs=0.005)


def test_perf037_states_what_teardown_must_show() -> None:
    """The one instrument an agent is most likely to skip is named explicitly."""

    requirements = burst.TEARDOWN_REQUIREMENTS
    assert len(requirements) == 5
    assert any(
        "encoder_containers_remaining exactly 0" in line for line in requirements
    )
    # PERF036's cleanup claim stood on two independent readings, not on the
    # launcher's own report. This packet preregisters the same.
    assert any("modal container list" in line for line in requirements)
    assert any("32/32 measured sessions" in line for line in requirements)


def test_perf037_independent_teardown_check_is_app_scoped() -> None:
    """The second instrument measures this run's leak, not the environment's.

    The `dev` environment is shared with foreign lanes this experiment cannot
    stop, so an environment-wide emptiness gate would be unsatisfiable by a
    perfect run and would reward stopping another lane's work. The gate is
    scoped to this packet's own app -- the same filter the launcher applies --
    and the evidence bar rises to a full quoted listing instead.
    """

    independent = [
        line for line in burst.TEARDOWN_REQUIREMENTS if "modal container list" in line
    ]
    assert len(independent) == 1
    requirement = independent[0]
    # App-scoped, not environment-scoped.
    assert "PERF037_APP_NAME" in requirement
    assert "returns empty" not in requirement
    # The compensating evidence bar is preregistered, not left to judgement.
    assert "pre-run and post-teardown" in requirement
    # The launcher's own check filters on exactly this name, so the two
    # instruments measure the same thing by two routes.
    assert burst.PERF037.encoder_app_name == burst.PERF037_APP_NAME


def test_perf037_inherits_perf034s_unrunnable_continuity_check() -> None:
    """A check that cannot run is recorded as such, not quietly dropped.

    The probe hashes the selected-worker trace without persisting its
    entries, so PERF034 could not compute the 32-case prefix digest and
    neither can this run. The within-run chain (one digest across all cells)
    is what remains, and a successor that wants continuity must change the
    probe first.
    """

    assert burst.TRACE_PREFIX_CHECK_IS_RUNNABLE is False
    assert burst.PERF020_TRACE_PREFIX_SHA256 == capacity.PERF020_TRACE_PREFIX_SHA256
    assert burst.PERF020_TRACE_PREFIX_SHA256 == rtx.PERF020_TRACE_SHA256


def test_perf037_corpus_encoder_tokens_are_the_corpus_they_claim() -> None:
    """The memory bound rests on this corpus, so read it off the packet.

    Retained pooling prefills each turn's delta exactly once, so an episode's
    total encoder work is its final serialized length. The sum must be the
    4,261,735 PERF034 preregistered as its memory peak, because it is the
    same corpus file.
    """

    corpus_path = PACKET_DIR / "corpus.json"
    if not corpus_path.is_file():
        pytest.skip("packet-perf034 is not present")
    corpus = json.loads(corpus_path.read_text())

    episodes: dict[str, int] = {}
    for case in corpus["measured"]:
        episode = case["episode_id"]
        episodes[episode] = max(episodes.get(episode, 0), case["input_tokens"])

    assert len(episodes) == burst.MEASURED_EPISODES
    assert sum(episodes.values()) == burst.CORPUS_ENCODER_TOKENS
    assert len(corpus["measured"]) == burst.MEASURED_CASES
    assert len(corpus["warmup"]) == burst.WARMUP_CASES


def test_perf037_digests_match_the_generated_packet() -> None:
    """The contract's digests are the packet's, not plausible-looking strings."""

    if not PACKET_DIR.is_dir():
        pytest.skip("packet-perf034 is not present")
    from rayline_three_arm_launcher import _sha256

    assert _sha256(PACKET_DIR / "manifest.json") == burst.PERF037.packet_manifest_sha256
    assert _sha256(PACKET_DIR / "corpus.json") == burst.PERF037.corpus_sha256
    assert _sha256(PACKET_DIR / "topology.json") == burst.PERF037.topology_sha256
    for cell in burst.PERF037.cells:
        cell_dir = PACKET_DIR / "cells" / cell.label
        assert _sha256(cell_dir / "workload.json") == cell.workload_sha256
        assert _sha256(cell_dir / "identity.json") == cell.identity_sha256
        workload = json.loads((cell_dir / "workload.json").read_text())
        assert workload["offered_rate_rps"] == cell.offered_rate_rps
        assert workload["max_episode_lanes"] == burst.EPISODE_LANES
        # The rig produces one finite Poisson pulse per cell and nothing else.
        # The burst shape this packet measures is that pulse, and the contract
        # may not silently claim a shape the generator cannot make.
        assert workload["arrival_process"] == "seeded_poisson"
        assert workload["coordinated_omission_policy"] == "scheduled_arrival"
