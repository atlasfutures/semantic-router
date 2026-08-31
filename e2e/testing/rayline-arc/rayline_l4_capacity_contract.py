#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepared PERF035 eight-lane FlashInfer capacity packet on a Modal L4.

Every recorded Rayline capacity number was measured on an H100. The
deployment target is GCP Cloud Run with GPU, which sells exactly one GPU class
this repo can also rent on Modal: the 24 GB L4. Cloud Run's other option, the
RTX PRO 6000, Modal does not offer at all. So the only capacity claim the
deployment can honestly carry today is a TFLOPS-scaled estimate with a stated
`+/-30%` band -- an estimate that has never been checked against the silicon it
describes. PERF035 converts that estimate into a measured floor.

Nothing about the measurement changes except the card. The corpus, the
topology, the engine build, the model revision, the prefill backend, the lane
count and the case counts are all PERF033's. `corpus_sha256` and
`topology_sha256` are byte-identical to PERF020's, as they have been in every
eight-lane packet since. That is deliberate: with one variable moved, the
cross-GPU ratio is the result, and it is readable directly against PERF033's
recorded curve.

**What the ladder brackets.** The calculator
(`~/.agent/diagrams/rayline-arc-encoder-serving-cost.html`) scales measured
H100 throughput by the TFLOPS ratio: `115,764 tok/s * 121/989 = 14,163 tok/s`
predicted for the L4 on FlashInfer. Turning that into decisions per second
needs the corpus's own encoder work. Retained pooling prefills each turn's
delta exactly once, so an episode's total encoder work equals its final
serialized length; summed over the eight episodes the frozen corpus costs
`1,129,231` tokens. On the L4 that is `79.73` GPU-busy seconds for 32
decisions, hence a predicted ceiling of `0.40` completed decisions per second.

That derivation is not free-floating. Run it on the H100 and the same corpus
costs `9.755` seconds against PERF033's measured `18.13`-second top cell -- a
busy fraction of `0.538` against the `0.53` PERF033 independently recorded.
The token model reproduces a number it was not fitted to, within 1.5%, which
is why the L4 figure is quoted as a prediction rather than a guess. It is
still a prediction: the `+/-30%` band the calculator states is carried here as
`PREDICTED_CEILING_TOLERANCE`, and PERF034's lesson is the reason it is
written down at all. PERF034's anchor missed its H100 prediction by 7.15% with
no preregistered tolerance, so nothing could say whether that was a hit or a
miss. This contract states the band before the run. It is a validation target
and explicitly not an integrity gate: a measured ceiling outside
`0.28 .. 0.52` falsifies the calculator's L4 extrapolation, which is a result,
not a voided arm.

**The rungs.** At the measured `1.2413` realized-per-offered ratio, offered
`0.16 / 0.32 / 0.48 / 0.72` gives realized arrivals of about
`0.20 / 0.40 / 0.60 / 0.89` decisions per second -- `0.49x / 0.99x / 1.48x /
2.23x` of the prediction. The anchor sits at half the predicted ceiling, well
inside where the encoder cannot bind; the top rung sits `1.7x` above even the
optimistic edge of the band, so the plateau has somewhere to appear. Both
firing points stay armed, for PERF034's reason: occupancy alone cannot see an
encoder that binds below the rig's ceiling, and on an L4 that is the expected
outcome rather than a remote one.

The paid wall holds across the whole band. The slowest rung is
arrival-limited at `36/0.16 = 225` seconds; the three faster ones are
completion-limited at `36/ceiling`. Both arms together come to about `1,160`
seconds at the predicted ceiling, `1,350` at the pessimistic edge, and `1,660`
even if the L4 lands at half the prediction, against a `2,400`-second
envelope. Only a card slower than `~0.17` decisions per second -- 2.4x worse
than predicted, far outside the stated band -- would breach it.

**Memory is preregistered as a cliff, not a curve, and it is the one place
this packet is tighter than its predecessors.** Worst case at eight lanes is
`8 * 262,144 = 2,097,152` resident tokens, which at the contract's 12 KiB per
token is 24 GiB -- more than the whole card. The eight-lane packet is
therefore *not* safe by the derived cap, and only ever safe by corpus
construction: the frozen corpus peaks at `1,129,231` tokens, `12.92` GiB,
against a pool of roughly `18-19` GiB (24 GB at `0.92` utilization, less
~1.6 GB of weights). That is about 70% of the pool, where PERF034's 32-lane
packet sat at 51% of its own. The margin is real but thinner, and it is stated
here so the run is read with that in mind. If memory binds anyway the
behaviour is unchanged from PERF034: the coordinator's `SessionCapacityError`
becomes a failed case and the `failed == 0` integrity gate voids the arm, with
`cache_miss_tokens` and `session_actions.rebuilt` required to stay exactly
zero. The single largest case is `247,808` tokens, `2.84` GiB in one lane,
which also fits `max_model_len = 262,144` on this pool.

`CHUNK_SCHEDULE_TOKENS = 8192` and `enforce_eager = True` are held fixed from
PERF033 and PERF034. On a card with an eighth of the compute they are the
first suspects if measured throughput falls below the band.
"""

from __future__ import annotations

from rayline_open_loop_contract import (
    MAXIMUM_ORPHAN_REQUEST_SECONDS,
    MAXIMUM_PAID_WALL_SECONDS,
    MAXIMUM_SCALEDOWN_SECONDS,
    PERF020,
    OpenLoopCell,
    OpenLoopRunContract,
    SaturationCriterion,
)
from rayline_saturation_knee_contract import FLASHINFER_BUILD_ID
from rayline_three_arm_budget import BudgetContract

PERF035_RUN_ID = "rayline-l4-capacity-perf035-20260811"

# ---------------------------------------------------------------------------
# HUMAN GATES. Both are fail-closed placeholders. Preparation may not move
# either one; only a reviewed authorization checkpoint may.
# ---------------------------------------------------------------------------

# The Pathfinder head this run's authority is pinned to: the pushed
# codex/rayline-vsr-mvp head whose registry entry records the confirmed
# grant. `_assert_pushed` forces the Pathfinder HEAD to equal this.
PATHFINDER_AUTHORIZATION_COMMIT = "511760abd07a978802bbfc2065dac19e2062f050"

# GRANTED 2026-08-11. The user approved the run in-session with the words
# "I approve the run. anything under $10. just do it." Per the family's
# precedent the ceiling moves by exactly the minimum viable grant --
# `$189.421808666383` plus `$2.4194208`, the L4 envelope itself, because
# PERF034's grant left the reserve exactly on the `$3.00` floor. The reserve
# after a full envelope is again exactly `$3.00`. The authorization checkpoint
# is Pathfinder commit 511760ab (registry entry records the confirmed grant).
AUTHORIZED_CUMULATIVE_USD = 191.841229466383
PREVIOUS_CONSERVATIVE_USD = 186.421808666383
MINIMUM_VIABLE_GRANT_USD = 2.4194208

# The eight-lane shape, unchanged from PERF033. This is not the largest rig the
# corpus can express -- PERF034 already ran the 32-lane one -- it is the
# largest that a 24 GB card can be preregistered to survive. The 128-case
# corpus peaks at 4,261,735 tokens, 49 GiB of KV at 12 KiB per token, which
# does not fit an L4 at any utilization. Eight lanes is a memory constraint
# here, not a rig choice.
EPISODE_LANES = 8
MEASURED_CASES = 32
WARMUP_CASES = 4
MEASURED_EPISODES = 8
WARMUP_EPISODES = 1

# The PERF035 encoder deployment. Same engine build, image, model revision and
# prefill backend as PERF031/032/033/034; a distinct app because the GPU class
# is part of the deployed service and the closed runs' evidence names apps that
# must stay on H100 forever.
PERF035_APP_NAME = "rayline-arc-session-encoder-flashinfer-perf035-l4"
PERF035_ENCODER_GPU = "L4"

# ---------------------------------------------------------------------------
# The anchor, which is a prediction rather than a measurement.
# ---------------------------------------------------------------------------

# No L4 cell has ever been recorded, so there is no measured cross-GPU anchor
# and this contract does not pretend otherwise. What it anchors against is the
# calculator's extrapolation, stated below with the derivation the docstring
# walks through. Recorded honestly: a predicted anchor, not a measured one.
H100_FLASHINFER_TOKENS_PER_SECOND = 115764.0  # PERF030, measured
H100_TFLOPS = 989.0
L4_TFLOPS = 121.0
PREDICTED_L4_TOKENS_PER_SECOND = (
    H100_FLASHINFER_TOKENS_PER_SECOND * L4_TFLOPS / H100_TFLOPS
)

# The corpus's own encoder work: the sum over the eight measured episodes of
# each episode's final serialized length. Retained pooling prefills each turn's
# delta exactly once, so this is the total prefill the frozen corpus costs.
CORPUS_ENCODER_TOKENS = 1_129_231
PREDICTED_CEILING_DPS = MEASURED_CASES / (
    CORPUS_ENCODER_TOKENS / PREDICTED_L4_TOKENS_PER_SECOND
)

# The band, preregistered because PERF034 had none and could not say whether
# its 7.15% anchor miss counted. This is the calculator's own stated
# uncertainty on TFLOPS-scaled figures. It is a validation target: a measured
# ceiling outside the band falsifies the extrapolation and is the result. It
# is NOT an integrity gate and voids nothing.
PREDICTED_CEILING_TOLERANCE = 0.30

# The same token model run on the H100, where PERF033 recorded the answer
# independently. It is carried as a constant so a regression that breaks the
# derivation is caught by the test rather than by a wrong L4 prediction.
PERF033_RECORDED_GPU_BUSY_FRACTION = 0.53
PERF033_TOP_COMPLETION_THROUGHPUT_DPS = 1.7651

# Realized-per-offered, measured on PERF033/PERF034's seeded schedules.
ANCHOR_CELL = "r016"
ANCHOR_REALIZED_PER_OFFERED = 1.2413

# PERF020's recorded `selected_worker_trace_sha256`, identical in every 32-case
# closed run from PERF018 through PERF033. PERF035 runs that same 32-case
# corpus, so unlike PERF034 the continuity check is exact equality rather than
# a prefix property. The digest is a recorded runtime value -- corpus cases
# carry no worker field -- and is carried here as the constant it is.
PERF020_TRACE_SHA256 = (
    "d9e93cf0f4c636a3838e41938d2ef3ff6e1d66a60860922f84771b3fa5158ac9"
)

PERF035 = OpenLoopRunContract(
    run_id=PERF035_RUN_ID,
    packet_manifest_sha256=(
        "63ace3342d79c686b8e23ea66e1bcdab974fe9ffea32b2924467096a4b8ef217"
    ),
    # Byte-identical to PERF020's, and to every eight-lane packet since. One
    # variable moves in this packet and it is the card.
    corpus_sha256=PERF020.corpus_sha256,
    topology_sha256=PERF020.topology_sha256,
    cells=(
        OpenLoopCell(
            label="r016",
            offered_rate_rps=0.16,
            workload_sha256=(
                "157026abf3d00c86ac9a0517fc2c1dc2b64ec1613f2df4103b972a98f7eeed54"
            ),
            identity_sha256=(
                "9cf0bd1392d476ba5a2ec2665c1d2502c99ea75ccda7eff0df329832588a906b"
            ),
        ),
        OpenLoopCell(
            label="r032",
            offered_rate_rps=0.32,
            workload_sha256=(
                "05434beb179974ecb4a8c8ed04f2ff05b9789fe8b7aa64f7460dada9157de7ba"
            ),
            identity_sha256=(
                "6a592e77e093769d322a4df26bfbccc185a5da899b5938ef22a2255a841fbac8"
            ),
        ),
        OpenLoopCell(
            label="r048",
            offered_rate_rps=0.48,
            workload_sha256=(
                "9e68769d143d400de80acdee51f9a4703b72c4af9b68a9c06bdc7851b114cc04"
            ),
            identity_sha256=(
                "6874f725cc07d947ec53c78b42266bc3776f2ecee2cd6c54644e5c45af254085"
            ),
        ),
        OpenLoopCell(
            label="r072",
            offered_rate_rps=0.72,
            workload_sha256=(
                "c19bcc971750912378037bbdbcafe5a2c1604f43d7894f2263ccca8b24f0e4bc"
            ),
            identity_sha256=(
                "1209ce1c361de6b968d76c53590fdb85dfa830ca88bab6f3dc7813252a4b8cff"
            ),
        ),
    ),
    compose_project_prefix="rayline-l4-capacity-perf035",
    temporary_prefix="rayline-perf035-",
    budget=BudgetContract(
        run_id=PERF035_RUN_ID,
        previous_conservative_usd=PREVIOUS_CONSERVATIVE_USD,
        authorized_cumulative_usd=AUTHORIZED_CUMULATIVE_USD,
        # Tightened from PERF034's `$7.00`. The L4 envelope is `$2.4194208`, so
        # a ceiling of `$3.00` is a real bound on this packet rather than a
        # number inherited from a card that costs five times as much.
        packet_ceiling_usd=3.0,
        required_reserve_usd=3.0,
        # The unchanged 40-minute paid-wall envelope. The container keeps its
        # 8 cores and 64 GiB so the L4 run stays comparable to the H100 ones
        # on everything except the card; only the GPU line item changes.
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_gpu=PERF035_ENCODER_GPU,
    ),
    encoder_app_name=PERF035_APP_NAME,
    encoder_build_id=FLASHINFER_BUILD_ID,
    encoder_gdn_prefill_backend="flashinfer",
    encoder_gpu=PERF035_ENCODER_GPU,
    measured_cases=MEASURED_CASES,
    warmup_cases=WARMUP_CASES,
    measured_episodes=MEASURED_EPISODES,
    warmup_episodes=WARMUP_EPISODES,
    # Both firing points armed, for PERF034's reason and with PERF034's
    # calibration. Occupancy is the physical rig ceiling PERF033 validated at
    # this exact lane count. The plateau floor of one third sits inside the
    # measured gap between unqueued cells (at least `0.46` on every recorded
    # receipt) and rungs past a known knee (at most `0.32`). On an L4 the
    # expected outcome is plateau first with occupancy short of the ceiling --
    # which is the direct proof of encoder binding that four H100 packets have
    # now failed to produce.
    saturation=SaturationCriterion(
        episode_lanes=EPISODE_LANES,
        occupancy_ratio=1.0,
        throughput_plateau_gain=1 / 3,
    ),
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

L4_CAPACITY_ARMS = (PERF035,)

# Closed 2026-08-11 after the one authorized execution. All four cells
# measured cleanly (failed=0, traces exactly equal to the PERF020 digest,
# comparison passed) and the run closed normally, cleanup included. The L4
# saturated from the first rung: occupancy pinned at 1.0 on every cell and
# the plateau criterion fired at r032 on both arms, topping out near
# 0.198 dps -- about half the TFLOPS-scaled prediction, outside its
# preregistered +/-30% validation band. The pin above stays as the record.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = None


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline L4 capacity arm is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
