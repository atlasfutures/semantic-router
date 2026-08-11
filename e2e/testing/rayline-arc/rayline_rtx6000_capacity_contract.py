#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepared PERF036 eight-lane FlashInfer capacity packet on a Modal RTX PRO 6000.

PERF035 did two things at once: it measured the first non-H100 ceiling in the
family (`0.1977` decisions per second on a 24 GB L4), and it falsified the
token-model calculator that had predicted `0.40` for that card -- the measured
ceiling landed at half the prediction, outside its preregistered `+/-30%`
band. What survived the falsification is the naive method the calculator was
supposed to improve on: scale a measured ceiling by the dense-FP16 tensor
TFLOPS ratio. Run naively from PERF033's measured H100 ceiling, that method
predicts `1.7651 * 121/989 = 0.2160` for the L4, within 9.2% of the `0.1977`
PERF035 measured. PERF036 tests that surviving method on a third card.

The card is the 96 GB RTX PRO 6000 (Blackwell), the second of GCP Cloud Run's
two GPU classes, which Modal added in April 2026 at `$0.000842/s`. With it the
family's cross-GPU curve gets a third point -- L4, RTX PRO 6000, H100 -- on
one byte-identical packet. Nothing about the measurement changes except the
card: the corpus, the topology, the engine build, the model revision, the
prefill backend, the lane count and the case counts are all PERF033's and
PERF035's, and `corpus_sha256` / `topology_sha256` are byte-identical to
PERF020's, as in every eight-lane packet since.

**The prediction, and its basis.** For the first time the anchor is a measured
cross-GPU scaling, not a token model: `0.1977 * 480/121 = 0.784` decisions per
second. The `480` is NVIDIA's Server Edition figure derived from the published
`1 PFLOP` sparse FP16 (dense is half, clock-scaled by the published FP32
ratio 120/126 from the Workstation Edition's whitepaper `503.8`). Which
edition Modal racks is not documented; the Workstation number would predict
`0.823` instead, inside the band either way, so the edition ambiguity cannot
flip the validation outcome. The `+/-30%` band is carried as
`PREDICTED_CEILING_TOLERANCE`, a validation target and explicitly not an
integrity gate: a measured ceiling outside `0.549 .. 1.020` falsifies naive
TFLOPS scaling from a measured anchor, which is a result, not a voided arm.
The whole band sits below the `1.7651` rig ceiling PERF033 measured at these
exact lane counts, so the expected outcome is PERF035's again: plateau first,
with occupancy short of the rig bound -- encoder-bound, at about four times
the L4's level.

**The rungs.** Offered `0.32 / 0.64 / 0.96 / 1.44` is exactly double
PERF035's ladder. At the measured `1.2413` realized-per-offered ratio the
realized arrivals are about `0.40 / 0.79 / 1.19 / 1.79` decisions per second
-- `0.51x / 1.01x / 1.52x / 2.28x` of the prediction. The anchor rung `r032`
is byte-identical to PERF035's `r032` cell (same seed, same rate, same
digests), so the L4-vs-RTX ratio is readable directly off one shared rung,
the same anchor property PERF033 carried against PERF032. The top rung sits
above even the H100 rig ceiling, so the plateau has somewhere to appear
wherever in the band the card lands.

**The paid wall cannot bind anywhere the prediction can be wrong.** Both arms
together come to about `650` seconds at the predicted ceiling, `750` at the
pessimistic edge, `900` at half the prediction, and `1,585` even if the RTX
PRO 6000 delivers no speedup over the L4 at all -- every case far inside the
`2,400`-second envelope. Only a card slower than the L4 it is predicted to
quadruple would breach it.

**Memory, for the first time in an eight-lane packet, is safe by the derived
cap.** Worst case at eight lanes is `8 * 262,144 = 2,097,152` resident
tokens, 24 GiB at the contract's 12 KiB per token -- more than the whole L4,
which is why PERF035 was only ever safe by corpus construction. Against this
card's roughly `87` GB pool (96 GB at `0.92` utilization, less ~1.6 GB of
weights) the same worst case is under a third, and the frozen corpus's
`12.92` GiB peak is about 15%. The `failed == 0` integrity gate, the exactly
zero `cache_miss_tokens` and `session_actions.rebuilt` requirements, and the
memory-cliff behaviour are all carried unchanged regardless.

`CHUNK_SCHEDULE_TOKENS = 8192` and `enforce_eager = True` are held fixed from
PERF033/034/035. On an architecture generation the family has never measured
they are the first suspects if the ceiling falls below the band.
"""

from __future__ import annotations

from rayline_l4_capacity_contract import (
    H100_TFLOPS,
    L4_TFLOPS,
    PERF033_TOP_COMPLETION_THROUGHPUT_DPS,
)
from rayline_l4_capacity_contract import (
    PERF035 as PERF035_CONTRACT,
)
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

PERF036_RUN_ID = "rayline-rtx6000-capacity-perf036-20260811"

# ---------------------------------------------------------------------------
# HUMAN GATES. Both are fail-closed placeholders. Preparation may not move
# either one; only a reviewed authorization checkpoint may.
# ---------------------------------------------------------------------------

# The Pathfinder head this run's authority is pinned to: the pushed
# codex/rayline-vsr-mvp head whose registry entry records the confirmed
# grant. `_assert_pushed` forces the Pathfinder HEAD to equal this.
PATHFINDER_AUTHORIZATION_COMMIT = "PENDING"

# NOT GRANTED. `AUTHORIZED_CUMULATIVE_USD` is deliberately left where
# PERF035's grant put it. PERF035's grant landed the reserve at exactly the
# `$3.00` floor, so against the `$188.841229466383` conservative position this
# packet's `$5.6186208` RTX PRO 6000 envelope would leave a reserve of
# `-$2.6186208` -- below the floor by the whole envelope -- and
# `budget_receipt` raises `BudgetError`, making a launch arithmetically
# impossible. The minimum viable grant therefore equals the envelope exactly.
# Moving this number is the authorization act; preparation may not do it.
AUTHORIZED_CUMULATIVE_USD = 191.841229466383
PREVIOUS_CONSERVATIVE_USD = 188.841229466383
MINIMUM_VIABLE_GRANT_USD = 5.6186208

# The eight-lane shape, unchanged from PERF033 and PERF035. On this card it is
# a rig choice again rather than a memory constraint -- the 96 GB pool would
# hold the 32-lane corpus -- but the packet holds eight lanes anyway, because
# the cross-GPU curve is only readable if the silicon is the one variable.
EPISODE_LANES = 8
MEASURED_CASES = 32
WARMUP_CASES = 4
MEASURED_EPISODES = 8
WARMUP_EPISODES = 1

# The PERF036 encoder deployment. Same engine build, image, model revision and
# prefill backend as PERF031 through PERF035; a distinct app because the GPU
# class is part of the deployed service and every closed run's evidence names
# apps that must keep their recorded card forever.
PERF036_APP_NAME = "rayline-arc-session-encoder-flashinfer-perf036-rtx6000"
PERF036_ENCODER_GPU = "RTX-PRO-6000"

# ---------------------------------------------------------------------------
# The anchor: the family's first measured cross-GPU prediction.
# ---------------------------------------------------------------------------

# PERF035's measured top completion throughput (r072, ARC arm), the first
# non-H100 ceiling ever recorded, and the anchor this prediction scales from.
PERF035_TOP_COMPLETION_THROUGHPUT_DPS = 0.1977

# Dense-FP16 tensor TFLOPS, the one basis on which all three cards' figures
# are comparable. The RTX PRO 6000 figure is the Server Edition derivation
# (published "1 PFLOP" sparse FP16, halved to dense, clock-scaled 120/126);
# the Workstation Edition's whitepaper number is 503.8, which would move the
# prediction to 0.823 -- inside the band, so the unconfirmed edition cannot
# flip the outcome. H100 and L4 figures are imported from the PERF035
# contract so the three cards share one basis by construction.
RTX6000_TFLOPS = 480.0
PREDICTED_CEILING_DPS = (
    PERF035_TOP_COMPLETION_THROUGHPUT_DPS * RTX6000_TFLOPS / L4_TFLOPS
)

# The band, carried from PERF035 with the same standing: a validation target
# for naive TFLOPS scaling from a measured anchor, and NOT an integrity gate.
# PERF035 falsified the token-model calculator; this run tests the method
# that survived it, which predicted the L4 within 9.2% when run from the
# H100's measured ceiling (the cross-check the test pins).
PREDICTED_CEILING_TOLERANCE = 0.30

# The corpus's own encoder work, unchanged: the sum over the eight measured
# episodes of each episode's final serialized length.
CORPUS_ENCODER_TOKENS = 1_129_231

# Realized-per-offered, measured on the seeded schedules every eight-lane
# packet shares. The anchor cell is the rung PERF035 also ran: byte-identical
# workload and identity digests, so the L4-vs-RTX ratio reads off it directly.
ANCHOR_CELL = "r032"
ANCHOR_REALIZED_PER_OFFERED = 1.2413

# PERF020's recorded `selected_worker_trace_sha256`, identical in every
# 32-case closed run from PERF018 on. PERF036 runs that same corpus, so the
# continuity check is exact equality, as it was for PERF035.
PERF020_TRACE_SHA256 = (
    "d9e93cf0f4c636a3838e41938d2ef3ff6e1d66a60860922f84771b3fa5158ac9"
)

PERF036 = OpenLoopRunContract(
    run_id=PERF036_RUN_ID,
    packet_manifest_sha256=(
        "fccb409b5a6c829cce7994d391a109f6d69cdf0a09539330f77a871a3386e4ec"
    ),
    # Byte-identical to PERF020's, and to every eight-lane packet since. One
    # variable moves in this packet and it is the card.
    corpus_sha256=PERF020.corpus_sha256,
    topology_sha256=PERF020.topology_sha256,
    cells=(
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
            label="r064",
            offered_rate_rps=0.64,
            workload_sha256=(
                "9b73d905c9e9746680c2ad812b1cd6d96dde6b02522aa03a2e185cb513b0bcaf"
            ),
            identity_sha256=(
                "380f9844dea77b03834b00ac0f1c29da16c37141e63e58bacb115528332791ae"
            ),
        ),
        OpenLoopCell(
            label="r096",
            offered_rate_rps=0.96,
            workload_sha256=(
                "888a10fb8c34c52c05c5bfa1ab92ae4a886b6a22585f44480095b76aee8aff1b"
            ),
            identity_sha256=(
                "7a71aa93c31e750df2ba8b537630d8551853f81deb61459a2873e96f8f98a2eb"
            ),
        ),
        OpenLoopCell(
            label="r144",
            offered_rate_rps=1.44,
            workload_sha256=(
                "c6e5b8b41333b864c01ae68c074d73a3dd8a2472d50c2eaf32bd7fb3c6bddab8"
            ),
            identity_sha256=(
                "11ecde795f8f3492cd7dbd84bdc5f913ea0a4caab35bf5df792f95388121695b"
            ),
        ),
    ),
    compose_project_prefix="rayline-rtx6000-capacity-perf036",
    temporary_prefix="rayline-perf036-",
    budget=BudgetContract(
        run_id=PERF036_RUN_ID,
        previous_conservative_usd=PREVIOUS_CONSERVATIVE_USD,
        authorized_cumulative_usd=AUTHORIZED_CUMULATIVE_USD,
        # A real bound on an RTX PRO 6000 packet: the envelope is
        # `$5.6186208`, so `$6.00` binds where PERF034's inherited `$7.00`
        # would not.
        packet_ceiling_usd=6.0,
        required_reserve_usd=3.0,
        # The unchanged 40-minute paid-wall envelope. The container keeps its
        # 8 cores and 64 GiB so the run stays comparable to the H100 and L4
        # ones on everything except the card; only the GPU line item changes.
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_gpu=PERF036_ENCODER_GPU,
    ),
    encoder_app_name=PERF036_APP_NAME,
    encoder_build_id=FLASHINFER_BUILD_ID,
    encoder_gdn_prefill_backend="flashinfer",
    encoder_gpu=PERF036_ENCODER_GPU,
    measured_cases=MEASURED_CASES,
    warmup_cases=WARMUP_CASES,
    measured_episodes=MEASURED_EPISODES,
    warmup_episodes=WARMUP_EPISODES,
    # Both firing points armed, with PERF034's calibration unchanged. The
    # whole prediction band sits below the 1.7651 dps rig ceiling PERF033
    # measured at these lane counts, so the expected outcome is PERF035's:
    # the plateau criterion fires with occupancy short of the rig bound,
    # which is the direct signature of an encoder-bound ceiling.
    saturation=SaturationCriterion(
        episode_lanes=EPISODE_LANES,
        occupancy_ratio=1.0,
        throughput_plateau_gain=1 / 3,
    ),
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

RTX6000_CAPACITY_ARMS = (PERF036,)

# Deliberately None: prepared is not launchable. Binding this name to PERF036
# is the launch-authorization act and needs the registry entry's confirmed
# grant plus the Pathfinder pin above replaced with a real pushed head.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = None

# Re-exported so the cross-GPU sanity check the pin test walks through stays
# importable from one place.
__all__ = [
    "ANCHOR_CELL",
    "ANCHOR_REALIZED_PER_OFFERED",
    "AUTHORIZED_CUMULATIVE_USD",
    "CORPUS_ENCODER_TOKENS",
    "EPISODE_LANES",
    "H100_TFLOPS",
    "L4_TFLOPS",
    "LAUNCHABLE_CONTRACT",
    "MINIMUM_VIABLE_GRANT_USD",
    "PERF020_TRACE_SHA256",
    "PERF033_TOP_COMPLETION_THROUGHPUT_DPS",
    "PERF035_CONTRACT",
    "PERF035_TOP_COMPLETION_THROUGHPUT_DPS",
    "PERF036",
    "PERF036_APP_NAME",
    "PERF036_ENCODER_GPU",
    "PERF036_RUN_ID",
    "PREDICTED_CEILING_DPS",
    "PREDICTED_CEILING_TOLERANCE",
    "PREVIOUS_CONSERVATIVE_USD",
    "RTX6000_CAPACITY_ARMS",
    "RTX6000_TFLOPS",
    "resolve_launch_contract",
]


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline RTX PRO 6000 capacity arm is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
