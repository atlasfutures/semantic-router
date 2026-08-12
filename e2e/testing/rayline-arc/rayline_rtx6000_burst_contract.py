#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepared PERF037 32-lane burst-absorption packet on a Modal RTX PRO 6000.

Three packets in a row ended the same way. PERF033 pinned eight lanes and
found the rig, not the encoder. PERF034 raised the rig to 32 lanes -- every
episode the corpus has -- and both criteria fired at the same rung, which the
preregistration had reserved for "the corpus ran out of lanes again". PERF036
moved the card to the RTX PRO 6000 and its `r144` plateau was voided by the
drain clause, occupancy-first, a third time. Each of those runs was asking
where the encoder saturates. PERF037 asks a different question, because a
deployment decision is waiting on a different question.

**What PERF037 asks.** The worst recorded production burst is `2.33` decisions
per second. PERF036 measured this card completing `0.8877` at eight lanes.
Does the deployment absorb that burst, and if so at what lane count? The
answer decides whether single-instance Cloud Run is committed to, or whether
queueing during bursts is accepted explicitly.

**Why this design does not repeat the failure.** The drain clause voided three
plateau verdicts because a finite corpus makes `completion_throughput_rps =
completed / (span + drain)` roll off with rate even under zero queueing, and
the *marginal gain between rungs* inherits that artifact. PERF037's verdict
does not use marginal gain. Absorption is decided on quantities the receipts
measure directly -- realized arrival rate, completion throughput, peak
backlog, and the integrity counters -- so the drain arithmetic has nothing to
corrupt. The clause stays armed over the plateau verdict, which is carried
unchanged from PERF034 as non-voting corroboration, and it is preregistered
here that it may void that and nothing else.

More than that: for this question the finite corpus is the instrument rather
than the flaw. A production burst *is* a finite pulse. Each cell offers 128
decisions at a Poisson rate and then stops, which is a burst of `128 / rate`
seconds, and what the run measures is exactly what a burst asks of a
deployment -- how far behind it falls, and how long it takes to catch up.

**The prediction, and its basis.** The anchor is PERF034's measured 32-lane
H100 ceiling, `2.3061533124360074` decisions per second at `r645`, scaled by
the same dense-FP16 TFLOPS ratio PERF036 validated: `2.3062 * 480/989 =
1.1193`. The `+/-30%` band is `0.7835 .. 1.4550`, carried as
`PREDICTED_CEILING_TOLERANCE` with PERF036's exact standing -- a validation
target for naive TFLOPS scaling, explicitly not an integrity gate.

The method now has a same-corpus, same-lane-count validation it did not have
when PERF036 ran: scaling PERF033's measured eight-lane H100 ceiling by the
same ratio predicts `1.7651 * 480/989 = 0.8567` for the eight-lane RTX PRO
6000, and PERF036 measured `0.8877` -- `3.5%` low. A second route to the same
prediction, scaling PERF036's measured eight-lane RTX ceiling by the H100's
own 8-to-32-lane gain, gives `1.1598`. It is not independent -- it differs
from the TFLOPS route by exactly that `3.5%` residual, algebraically -- but it
sits inside the band, so the choice of route cannot flip the outcome, the same
way PERF036's Server-versus-Workstation TFLOPS ambiguity could not.

**The predicted answer is no, at every lane count this corpus can express.**
`2.33` is `2.08x` the point prediction and `1.60x` the top of the band, and 32
lanes is every episode the corpus has, so no wider packet over this corpus can
exist. Falsifying that needs a measured ceiling of `2.33` or better -- more
than double the prediction, far outside the band. What the run then converts
the measurement into is the number the decision actually needs:
`absorbable_burst_seconds`, how long a `2.33` burst may last before recovery
exceeds `RECOVERY_BUDGET_SECONDS`. At the prediction that is about `28`
seconds; across the band, `15` to `50`.

**Only the card moves.** Every digest here is imported from the PERF034
contract rather than retyped, so the corpus, the topology, the manifest and
all four cells are byte-identical to the packet PERF034 ran by construction --
the packet on disk at `.agent-harness/rayline-parity/packet-perf034`, whose
eleven digests were re-verified at preparation time. Every rung is therefore a
paired cross-GPU comparison, not just the anchor.

**Memory is safe by corpus construction, not by the derived cap.** Worst case
at 32 lanes is `8,388,608` resident tokens, 96 GiB at the contract's 12 KiB
per token, against this card's roughly `81` GiB pool -- it does not fit, as it
did not fit the H100's `67` GiB. The frozen corpus peaks at `4,261,735` tokens
(`48.77` GiB), `60%` of this pool where it was `73%` of the pool that already
ran it clean, and PERF034's recorded peak retained was `1,205,793` tokens. So
this packet has strictly more headroom than the run that produced its anchor.
The `failed == 0` gate, the exactly zero `cache_miss_tokens` and
`session_actions.rebuilt` requirements, and the memory-cliff behaviour are
carried unchanged.

**The wall is the one real schedule risk.** PERF034's eight cells spanned 845
recorded seconds against a formula estimate of `708.5`, so the formula runs
about `1.19x` optimistic on a 32-lane run. Applied to this card at the band
floor the estimate is about `1,809` seconds plus a `~133`-second image deploy,
inside the 2,400-second wall with roughly 19% to spare -- thinner than PERF036
had. The wall binds only below about `0.61` decisions per second, which is
below even PERF036's measured eight-lane figure and would mean 32 lanes ran
slower than eight.

`CHUNK_SCHEDULE_TOKENS = 8192` and `enforce_eager = True` are held fixed from
PERF033/034/035/036.
"""

from __future__ import annotations

from rayline_l4_capacity_contract import (
    H100_TFLOPS,
    L4_TFLOPS,
    PERF033_TOP_COMPLETION_THROUGHPUT_DPS,
)
from rayline_open_loop_contract import (
    MAXIMUM_ORPHAN_REQUEST_SECONDS,
    MAXIMUM_PAID_WALL_SECONDS,
    MAXIMUM_SCALEDOWN_SECONDS,
    PERF020,
    OpenLoopRunContract,
    SaturationCriterion,
)
from rayline_rtx6000_capacity_contract import PERF036 as PERF036_CONTRACT
from rayline_rtx6000_capacity_contract import RTX6000_TFLOPS
from rayline_saturation_capacity_contract import PERF034 as PERF034_CONTRACT
from rayline_saturation_knee_contract import FLASHINFER_BUILD_ID
from rayline_three_arm_budget import BudgetContract

PERF037_RUN_ID = "rayline-rtx6000-burst-perf037-20260812"

# ---------------------------------------------------------------------------
# HUMAN GATES. Both are fail-closed placeholders. Preparation may not move
# either one; only a reviewed authorization checkpoint may.
# ---------------------------------------------------------------------------

# The Pathfinder head this run's authority is pinned to: the pushed
# codex/rayline-vsr-mvp head whose registry entry records the confirmed
# grant. `_assert_pushed` forces the Pathfinder HEAD to equal this, and the
# literal below can never equal a commit.
PATHFINDER_AUTHORIZATION_COMMIT = "PENDING"

# NOT GRANTED. The authority still reads where PERF036's grant left it, and
# PERF036 closed on the reserve floor exactly: `$194.459850266383` conservative
# under a `$197.459850266383` ceiling is `$3.00`, all of it reserve. This
# packet's `$5.6186208` envelope therefore breaches the floor by exactly
# itself, and `budget_receipt` refuses. The minimum viable grant is the whole
# envelope -- there is no partial headroom to draw on first -- which takes the
# ceiling to `$203.078471066383` and leaves the reserve at exactly `$3.00`
# again. Only a human may move it, at a reviewed authorization checkpoint.
AUTHORIZED_CUMULATIVE_USD = 197.459850266383
PREVIOUS_CONSERVATIVE_USD = 194.459850266383
MINIMUM_VIABLE_GRANT_USD = 5.6186208
GRANTED_CUMULATIVE_WOULD_BE_USD = 203.078471066383

# PERF034's shape, because the burst question needs the widest rig the frozen
# corpus can express. 32 lanes is every episode the corpus has, so this is not
# a rung on a ladder of lane counts -- it is the end of that ladder.
EPISODE_LANES = 32
MEASURED_CASES = 128
WARMUP_CASES = 8
MEASURED_EPISODES = 32
WARMUP_EPISODES = 2

# The PERF037 encoder deployment. Same engine build, image, model revision and
# prefill backend as PERF031 through PERF036, and the first app to carry two
# earlier packets' deviations at once: PERF036's card and PERF034's 32/64 caps.
PERF037_APP_NAME = "rayline-arc-session-encoder-flashinfer-perf037-rtx6000-32lane"
PERF037_ENCODER_GPU = "RTX-PRO-6000"

# ---------------------------------------------------------------------------
# The question: a production burst, and what absorbing it would mean.
# ---------------------------------------------------------------------------

# The worst burst production has recorded, in decisions per second. It is the
# figure the PERF036 handoff carries as the reason burst absorption cannot be
# claimed: it already exceeds the `0.8877` eight-lane floor this card measured.
PRODUCTION_BURST_DPS = 2.33

# How long a backlog may take to clear before the burst counts as unabsorbed.
# Thirty seconds is a decision input, not a measurement: it is the recovery
# tail a single-instance deployment is willing to show. Changing it changes
# `absorbable_burst_seconds` and nothing else, which is why it is named here
# rather than buried in the arithmetic.
RECOVERY_BUDGET_SECONDS = 30.0

# A cell absorbs its burst only if it kept up. `0.95` allows the measurement
# noise a single run carries and nothing more; below that the deployment is
# falling behind, which is the definition of not absorbing.
ABSORPTION_COMPLETION_RATIO_FLOOR = 0.95

# The drain clause's scope, preregistered rather than left to analysis-time
# judgement. It voided three plateau verdicts in a row because marginal gain
# between rungs inherits the finite corpus's drain artifact. Absorption and the
# ceiling are read off directly measured quantities within a single cell, so
# the artifact has nothing to corrupt there, and the clause may not reach them.
DRAIN_CLAUSE_VOIDS_PLATEAU_VERDICT = True
DRAIN_CLAUSE_VOIDS_ABSORPTION_VERDICT = False
DRAIN_CLAUSE_VOIDS_CEILING_MEASUREMENT = False

# ---------------------------------------------------------------------------
# The anchor: PERF034's measured 32-lane ceiling, and the cross-checks on it.
# ---------------------------------------------------------------------------

# PERF034's measured top completion throughput (`r645`, ARC arm) -- the only
# 32-lane figure this family has, and the anchor this prediction scales from.
# It is qualified: PERF034's own anchor rung came in 7.15% under PERF033's
# recorded throughput with no tolerance preregistered, so the gate is recorded
# as not held and the absolute scale of every PERF034 figure inherits that.
PERF034_TOP_COMPLETION_THROUGHPUT_DPS = 2.3061533124360074
PERF034_REMOTE_TOP_COMPLETION_THROUGHPUT_DPS = 2.1262018706449126

# PERF036's measured eight-lane ceiling on this exact card. It is what makes
# the prediction checkable before the run: the same TFLOPS ratio applied to
# PERF033's eight-lane H100 figure predicts this number within 3.5%.
PERF036_TOP_COMPLETION_THROUGHPUT_DPS = 0.8877

PREDICTED_CEILING_DPS = (
    PERF034_TOP_COMPLETION_THROUGHPUT_DPS * RTX6000_TFLOPS / H100_TFLOPS
)

# The band, carried from PERF036 with the same standing: a validation target
# for naive TFLOPS scaling from a measured anchor, and NOT an integrity gate.
# A ceiling outside it falsifies the method at a second lane count, which is a
# result rather than a voided arm.
PREDICTED_CEILING_TOLERANCE = 0.30

# The corpus's own encoder work: the sum over the 32 measured episodes of each
# episode's final serialized length. Verified against
# `.agent-harness/rayline-parity/packet-perf034/corpus.json` at preparation
# time, and it is the figure PERF034 preregistered as the memory peak.
CORPUS_ENCODER_TOKENS = 4_261_735

# PERF034's measured realized-per-offered ratio, which is the one design input
# that moved under it: the ladder was built for `1.2413` and the seeded
# schedules delivered `1.0432` at 32 lanes. Reusing PERF034's cells byte for
# byte means reusing that measured ratio, so the realized rungs are known in
# advance rather than designed.
ANCHOR_CELL = "r120"
ANCHOR_REALIZED_PER_OFFERED = 1.0432
PERF034_ANCHOR_COMPLETION_THROUGHPUT_DPS = 1.0721298521209137

# What PERF034 recorded at the one rung whose realized arrivals exceed the
# production burst. Carried so the absorption predicate can be replayed
# against a real receipt rather than only against a prediction: on an H100 at
# 32 lanes, this rung did not absorb, which is the calibration that says the
# predicate is not trivially satisfiable.
PERF034_BURST_CELL = "r240"
PERF034_BURST_REALIZED_DPS = 2.5036
PERF034_BURST_COMPLETION_DPS = 1.3381459770827528
PERF034_BURST_PEAK_BACKLOG = 33

# PERF034's recorded schedule, for the wall calibration the docstring states.
# The eight cells spanned this many seconds of receipts, against the estimate
# the shared fit formula produces at PERF034's own measured ceiling.
PERF034_RECEIPT_SPAN_SECONDS = 845.0
PERF034_IMAGE_DEPLOY_SECONDS = 133.0

# The continuity digest every 32-case closed run from PERF018 on recorded.
# PERF034 could not check it -- the probe hashes the selected-worker trace
# without persisting its entries, so the 32-case prefix is not recomputable
# from any receipt -- and PERF037 inherits that gap unchanged, because it runs
# PERF034's corpus and changes nothing about the probe.
PERF020_TRACE_PREFIX_SHA256 = (
    "d9e93cf0f4c636a3838e41938d2ef3ff6e1d66a60860922f84771b3fa5158ac9"
)
TRACE_PREFIX_CHECK_IS_RUNNABLE = False

# What teardown must show before this run may be called closed. The launcher's
# `finally` path does the first four; the fifth is a separate instrument, and
# it is listed because it is the one an agent is most likely to skip -- PERF036
# ran it and found zero, which is why its cleanup claim stands on two
# independent readings rather than the launcher's own.
TEARDOWN_REQUIREMENTS = (
    "the launcher stops the encoder app and deletes the proxy token",
    "the run manifest records encoder_containers_remaining exactly 0",
    "every cell's state reset closes 32/32 measured sessions with exact zeros",
    "every local compose project is removed -- cell_cleanup all true",
    "an independent `modal container list` in the dev environment returns empty",
)


def absorbs_burst(
    realized_dps: float,
    completion_dps: float,
    peak_backlog: int,
    *,
    lanes: int = EPISODE_LANES,
    burst_dps: float = PRODUCTION_BURST_DPS,
    completion_ratio_floor: float = ABSORPTION_COMPLETION_RATIO_FLOOR,
) -> bool:
    """Whether one cell absorbed a burst at least as hard as production's.

    Three conditions, all read straight off the cell's own receipt. The cell
    must actually have offered a burst worth the name; it must have kept up
    with it; and the rig must never have gone over-full, because a backlog
    past the lane count is work waiting behind a saturated encoder rather than
    work in flight. The integrity gates (`failed == 0`, zero
    `cache_miss_tokens`, zero `session_actions.rebuilt`, zero provider calls)
    are conditions on the whole arm, not on one cell, so they are not repeated
    here -- an arm that breaks them is void and has no cells to judge.
    """

    return (
        realized_dps >= burst_dps
        and completion_dps >= completion_ratio_floor * realized_dps
        and peak_backlog <= lanes
    )


def absorbable_burst_seconds(
    ceiling_dps: float,
    *,
    burst_dps: float = PRODUCTION_BURST_DPS,
    recovery_budget_seconds: float = RECOVERY_BUDGET_SECONDS,
) -> float:
    """How long a `burst_dps` burst may last before recovery exceeds budget.

    The deployment decision is not "is the burst absorbed" in the abstract --
    a burst is transient, so a deployment slower than the burst still absorbs
    a short one by queueing. Backlog accumulates at `burst - ceiling` and
    clears at `ceiling`, so a burst of `T` seconds needs
    `(burst - ceiling) * T / ceiling` seconds to recover. This inverts that
    for the budget. A deployment at or above the burst rate never falls
    behind, and returns infinity.
    """

    if ceiling_dps <= 0.0:
        raise ValueError("a ceiling must be positive to absorb anything")
    if ceiling_dps >= burst_dps:
        return float("inf")
    return recovery_budget_seconds * ceiling_dps / (burst_dps - ceiling_dps)


PERF037 = OpenLoopRunContract(
    run_id=PERF037_RUN_ID,
    # Every digest is PERF034's own, imported rather than retyped, so
    # byte-identity is structural: the packet this contract admits is the
    # packet PERF034 ran, and no transcription can make it otherwise.
    packet_manifest_sha256=PERF034_CONTRACT.packet_manifest_sha256,
    corpus_sha256=PERF034_CONTRACT.corpus_sha256,
    topology_sha256=PERF034_CONTRACT.topology_sha256,
    cells=PERF034_CONTRACT.cells,
    compose_project_prefix="rayline-rtx6000-burst-perf037",
    temporary_prefix="rayline-perf037-",
    budget=BudgetContract(
        run_id=PERF037_RUN_ID,
        previous_conservative_usd=PREVIOUS_CONSERVATIVE_USD,
        authorized_cumulative_usd=AUTHORIZED_CUMULATIVE_USD,
        # PERF036's real bound on an RTX PRO 6000 packet, unchanged: the
        # envelope is `$5.6186208`, so `$6.00` binds.
        packet_ceiling_usd=6.0,
        required_reserve_usd=3.0,
        # The unchanged 40-minute paid-wall envelope. The container keeps its
        # 8 cores and 64 GiB, so only the GPU line item differs from PERF034
        # and nothing differs from PERF036.
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_gpu=PERF037_ENCODER_GPU,
    ),
    encoder_app_name=PERF037_APP_NAME,
    encoder_build_id=FLASHINFER_BUILD_ID,
    encoder_gdn_prefill_backend="flashinfer",
    encoder_gpu=PERF037_ENCODER_GPU,
    measured_cases=MEASURED_CASES,
    warmup_cases=WARMUP_CASES,
    measured_episodes=MEASURED_EPISODES,
    warmup_episodes=WARMUP_EPISODES,
    # Carried from PERF034 unchanged, and demoted to corroboration. Both
    # criteria fired at the same rung there, and the plateau verdict is the one
    # the drain clause is allowed to void; the absorption verdict is decided
    # elsewhere and is not reached by either.
    saturation=SaturationCriterion(
        episode_lanes=EPISODE_LANES,
        occupancy_ratio=1.0,
        throughput_plateau_gain=1 / 3,
    ),
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

RTX6000_BURST_ARMS = (PERF037,)

# Prepared, not launchable. Only a reviewed authorization checkpoint, against
# a fresh human grant and a Pathfinder registry entry recording it, may bind
# this to the contract above.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = None

__all__ = [
    "ABSORPTION_COMPLETION_RATIO_FLOOR",
    "ANCHOR_CELL",
    "ANCHOR_REALIZED_PER_OFFERED",
    "AUTHORIZED_CUMULATIVE_USD",
    "CORPUS_ENCODER_TOKENS",
    "DRAIN_CLAUSE_VOIDS_ABSORPTION_VERDICT",
    "DRAIN_CLAUSE_VOIDS_CEILING_MEASUREMENT",
    "DRAIN_CLAUSE_VOIDS_PLATEAU_VERDICT",
    "EPISODE_LANES",
    "GRANTED_CUMULATIVE_WOULD_BE_USD",
    "H100_TFLOPS",
    "L4_TFLOPS",
    "LAUNCHABLE_CONTRACT",
    "MINIMUM_VIABLE_GRANT_USD",
    "PERF020",
    "PERF020_TRACE_PREFIX_SHA256",
    "PERF033_TOP_COMPLETION_THROUGHPUT_DPS",
    "PERF034_ANCHOR_COMPLETION_THROUGHPUT_DPS",
    "PERF034_BURST_CELL",
    "PERF034_BURST_COMPLETION_DPS",
    "PERF034_BURST_PEAK_BACKLOG",
    "PERF034_BURST_REALIZED_DPS",
    "PERF034_CONTRACT",
    "PERF034_IMAGE_DEPLOY_SECONDS",
    "PERF034_RECEIPT_SPAN_SECONDS",
    "PERF034_REMOTE_TOP_COMPLETION_THROUGHPUT_DPS",
    "PERF034_TOP_COMPLETION_THROUGHPUT_DPS",
    "PERF036_CONTRACT",
    "PERF036_TOP_COMPLETION_THROUGHPUT_DPS",
    "PERF037",
    "PERF037_APP_NAME",
    "PERF037_ENCODER_GPU",
    "PERF037_RUN_ID",
    "PREDICTED_CEILING_DPS",
    "PREDICTED_CEILING_TOLERANCE",
    "PREVIOUS_CONSERVATIVE_USD",
    "PRODUCTION_BURST_DPS",
    "RECOVERY_BUDGET_SECONDS",
    "RTX6000_BURST_ARMS",
    "RTX6000_TFLOPS",
    "TEARDOWN_REQUIREMENTS",
    "TRACE_PREFIX_CHECK_IS_RUNNABLE",
    "absorbable_burst_seconds",
    "absorbs_burst",
    "resolve_launch_contract",
]


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline RTX PRO 6000 burst arm is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
