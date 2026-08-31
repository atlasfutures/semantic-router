#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepared PERF034 32-lane FlashInfer saturation-capacity packet.

PERF033 closed the eight-lane question: the occupancy criterion fired at
`r160` on both sub-arms, exactly as preregistered, and the anchor reproduced.
What it also proved is that eight lanes measure the rig, not the encoder --
throughput was still climbing 13% per rung at the top of the ladder with every
lane pinned, and the derived GPU-busy fraction stood at `0.53`. The encoder's
own capacity has never been reached by any packet this repo has recorded.

PERF034 removes the rig's ceiling instead of inferring past it. Three things
change together, and nothing else does.

- **32 lanes.** The full 128-case directional corpus carries 32 episodes; the
  probe runs one thread per episode; the PERF034 encoder profile raises
  `MAX_SESSIONS` to 32 and `MAX_CONCURRENT_INPUTS` to 64 to match. The lane
  count stops being the binding capacity by construction.
- **A ladder that reaches the predicted knee.** Realized targets of roughly
  `1.5 / 3 / 5 / 8` decisions per second at the measured `1.2413`
  realized-per-offered ratio give offered rungs `1.20 / 2.40 / 4.00 / 6.45`.
- **A second firing point.** The predicted knee sits at 15-20 lanes -- an
  occupancy near `0.5` -- so the occupancy criterion alone would stay
  correctly silent while the encoder saturates. The contract therefore also
  arms `throughput_plateau_gain`: the diagnosis is in which criterion fires
  first. Occupancy first means the rig bound again; plateau first, with
  occupancy short of the ceiling, is direct proof the encoder bound.

Memory is preregistered as a cliff, not a curve. Worst case at 32 lanes is
8,388,608 resident tokens (96 GiB) against a ~70 GiB pool, but the frozen
corpus peaks at 4,261,735 tokens -- 51% of the derived cap -- so the packet is
safe by corpus construction, not by headroom at the limit. If memory does
bind, the coordinator's `SessionCapacityError` becomes a failed case and the
`failed == 0` integrity gate voids the arm; `cache_miss_tokens` and
`session_actions.rebuilt` must stay exactly zero, as they have in every cell
ever recorded. `CHUNK_SCHEDULE_TOKENS = 8192` and `enforce_eager = True` are
held fixed from PERF033, and are the first suspects if throughput plateaus
below the predicted knee.
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

PERF034_RUN_ID = "rayline-saturation-capacity-perf034-20260811"

# ---------------------------------------------------------------------------
# HUMAN GATES. Both are fail-closed placeholders. Preparation may not move
# either one; only a reviewed authorization checkpoint may.
# ---------------------------------------------------------------------------

# The Pathfinder head this run's authority is pinned to: the pushed
# codex/rayline-vsr-mvp head whose registry entry records the confirmed
# grant. `_assert_pushed` forces the Pathfinder HEAD to equal this.
PATHFINDER_AUTHORIZATION_COMMIT = "da85e1045a92aa3d6aa6d765a2dc2f5257e1d31d"

# GRANTED 2026-08-11. The user approved the spend without naming a figure,
# so the ceiling moves by exactly the minimum viable grant recorded in
# authorize commit e282f16c: `$184.31282402` plus `$5.108984646383`. Against
# the `$179.487387866383` conservative position and the unchanged
# `$6.9344208` envelope, this leaves the reserve at exactly the `$3.00`
# floor.
AUTHORIZED_CUMULATIVE_USD = 189.421808666383
PREVIOUS_CONSERVATIVE_USD = 179.487387866383
MINIMUM_VIABLE_GRANT_USD = 5.108984646383

# The lane ceiling this packet raises, and why 32 is the whole corpus. The
# directional corpus has 32 episodes of four decisions each; the probe runs
# one thread per episode; the PERF034 encoder profile in
# `modal_session_service.py` raises `MAX_SESSIONS` to the same 32. A packet
# cannot go wider than the corpus has episodes, so 32 lanes is the largest
# rig this corpus can ever express.
EPISODE_LANES = 32
MEASURED_CASES = 128
WARMUP_CASES = 8
MEASURED_EPISODES = 32
WARMUP_EPISODES = 2

# The PERF034 encoder deployment. Same engine build as PERF031/032/033's
# FlashInfer app -- the image, model revision and prefill backend are
# unchanged -- but a distinct app, because the session-service caps are part
# of the deployed service and the cap raise must not touch the app the closed
# runs' evidence names.
PERF034_APP_NAME = "rayline-arc-session-encoder-flashinfer-perf034"

# The anchor. `r120` re-offers PERF033's anchor rate, but unlike every prior
# anchor it is NOT byte-identical to its predecessor cell: a rung's workload
# document carries the lane count and case counts, so the 32-lane `r120`
# necessarily differs from the 8-lane one. The anchor property is therefore a
# measured quantity, not a digest: at 1.20 offered, an unconstrained encoder
# must reproduce PERF033's completion throughput, because PERF033 showed
# eight lanes were already enough to carry this rate.
ANCHOR_CELL = "r120"
ANCHOR_OFFERED_RATE_RPS = 1.20
ANCHOR_REALIZED_PER_OFFERED = 1.2413
PERF033_ANCHOR_COMPLETION_THROUGHPUT_DPS = 1.1547543726851863

# Where the knee is predicted, and how weakly. The derived GPU-busy fraction
# at eight fully saturated lanes was `0.53`, which extrapolates to a compute
# knee near 15 lanes -- occupancy `~0.5` of this packet's 32 -- with about 2x
# uncertainty either way. If occupancy pins at `1.0` with throughput still
# climbing, the corpus has run out of lanes again and that outcome, not a
# wider packet, is the recorded result.
PREDICTED_KNEE_LANES = 15

# PERF020's recorded `selected_worker_trace_sha256`, identical in every
# 32-case closed run from PERF018 through PERF033. The 128-case corpus routes
# a superset of those cases, so the continuity check at analysis time is a
# prefix property: the sha256 of the new trace's first 32
# `[case_id, canonical]` entries, serialised exactly as `_render_receipt`
# serialises the full trace, must equal this. The digest is a recorded value
# -- corpus cases carry no worker field, so it cannot be recomputed from the
# packet and is carried here as the constant it is.
PERF020_TRACE_PREFIX_SHA256 = (
    "d9e93cf0f4c636a3838e41938d2ef3ff6e1d66a60860922f84771b3fa5158ac9"
)

PERF034 = OpenLoopRunContract(
    run_id=PERF034_RUN_ID,
    packet_manifest_sha256=(
        "b579a2e0124580d999c7cb7ad955396624b5ca2b690aeba97683d65e9b182b47"
    ),
    # The full 128-case directional corpus, byte-identical to the source
    # packet's -- the 32-case subset every prior open-loop packet carried is a
    # selection from this corpus, not a different one. The topology is
    # unchanged from PERF020: the same canonical workers over a single
    # encoder.
    corpus_sha256=(
        "5e4edcbcb44be32818f9b8e855e38a5d84e3b3a8358781ce4d228e6266ce54f3"
    ),
    topology_sha256=PERF020.topology_sha256,
    cells=(
        OpenLoopCell(
            label="r120",
            offered_rate_rps=1.20,
            workload_sha256=(
                "3f517a8b803e14b97a28a1e24dd18586b12b62cddfe95e9c48fe7d70117c1c08"
            ),
            identity_sha256=(
                "594dfc88477fb4d1d1d73edfd0200629c68c62e913363d857d95242e43467a33"
            ),
        ),
        OpenLoopCell(
            label="r240",
            offered_rate_rps=2.40,
            workload_sha256=(
                "f15a9c17690c46c6e314bf56231ad8a7f1c2f73bf094c33e464c87f4af49ccd4"
            ),
            identity_sha256=(
                "486e799cc7698f784455c0d763078963a17b6593f46b452bb8d8ccaaa56498a4"
            ),
        ),
        OpenLoopCell(
            label="r400",
            offered_rate_rps=4.00,
            workload_sha256=(
                "fccf3d9f39be61843a4cd8c01de5d118b26df0de0bc8d8def99fc923c5cb2fec"
            ),
            identity_sha256=(
                "92a8f2594002cead080bd40ba3c32ed42ccdfb89465bb1a3f94d42669c469cff"
            ),
        ),
        OpenLoopCell(
            label="r645",
            offered_rate_rps=6.45,
            workload_sha256=(
                "c5498c65aa7a9e669d84033339891ea887121352161bccd3c9492bf620745178"
            ),
            identity_sha256=(
                "56b11187f73f05a1e9e8eac6e641baa3eedc4b09b8e2cc347ee3eac14b408e06"
            ),
        ),
    ),
    compose_project_prefix="rayline-saturation-capacity-perf034",
    temporary_prefix="rayline-perf034-",
    budget=BudgetContract(
        run_id=PERF034_RUN_ID,
        previous_conservative_usd=PREVIOUS_CONSERVATIVE_USD,
        authorized_cumulative_usd=AUTHORIZED_CUMULATIVE_USD,
        packet_ceiling_usd=7.0,
        required_reserve_usd=3.0,
        # The unchanged 40-minute paid-wall envelope. The container keeps its
        # 8 cores and 64 GiB -- the cap raise moves session-service constants,
        # not resources -- so the envelope stays `$6.9344208` and four rungs
        # still fit: the slowest arrival schedule here spans under two
        # minutes.
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
    ),
    encoder_app_name=PERF034_APP_NAME,
    encoder_build_id=FLASHINFER_BUILD_ID,
    encoder_gdn_prefill_backend="flashinfer",
    measured_cases=MEASURED_CASES,
    warmup_cases=WARMUP_CASES,
    measured_episodes=MEASURED_EPISODES,
    warmup_episodes=WARMUP_EPISODES,
    # Both firing points armed. `occupancy_ratio = 1.0` keeps the physical
    # rig-ceiling definition PERF033 validated. `throughput_plateau_gain` is
    # one third: on every recorded receipt, unqueued cells convert at least
    # `0.46` of additional realized arrivals into completed throughput while
    # rungs past a known capacity knee convert at most `0.32`, so the floor
    # sits inside the measured gap with margin on both sides rather than at
    # the intuitive `0.5`, which PERF032's unqueued top rung already crosses
    # on drain arithmetic alone.
    saturation=SaturationCriterion(
        episode_lanes=EPISODE_LANES,
        occupancy_ratio=1.0,
        throughput_plateau_gain=1 / 3,
    ),
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

SATURATION_CAPACITY_ARMS = (PERF034,)

# Closed 2026-08-11 after the one authorized execution. All four cells
# measured cleanly (failed=0, traces match); the run then raised
# StateResetError on an HTTP 502 during the final post-r645 state reset,
# which under the registry's no-retry clause closes this ID for good.
# The pin above stays as the record of what was measured.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = None


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError(
            "no Rayline saturation capacity arm is currently launchable"
        )
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
