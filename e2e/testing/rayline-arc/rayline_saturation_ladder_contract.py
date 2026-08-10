#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen PERF031 two-arm single-encoder saturation ladder.

Every saturation measurement in existence (PERF015 through PERF027) ran on the
`torch_reference` GDN prefill backend, and every FlashInfer measurement
(PERF030, AGT017/018/019) is strictly serial and states that it does not
establish concurrency saturation throughput. PERF031 closes that gap with two
sequential runs over the frozen PERF021 packet whose only variable is the GDN
backend:

- PERF031A is the negative control. It deploys the DEFAULT encoder app, not a
  `-reference-perf031` profile, because registering a profile would stamp the
  engine build id `...+gdn-torch-reference-eager` while PERF021's recorded
  `source.engine_build_id` is the bare `vllm@9f5ea81c...`. A profiled control
  would not be identity-matched to the run it exists to reproduce.
- PERF031B is the FlashInfer treatment on its own confined app.

Nothing here regenerates a corpus, a workload, a topology or a rung. The
packet digests and the r015/r030/r045 rungs are PERF021's, reused verbatim, so
PERF031A's only admissible outcome is that it reproduces PERF021's knee.
"""

from __future__ import annotations

from rayline_open_loop_contract import (
    MAXIMUM_ORPHAN_REQUEST_SECONDS,
    MAXIMUM_PAID_WALL_SECONDS,
    MAXIMUM_SCALEDOWN_SECONDS,
    PERF021,
    OpenLoopRunContract,
)
from rayline_three_arm_budget import BudgetContract
from rayline_three_arm_contract import IDENTITY

PERF031A_RUN_ID = "rayline-saturation-ladder-perf031a-20260810"
PERF031B_RUN_ID = "rayline-saturation-ladder-perf031b-20260810"

# Pathfinder head both arms run against, pinned at the authorization
# checkpoint. This is `origin/codex/rayline-vsr-mvp`, which `_assert_pushed`
# forces HEAD to equal, so it cannot be pinned to PERF021's older head.
#
# The measured path is unchanged since PERF021's b53434ab despite 529
# intervening commits: policy_selection, selection_transactions,
# selection_transaction_{http,contract} and the whole parity tree all diff
# empty. The four files that did change are provider-response conversion
# (PR #602), which an open-loop sweep never exercises at provider_calls=0,
# plus an additive opt-in RAYLINE_ROUTER_DECISION_ONLY data-plane guard in
# app.py that leaves /v1/route alone. Arm 0 therefore remains a reproduction
# of PERF021's knee rather than merely a contemporaneous control.
PATHFINDER_AUTHORIZATION_COMMIT = "fb78b2fbbd579d10cd14a78ce71af7c0e9216306"

FLASHINFER_APP_NAME = "rayline-arc-session-encoder-flashinfer-perf031"
FLASHINFER_BUILD_ID = f"{IDENTITY.engine_build_id}+gdn-flashinfer-eager"

# PERF021's knee, which arm 0 must reproduce. The realized arrival rate is
# derived from the frozen schedule span, so these are properties of the packet
# rather than of the run, and they are the same for both arms.
CONTROL_FIRST_OVERLOADED_CELL = "r030"
CONTROL_NOT_OVERLOADED_RATE_RPS = 0.1862
CONTROL_OVERLOADED_RATE_RPS = 0.3724


PERF031A = OpenLoopRunContract(
    run_id=PERF031A_RUN_ID,
    packet_manifest_sha256=PERF021.packet_manifest_sha256,
    corpus_sha256=PERF021.corpus_sha256,
    topology_sha256=PERF021.topology_sha256,
    cells=PERF021.cells,
    compose_project_prefix="rayline-saturation-ladder-perf031a",
    temporary_prefix="rayline-perf031a-",
    budget=BudgetContract(
        run_id=PERF031A_RUN_ID,
        previous_conservative_usd=151.749704666383,
        authorized_cumulative_usd=174.31282402,
        packet_ceiling_usd=7.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
    ),
    # Deliberately the frozen defaults: default app, bare engine build id,
    # torch_reference backend. This is what makes it PERF021's control.
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

PERF031B = OpenLoopRunContract(
    run_id=PERF031B_RUN_ID,
    packet_manifest_sha256=PERF021.packet_manifest_sha256,
    corpus_sha256=PERF021.corpus_sha256,
    topology_sha256=PERF021.topology_sha256,
    cells=PERF021.cells,
    compose_project_prefix="rayline-saturation-ladder-perf031b",
    temporary_prefix="rayline-perf031b-",
    budget=BudgetContract(
        run_id=PERF031B_RUN_ID,
        # Arm 1 charges arm 0's complete envelope first, whatever arm 0
        # actually consumed. The arms are sequential runs, not parallel ones.
        previous_conservative_usd=158.684125466383,
        authorized_cumulative_usd=174.31282402,
        packet_ceiling_usd=7.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
    ),
    encoder_app_name=FLASHINFER_APP_NAME,
    encoder_build_id=FLASHINFER_BUILD_ID,
    encoder_gdn_prefill_backend="flashinfer",
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

SATURATION_LADDER_ARMS = (PERF031A, PERF031B)

# Binding an arm is a separate, human-gated step. Preparation never opens
# launch authority: only a reviewed authorization checkpoint may set this, and
# it opens exactly one run id at a time.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = PERF031B


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline saturation ladder arm is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
