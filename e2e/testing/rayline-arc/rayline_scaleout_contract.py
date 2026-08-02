#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen PERF022 single-versus-dual retained-session scale-out contract."""

from __future__ import annotations

from dataclasses import dataclass

from rayline_open_loop_contract import PERF021, OpenLoopCell
from rayline_three_arm_budget import BudgetContract

PERF022_RUN_ID = "rayline-affinity-scaleout-perf022-20260802"
PATHFINDER_AUTHORIZATION_COMMIT = "24b4a3d6a548e5b96589432a5f5d32f572575165"
SCALEOUT_ARMS = ("arc_single", "arc_dual_affinity")
ENCODER_APP_NAMES = (
    "rayline-arc-session-encoder-a",
    "rayline-arc-session-encoder-b",
)
MAXIMUM_PAID_WALL_SECONDS = 40 * 60
MAXIMUM_ORPHAN_REQUEST_SECONDS = 41 * 60
MAXIMUM_SCALEDOWN_SECONDS = 5 * 60
MAXIMUM_RESOURCE_SECONDS = (
    MAXIMUM_PAID_WALL_SECONDS
    + MAXIMUM_ORPHAN_REQUEST_SECONDS
    + MAXIMUM_SCALEDOWN_SECONDS
)


@dataclass(frozen=True)
class ScaleoutRunContract:
    run_id: str
    packet_manifest_sha256: str
    corpus_sha256: str
    topology_sha256: str
    cells: tuple[OpenLoopCell, ...]
    compose_project_prefix: str
    temporary_prefix: str
    budget: BudgetContract


PERF022 = ScaleoutRunContract(
    run_id=PERF022_RUN_ID,
    packet_manifest_sha256=PERF021.packet_manifest_sha256,
    corpus_sha256=PERF021.corpus_sha256,
    topology_sha256=PERF021.topology_sha256,
    cells=PERF021.cells[1:],
    compose_project_prefix="rayline-affinity-perf022",
    temporary_prefix="rayline-perf022-",
    budget=BudgetContract(
        run_id=PERF022_RUN_ID,
        previous_conservative_usd=61.80928732218463,
        authorized_cumulative_usd=134.31282402,
        packet_ceiling_usd=14.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_replicas=2,
    ),
)

# PERF022 is the only launchable contract after its exact Pathfinder
# preregistration, attestation, and authorization checkpoints became visible.
LAUNCHABLE_CONTRACT: ScaleoutRunContract | None = PERF022


def resolve_launch_contract(run_id: str) -> ScaleoutRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline scale-out experiment is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
