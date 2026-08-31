#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen PERF025/PERF026 retained-session affinity-loss contracts."""

from __future__ import annotations

from rayline_scaleout_contract import ENCODER_APP_NAMES, PERF024, ScaleoutRunContract
from rayline_three_arm_budget import BudgetContract

PERF025_RUN_ID = "rayline-affinity-failover-perf025-20260803"
PERF026_RUN_ID = "rayline-affinity-failover-perf026-20260803"
PATHFINDER_AUTHORIZATION_COMMIT = "c7aaca5bdfcee0c398569b1019e5fd8985461b84"
FAILOVER_ARMS = ("arc_dual_sticky", "arc_dual_forced_failover")
TURNS_PER_EPISODE = 4
FAILOVER_AFTER_POOLING = TURNS_PER_EPISODE // 2
MAXIMUM_PAID_WALL_SECONDS = 20 * 60
MAXIMUM_ORPHAN_REQUEST_SECONDS = 21 * 60
MAXIMUM_SCALEDOWN_SECONDS = 5 * 60
MAXIMUM_RESOURCE_SECONDS = (
    MAXIMUM_PAID_WALL_SECONDS
    + MAXIMUM_ORPHAN_REQUEST_SECONDS
    + MAXIMUM_SCALEDOWN_SECONDS
)

PERF025 = ScaleoutRunContract(
    run_id=PERF025_RUN_ID,
    packet_manifest_sha256=PERF024.packet_manifest_sha256,
    corpus_sha256=PERF024.corpus_sha256,
    topology_sha256=PERF024.topology_sha256,
    cells=PERF024.cells[:1],
    compose_project_prefix="rayline-failover-perf025",
    temporary_prefix="rayline-perf025-",
    budget=BudgetContract(
        run_id=PERF025_RUN_ID,
        previous_conservative_usd=68.343219631823516,
        authorized_cumulative_usd=134.31282402,
        packet_ceiling_usd=7.5,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_replicas=len(ENCODER_APP_NAMES),
    ),
)

PERF026 = ScaleoutRunContract(
    run_id=PERF026_RUN_ID,
    packet_manifest_sha256=PERF024.packet_manifest_sha256,
    corpus_sha256=PERF024.corpus_sha256,
    topology_sha256=PERF024.topology_sha256,
    cells=PERF024.cells[:1],
    compose_project_prefix="rayline-failover-perf026",
    temporary_prefix="rayline-perf026-",
    budget=BudgetContract(
        run_id=PERF026_RUN_ID,
        previous_conservative_usd=70.1005119398672,
        authorized_cumulative_usd=134.31282402,
        packet_ceiling_usd=7.5,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_replicas=len(ENCODER_APP_NAMES),
    ),
)

# PERF025 and PERF026 are closed after their one authorized executions.
LAUNCHABLE_CONTRACT: ScaleoutRunContract | None = None


def resolve_launch_contract(run_id: str) -> ScaleoutRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline failover experiment is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
