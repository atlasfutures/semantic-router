#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen closed PERF027 real-replica-stop contract."""

from __future__ import annotations

from rayline_scaleout_contract import ENCODER_APP_NAMES, PERF024, ScaleoutRunContract
from rayline_three_arm_budget import BudgetContract

PERF027_RUN_ID = "rayline-replica-stop-perf027-20260803"
PATHFINDER_AUTHORIZATION_COMMIT = "afb5aa1be2fb9416422ac3adeb5bccefa360e401"
STOP_ARMS = ("arc_dual_staged_control", "arc_dual_replica_stop")
SESSION_NAMESPACE = "shared-replica-stop"
UNAVAILABLE_REPLICA = 0
UNAVAILABLE_APP_NAME = ENCODER_APP_NAMES[UNAVAILABLE_REPLICA]
EXPECTED_MEASURED_PRIMARY_SESSIONS = (4, 4)
EXPECTED_ALL_PRIMARY_SESSIONS = (5, 4)
EXPECTED_AFFECTED_SESSIONS = EXPECTED_MEASURED_PRIMARY_SESSIONS[UNAVAILABLE_REPLICA]
MAXIMUM_PAID_WALL_SECONDS = 20 * 60
MAXIMUM_ORPHAN_REQUEST_SECONDS = 21 * 60
MAXIMUM_SCALEDOWN_SECONDS = 5 * 60
MAXIMUM_RESOURCE_SECONDS = (
    MAXIMUM_PAID_WALL_SECONDS
    + MAXIMUM_ORPHAN_REQUEST_SECONDS
    + MAXIMUM_SCALEDOWN_SECONDS
)

PERF027 = ScaleoutRunContract(
    run_id=PERF027_RUN_ID,
    packet_manifest_sha256=PERF024.packet_manifest_sha256,
    corpus_sha256=PERF024.corpus_sha256,
    topology_sha256=PERF024.topology_sha256,
    cells=PERF024.cells[:1],
    compose_project_prefix="rayline-replica-stop-perf027",
    temporary_prefix="rayline-perf027-",
    budget=BudgetContract(
        run_id=PERF027_RUN_ID,
        previous_conservative_usd=71.9354755968929,
        authorized_cumulative_usd=134.31282402,
        packet_ceiling_usd=7.5,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_replicas=len(ENCODER_APP_NAMES),
    ),
)

LAUNCHABLE_CONTRACT: ScaleoutRunContract | None = None


def resolve_launch_contract(run_id: str) -> ScaleoutRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline replica-stop experiment is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
