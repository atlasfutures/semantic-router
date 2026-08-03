#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Closed preregistration contract for the DYN006 dynamic-capacity stop cell."""

from __future__ import annotations

from rayline_scaleout_contract import PERF024, ScaleoutRunContract
from rayline_three_arm_budget import BudgetContract

DYN006_RUN_ID = "rayline-dynamic-capacity-stop-dyn006-20260803"
PATHFINDER_AUTHORIZATION_COMMIT = "PENDING"
DYNAMIC_STOP_ARMS = (
    "arc_dynamic_three_control",
    "arc_dynamic_drain_stop",
)
ENCODER_APP_NAMES = (
    "rayline-arc-session-encoder-a",
    "rayline-arc-session-encoder-b",
    "rayline-arc-session-encoder-c",
)
ENCODER_REPLICA_IDS = ("encoder-a", "encoder-b", "encoder-c")
INITIAL_MEMBERSHIP_REVISION = 1
REGISTERED_MEMBERSHIP_REVISION = 2
DRAINING_MEMBERSHIP_REVISION = 3
REMOVED_MEMBERSHIP_REVISION = 4
SURVIVOR_COUNT = len(ENCODER_REPLICA_IDS) - 1
MEMBERSHIP_ADOPTION_SECONDS = 2.5
SESSION_NAMESPACE = "dynamic-capacity-61"
WARMUP_EPISODE_ID = "942976817dbfe09ef61ec82daa871195d305691cd422291702f48d67ee1c73d6"
MEASURED_EPISODE_IDS = (
    "33e7e2685941b94d9715b6c7ecd4d4e898ff17147304e375f1d6be8c84131df1",
    "464c256df51bcb4412f3ad6a96d3c9ca858873bb131c2b5673a52cc0a2eff5b0",
    "4cdebfe6cb74bc83fcfe47491bcdf7b2da4185d7528784d43411c476f5c81f3d",
    "5ecc2a0a3f919fe9b1fac0e919b79df788f68c16325f7c86964a68aa4e4c4622",
    "68163ad4855616e493b67fdfc946691a99de23c4086c086efcc39c20f272c6cd",
    "6ee2332ba55484ad45667745f4ac87fb2f7e223a5a10664d58bda82643806657",
    "80ec2d2dd517812e0e11bb0cefe2eed50d1d94ecf3395b029f96f4aa536ed4bc",
    "ac01fc2ef6e4fc2d20b73c7903bfdbc50f05609031e9449080469a5b20b17501",
)
UNAVAILABLE_REPLICA = 0
UNAVAILABLE_APP_NAME = ENCODER_APP_NAMES[UNAVAILABLE_REPLICA]
UNAVAILABLE_REPLICA_ID = ENCODER_REPLICA_IDS[UNAVAILABLE_REPLICA]
EXPECTED_PRE_BOUNDARY_OWNERS = (2, 3, 3)
EXPECTED_POST_STOP_OWNERS = (0, 4, 4)
EXPECTED_AFFECTED_SESSIONS = EXPECTED_PRE_BOUNDARY_OWNERS[UNAVAILABLE_REPLICA]
MAXIMUM_PAID_WALL_SECONDS = 20 * 60
MAXIMUM_ORPHAN_REQUEST_SECONDS = 21 * 60
MAXIMUM_SCALEDOWN_SECONDS = 5 * 60
MAXIMUM_RESOURCE_SECONDS = (
    MAXIMUM_PAID_WALL_SECONDS
    + MAXIMUM_ORPHAN_REQUEST_SECONDS
    + MAXIMUM_SCALEDOWN_SECONDS
)

DYN006 = ScaleoutRunContract(
    run_id=DYN006_RUN_ID,
    packet_manifest_sha256=PERF024.packet_manifest_sha256,
    corpus_sha256=PERF024.corpus_sha256,
    topology_sha256=PERF024.topology_sha256,
    cells=PERF024.cells[:1],
    compose_project_prefix="rayline-dynamic-stop-dyn006",
    temporary_prefix="rayline-dyn006-",
    budget=BudgetContract(
        run_id=DYN006_RUN_ID,
        previous_conservative_usd=73.64050361447986,
        authorized_cumulative_usd=134.31282402,
        packet_ceiling_usd=12.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
        encoder_replicas=len(ENCODER_APP_NAMES),
    ),
)

# Source, registry preregistration, attestation, and authorization must all be
# pushed before this can name DYN006. The held 1,000-case qualification has no
# entrypoint in this launcher.
LAUNCHABLE_CONTRACT: ScaleoutRunContract | None = None


def resolve_launch_contract(run_id: str) -> ScaleoutRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline dynamic-stop experiment is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            "launcher only permits preregistered run id "
            f"{LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
