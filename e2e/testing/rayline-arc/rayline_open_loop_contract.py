#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen PERF020 open-loop sweep and one-shot launch authority."""

from __future__ import annotations

from dataclasses import dataclass

from rayline_three_arm_budget import BudgetContract

PERF020_RUN_ID = "rayline-open-loop-sweep-perf020-20260802"
PATHFINDER_AUTHORIZATION_COMMIT = "8785d0ca94b579accf128a06c369f9a06ab229f0"
MEASURED_CASES = 32
WARMUP_CASES = 4
MEASURED_EPISODES = 8
WARMUP_EPISODES = 1
OPEN_LOOP_ARMS = ("rayline_remote", "rayline_arc")
MAXIMUM_PAID_WALL_SECONDS = 40 * 60
MAXIMUM_ORPHAN_REQUEST_SECONDS = 41 * 60
MAXIMUM_SCALEDOWN_SECONDS = 5 * 60
MAXIMUM_RESOURCE_SECONDS = (
    MAXIMUM_PAID_WALL_SECONDS
    + MAXIMUM_ORPHAN_REQUEST_SECONDS
    + MAXIMUM_SCALEDOWN_SECONDS
)


@dataclass(frozen=True)
class OpenLoopCell:
    label: str
    offered_rate_rps: float
    workload_sha256: str
    identity_sha256: str

    @property
    def concurrency(self) -> int:
        """Compatibility with the shared eight-lane local-stack builder."""
        return 8


@dataclass(frozen=True)
class OpenLoopRunContract:
    run_id: str
    packet_manifest_sha256: str
    corpus_sha256: str
    topology_sha256: str
    cells: tuple[OpenLoopCell, ...]
    compose_project_prefix: str
    temporary_prefix: str
    budget: BudgetContract


PERF020 = OpenLoopRunContract(
    run_id=PERF020_RUN_ID,
    packet_manifest_sha256="e1b992793aad733ee63d586974172355c969a7c7f5781f580bb21143e9af23df",
    corpus_sha256="72bbb22c6a8673d78cb4eadbce46ffd88f882f91f1880b4163e117f4679b1105",
    topology_sha256="ad0970c68d2e6b035c187d193f3da8ca49f48a68267bd323e0d66c9d44bcfddd",
    cells=(
        OpenLoopCell(
            label="r015",
            offered_rate_rps=0.15,
            workload_sha256="b2529d40e824b133c1dd79bad3552f56dac9485c8cc412cdcc1a3bcf9a2c08af",
            identity_sha256="be489e730c15a47ef80298cc4341941130ebff909e36cca08611e3be63b1a77c",
        ),
        OpenLoopCell(
            label="r030",
            offered_rate_rps=0.30,
            workload_sha256="fb765a65038a15415e9776b52d0a465f53cd989453f2ddc54fda2583b9dfcb6c",
            identity_sha256="f72d61ab651052b55bf21ce32c74f80cd21b74812de12511f4d174517b29c814",
        ),
        OpenLoopCell(
            label="r045",
            offered_rate_rps=0.45,
            workload_sha256="4f396a19f2f35dd00379a262b0cad5e3871c14210fa80c30f3e3b01cb2cafc2e",
            identity_sha256="131d1d70a05463871ab1f40572f0f53e26cdb0c9ce6d44407570729bb48d4073",
        ),
    ),
    compose_project_prefix="rayline-open-loop-perf020",
    temporary_prefix="rayline-perf020-",
    budget=BudgetContract(
        run_id=PERF020_RUN_ID,
        previous_conservative_usd=58.09966073955587,
        authorized_cumulative_usd=84.31282402,
        packet_ceiling_usd=7.0,
        required_reserve_usd=3.0,
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
    ),
)

# The registry authorization above is signed and pushed. This separate source
# checkpoint opens exactly one run ID; any launched outcome closes it.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = PERF020


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline open-loop sweep is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
