#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Frozen Rayline concurrency sweeps and their one-shot launch authority."""

from __future__ import annotations

from dataclasses import dataclass

from rayline_three_arm_budget import BudgetContract
from rayline_three_arm_contract import IDENTITY

PERF017_RUN_ID = "rayline-concurrency-sweep-perf017-20260802"
PERF018_RUN_ID = "rayline-concurrency-sweep-perf018-20260802"
PERF019_RUN_ID = "rayline-concurrency-sweep-perf019-20260802"
MEASURED_CASES = 32
WARMUP_CASES = 4
MEASURED_EPISODES = 8
WARMUP_EPISODES = 1
SWEEP_ARMS = ("rayline_remote", "rayline_arc")


@dataclass(frozen=True)
class SweepCell:
    concurrency: int
    profile: str
    workload_sha256: str
    identity_sha256: str


@dataclass(frozen=True)
class ConcurrencyRunContract:
    run_id: str
    packet_manifest_sha256: str
    corpus_sha256: str
    topology_sha256: str
    cells: tuple[SweepCell, ...]
    compose_project_prefix: str
    temporary_prefix: str
    budget: BudgetContract
    # The engine build the run's encoder actually serves. The default is
    # PERF017/PERF018/PERF019's exact frozen identity, so those closed runs
    # keep their recorded build id; only a run that deliberately deploys an
    # experiment-profile engine carries a different one. The manifest reports
    # this rather than the frozen literal, so a profiled run cannot write a
    # receipt claiming the default engine.
    encoder_build_id: str = IDENTITY.engine_build_id


SWEEP_CELLS = (
    SweepCell(
        concurrency=1,
        profile="sweep-32-c1",
        workload_sha256=(
            "a350a92ee0f38c3feb72407e9590da29b9ef70da2ca466d3959358c0999f8230"
        ),
        identity_sha256=(
            "440f84ff6259e0af7d89f37cee9818f12dac46071a4686a29338d08d77903993"
        ),
    ),
    SweepCell(
        concurrency=4,
        profile="sweep-32-c4",
        workload_sha256=(
            "2a5cc697004b95c9384489663b7d6d67e69e78c63b2b606c22e55ed58d02e5fb"
        ),
        identity_sha256=(
            "ce9c63795682251107920fcbc4a11de730a14670d0cad4db06acda8180399133"
        ),
    ),
    SweepCell(
        concurrency=8,
        profile="sweep-32-c8",
        workload_sha256=(
            "a7cc6948e731fb8277bbc0d9b79a4b21539515402c3d2d1b146885056c31ebca"
        ),
        identity_sha256=(
            "c16c511864c6aded3275ba21fa275bbad602079e80db010301f290383034d334"
        ),
    ),
)


def _contract(
    run_id: str,
    previous_conservative_usd: float,
    authorized_cumulative_usd: float,
) -> ConcurrencyRunContract:
    return ConcurrencyRunContract(
        run_id=run_id,
        packet_manifest_sha256=(
            "37ea4baa2e935b2851fa6f7a167a9c0fa9f4243891edd6574c0cef9a87919b8c"
        ),
        corpus_sha256=(
            "72bbb22c6a8673d78cb4eadbce46ffd88f882f91f1880b4163e117f4679b1105"
        ),
        topology_sha256=(
            "ad0970c68d2e6b035c187d193f3da8ca49f48a68267bd323e0d66c9d44bcfddd"
        ),
        cells=SWEEP_CELLS,
        compose_project_prefix=f"rayline-concurrency-{run_id.split('-')[-2]}",
        temporary_prefix=f"rayline-{run_id.split('-')[-2]}-",
        budget=BudgetContract(
            run_id=run_id,
            previous_conservative_usd=previous_conservative_usd,
            authorized_cumulative_usd=authorized_cumulative_usd,
            packet_ceiling_usd=6.0,
            required_reserve_usd=3.0,
            maximum_paid_wall_seconds=30 * 60,
        ),
    )


PERF017 = _contract(PERF017_RUN_ID, 55.60064962, 64.31282402)
PERF018 = _contract(PERF018_RUN_ID, 55.89092770, 64.31282402)
PERF019 = _contract(PERF019_RUN_ID, 56.47282774, 84.31282402)

# PERF017 is closed after its first health request timed out before measurement.
# Conservatively accounting 216 resource seconds makes PERF018's cumulative
# full-envelope ceiling USD 61.21269250 and preserves USD 3.10013152.
PERF017_STARTUP_FAILURE_UPPER_USD = 0.29027808
PERF018_STATE_FAILURE_UPPER_USD = 0.58190004
PERF019_REQUIRED_CUMULATIVE_AUTHORITY_USD = 64.79459254
PERF019_ADDITIONAL_AUTHORITY_GRANTED_USD = 20.0
PERF019_RESERVE_AFTER_FULL_ENVELOPE_USD = 22.51823148
# PERF019 completed its only authorized execution with all six receipts and
# cleanup gates passing. Every sweep ID is now closed; another measurement
# requires a new preregistered namespace and explicit launch authority.
LAUNCHABLE_CONTRACT: ConcurrencyRunContract | None = None


def resolve_launch_contract(run_id: str) -> ConcurrencyRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline concurrency sweep is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
