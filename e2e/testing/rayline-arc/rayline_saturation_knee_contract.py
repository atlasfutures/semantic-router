#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepared PERF032 single-arm FlashInfer saturation-knee packet.

PERF031B put FlashInfer on the frozen PERF021 ladder and it never saturated:
`first_overloaded_cell` was `null` on both sub-arms, completion tracked the
offered rate at every rung, backlog at final arrival stayed 1/2/3 against the
control's 5/8/8, and drain after final arrival was 1.8s/1.4s/2.5s against the
control's 42.2s/60.0s/65.2s. The only admissible claim from that packet is that
the FlashInfer knee lies above `0.5586011607274736` realized decisions per
second on one H100, and that its location is unknown. PERF032 exists to locate
it.

Two design consequences follow.

- **Single arm.** There is no `torch_reference` arm. The control already
  saturates at `r030` and PERF031A reproduced that exactly, so running it above
  `r045` would measure nothing that is not already recorded. The comparison
  PERF032 supports is against PERF031's closed numbers, not against a
  contemporaneous arm.
- **A new packet, not an extended one.** PERF031's ladder cannot be widened
  after the fact without destroying the control that justified it. PERF032 gets
  its own packet, its own run id and its own namespaces.

Nothing about the workload changes. The packet is regenerated from the same
PERF017 source packet that produced PERF020, so the corpus, the topology, the
seed and the source identity are byte-identical to every open-loop run this
repo has recorded; only the rung set differs.
"""

from __future__ import annotations

from rayline_open_loop_contract import (
    MAXIMUM_ORPHAN_REQUEST_SECONDS,
    MAXIMUM_PAID_WALL_SECONDS,
    MAXIMUM_SCALEDOWN_SECONDS,
    PERF020,
    OpenLoopCell,
    OpenLoopRunContract,
)
from rayline_three_arm_budget import BudgetContract
from rayline_three_arm_contract import IDENTITY

PERF032_RUN_ID = "rayline-saturation-knee-perf032-20260810"

# ---------------------------------------------------------------------------
# HUMAN GATES. Both are fail-closed placeholders. Preparation may not move
# either one; only a reviewed authorization checkpoint may.
# ---------------------------------------------------------------------------

# The Pathfinder head this run's authority is pinned to. `_assert_pushed`
# forces the Pathfinder HEAD to equal this, and no commit can equal `PENDING`,
# so the packet cannot launch while it reads this.
PATHFINDER_AUTHORIZATION_COMMIT = "fb78b2fbbd579d10cd14a78ce71af7c0e9216306"

# NOT YET GRANTED. This is deliberately still PERF031's ceiling. With
# PERF031's closing position of `$165.618546266383` and this packet's
# `$6.9344208` envelope, it leaves a `$1.759856953617` reserve against the
# `$3.00` floor, so `budget_receipt` raises `BudgetError` and PERF032 cannot
# run. A human must raise this. The minimum viable grant is `$1.2401`, which
# takes the ceiling to `$175.552967066383` and the reserve to exactly `$3.00`.
# Do not invent a granted figure here.
AUTHORIZED_CUMULATIVE_USD = 184.31282402
MINIMUM_VIABLE_GRANT_USD = 1.240143046383

# The FlashInfer app PERF031B already deployed and measured. It is registered in
# `EXPERIMENT_APP_PROFILES`, so PERF032 needs no new profile and no allowlist
# change; reusing it also keeps the engine identity byte-identical to the run
# whose unsaturated result PERF032 extends.
FLASHINFER_APP_NAME = "rayline-arc-session-encoder-flashinfer-perf031"
FLASHINFER_BUILD_ID = f"{IDENTITY.engine_build_id}+gdn-flashinfer-eager"

# The unsaturated ceiling PERF031B established. `r045` re-offers exactly the
# rate that produced it, and is this packet's negative control: if PERF032 does
# not reproduce an unsaturated `r045`, the packet is measuring a different
# system and every higher rung is uninterpretable.
ANCHOR_CELL = "r045"
ANCHOR_OFFERED_RATE_RPS = 0.45
ANCHOR_REALIZED_ARRIVAL_RATE_RPS = 0.5586011607274736
ANCHOR_COMPLETION_THROUGHPUT_DPS = 0.5518306368768308

# The handoff capacity model's transport-bound FlashInfer branch,
# `1.055 / (0.637 + 0.286)`, which PERF031B topped out at less than half of and
# therefore never tested. `r090` and `r120` bracket it: if the model holds,
# `r090` completes unsaturated and `r120` overloads. Either rung falsifies it.
PREDICTED_KNEE_DECISIONS_PER_SECOND = 1.143

PERF032 = OpenLoopRunContract(
    run_id=PERF032_RUN_ID,
    packet_manifest_sha256=(
        "eeb1c69f57ae964b238c7763ff87abf2dc727ba94b757c45e24aa2e013b08fed"
    ),
    # Unchanged from PERF020/PERF021/PERF031: the same 32 measured cases over
    # the same eight-lane single-encoder topology. Only the rungs are new.
    corpus_sha256=PERF020.corpus_sha256,
    topology_sha256=PERF020.topology_sha256,
    cells=(
        # Byte-identical to PERF020's `r045` cell, because the workload and
        # identity documents a rung produces depend only on the rate and the
        # shared source packet. The anchor is therefore literally the same cell
        # PERF031B ran, not merely one offering the same rate.
        OpenLoopCell(
            label="r045",
            offered_rate_rps=0.45,
            workload_sha256=(
                "4f396a19f2f35dd00379a262b0cad5e3871c14210fa80c30f3e3b01cb2cafc2e"
            ),
            identity_sha256=(
                "131d1d70a05463871ab1f40572f0f53e26cdb0c9ce6d44407570729bb48d4073"
            ),
        ),
        OpenLoopCell(
            label="r060",
            offered_rate_rps=0.60,
            workload_sha256=(
                "d701ad4add973abf69b8a930c52984c050019edbc75ee96deff871c2316d6d94"
            ),
            identity_sha256=(
                "28fc634bb35affa7bd47e7da828ecf383729be036cefd5271d10c07fe3b8e1ec"
            ),
        ),
        OpenLoopCell(
            label="r090",
            offered_rate_rps=0.90,
            workload_sha256=(
                "f37e0a1d09b1be7dfb5d2e1e40164a12c636d1ba40c2f3e3439758d030e631af"
            ),
            identity_sha256=(
                "f54bc060795a0f800ed6362eacf99bfd7814bf97dd699dcdcb90b7809b80180b"
            ),
        ),
        OpenLoopCell(
            label="r120",
            offered_rate_rps=1.20,
            workload_sha256=(
                "8990bbe3b27b5d848b22d116d26c2f1d50d325ac9e5e9dbf2280b9d5590dcb92"
            ),
            identity_sha256=(
                "29a2cbc24842397fb47a14a66ca25d4c031453d8daa5d1982559e235ca1c3bd0"
            ),
        ),
    ),
    compose_project_prefix="rayline-saturation-knee-perf032",
    temporary_prefix="rayline-perf032-",
    budget=BudgetContract(
        run_id=PERF032_RUN_ID,
        # Where PERF031 left the ledger: both arms charged their complete
        # envelopes, whatever they actually consumed.
        previous_conservative_usd=165.618546266383,
        authorized_cumulative_usd=AUTHORIZED_CUMULATIVE_USD,
        packet_ceiling_usd=7.0,
        required_reserve_usd=3.0,
        # Four rungs on the same 40-minute paid-wall bound as the three-rung
        # ladder. The envelope does not grow because the arrival schedules
        # shrink: PERF032's slowest rung is PERF031's fastest.
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
    ),
    encoder_app_name=FLASHINFER_APP_NAME,
    encoder_build_id=FLASHINFER_BUILD_ID,
    encoder_gdn_prefill_backend="flashinfer",
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

SATURATION_KNEE_ARMS = (PERF032,)

# Binding is a separate, human-gated step. Preparation never opens launch
# authority.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = PERF032


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline saturation knee arm is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
