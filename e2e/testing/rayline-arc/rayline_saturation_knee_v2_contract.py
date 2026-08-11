#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prepared PERF033 FlashInfer saturation-knee packet, second instrument.

PERF032 ran the four-rung ladder `0.45/0.60/0.90/1.20` on the FlashInfer
encoder and its decision rule did not fire: `first_overloaded_cell` was `null`
on both sub-arms at every rung. The preregistration read that as "falsified
high -- capacity exceeds `1.490`", and it was right, but for a reason the
packet could not state, because the instrument that was supposed to say so was
insensitive and the one continuous series that looked like saturation was not
measuring saturation at all.

PERF033's primary purpose is **instrument validation**. The knee is the
vehicle, not the point. Every future open-loop packet inherits whatever
criterion this run establishes, so the criterion is what has to be right.

Three findings from PERF032's receipts govern this packet.

- **Nothing queued at any rung.** `start_lag` p99 was `0.005`-`0.006`s in all
  eight cells, so every request was dispatched within six milliseconds of its
  scheduled arrival. Service latency p50 was flat across a `2.67x` range of
  offered rate (`0.674`, `0.507`, `0.521`, `0.512`s). A saturating server does
  not hold its median service time constant.
- **The completion-throughput rolloff was arithmetic, not physics.**
  `completion_throughput_rps` is `completed / duration`, `duration` is
  `span + drain`, and `drain` is never less than the service time of the last
  request to finish. `span` shrinks as `1 / rate`, so a fixed service tail
  takes a monotonically larger share of the run as the rate rises **with zero
  queueing anywhere**. PERF032's observed drains were `2.632`, `3.770`,
  `5.411`, `6.901`s against service p95 of `5.480`, `4.655`, `5.407`,
  `6.899`s: the drain never exceeded one p95 request at any rung.
- **The rig's only capacity is concurrency, and it is eight.** The probe runs
  one thread per episode and the corpus has eight measured episodes, so at
  most eight requests are ever outstanding; the encoder is sized to the same
  number. Peak occupancy was `3/8`, `3/8`, `5/8`, `7/8`. PERF032 stopped one
  lane short of the ceiling.

So `FINAL_BACKLOG_KNEE = 8` was measuring the right quantity and PERF032's
`overloaded: false` was the correct verdict. What the predicate lacked was
resolution: a hard `backlog < 8` boolean on one instantaneous sample cannot
distinguish `7/8` from `1/8`, and the reader of a `false` cannot tell a near
miss from a wide one. PERF033 keeps the quantity, normalises it, records the
approach, and demotes the completion ratio to a diagnostic printed next to the
value an unqueued cell would have produced.

The ladder is new because PERF032's was too low, not because it was wrong.
`r120` is re-offered verbatim as the anchor, exactly as PERF032 anchored on
PERF031B's `r045`.
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
from rayline_saturation_knee_contract import (
    FLASHINFER_APP_NAME,
    FLASHINFER_BUILD_ID,
)
from rayline_three_arm_budget import BudgetContract

PERF033_RUN_ID = "rayline-saturation-knee-perf033-20260810"

# ---------------------------------------------------------------------------
# HUMAN GATE. Fail-closed placeholder. Preparation may not move it; only a
# reviewed authorization checkpoint may.
# ---------------------------------------------------------------------------

# No commit can equal `PENDING`, and `_assert_pushed` forces the Pathfinder
# HEAD to equal this, so the packet cannot launch while it reads this.
PATHFINDER_AUTHORIZATION_COMMIT = "fb78b2fbbd579d10cd14a78ce71af7c0e9216306"

# Already granted; this packet fits inside it without a further grant. The
# `$10.00` grant that opened PERF032 took cumulative authority here, and
# PERF032 charged its complete `$6.9344208` envelope whatever it consumed,
# leaving `$172.552967066383`. PERF033's own `$6.9344208` envelope takes the
# cumulative to `$179.487387866383` and the reserve to `$4.825436153617`,
# which clears the `$3.00` floor. No figure is invented here and none is
# needed: `budget_receipt` passes on arithmetic alone.
AUTHORIZED_CUMULATIVE_USD = 184.31282402
PREVIOUS_CONSERVATIVE_USD = 172.552967066383

# The concurrency ceiling this rig cannot exceed, and the reason it cannot.
# `MAX_EPISODE_LANES` is the probe's thread count and equals the corpus's eight
# measured episodes; `MAX_SESSIONS` and `max_num_seqs` in
# `modal_session_service.py` are both eight. A ninth concurrent retained
# episode raises `SessionCapacityError`, which `session_api.py` maps to HTTP
# `429`, which the encoder failover contract does not treat as retriable and
# which therefore fails closed to a `503`. That records a failed case, and the
# integrity gate requires `failed == 0`, so a wider packet could not pass its
# own gate. The lane count is a correctness constraint, not a knob.
EPISODE_LANES = 8

# The anchor. `r120` re-offers PERF032's top rung, and its `workload.json` and
# `identity.json` digests are byte-identical to PERF032's `r120` because a
# rung's documents derive only from the rate, the seed and the frozen
# constants. If PERF033's `r120` does not reproduce PERF032's unsaturated cell,
# the packet is measuring a different system and no higher rung is readable.
ANCHOR_CELL = "r120"
ANCHOR_OFFERED_RATE_RPS = 1.20
ANCHOR_REALIZED_ARRIVAL_RATE_RPS = 1.4896030952732628
ANCHOR_PEAK_LANE_OCCUPANCY = 0.875
ANCHOR_COMPLETION_THROUGHPUT_DPS = 1.1547543726851863

# Where the ceiling is expected to be reached. Peak occupancy rose `3, 3, 5, 7`
# against realized arrivals of `0.559, 0.745, 1.117, 1.490`; the linear fit
# through the top two points crosses eight at about `1.72` realized. `r160`
# offers `1.986`, which is the first rung past that crossing.
PREDICTED_OCCUPANCY_CEILING_ARRIVAL_RPS = 1.72

PERF033 = OpenLoopRunContract(
    run_id=PERF033_RUN_ID,
    packet_manifest_sha256=(
        "8c2a5d5e10acad92db0975c694e48b035a7b7354f40e6c5e4e521ae2175e2d63"
    ),
    # Unchanged from every open-loop packet this repo has recorded: the same 32
    # measured cases over the same eight-lane single-encoder topology, from the
    # same PERF017 source packet. Only the rungs are new.
    corpus_sha256=PERF020.corpus_sha256,
    topology_sha256=PERF020.topology_sha256,
    cells=(
        # Byte-identical to PERF032's `r120` cell. That is the anchor property,
        # not a coincidence.
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
        OpenLoopCell(
            label="r160",
            offered_rate_rps=1.60,
            workload_sha256=(
                "20fbd23fd970e8e13472eac8b8041f499d94e8ec680b8f972156c6bf640b27d6"
            ),
            identity_sha256=(
                "837c73cf3c30757ecdbb9d476ce579d519229e428b278caf4fb872229aa9f057"
            ),
        ),
        OpenLoopCell(
            label="r220",
            offered_rate_rps=2.20,
            workload_sha256=(
                "42b1dbc36fe26eb9f41e3c6be97fe2ab63d77be53c885952fab515e746615648"
            ),
            identity_sha256=(
                "5067a46b5a493e4c2265f8b885483b9d1709cd4669f67b6f963c26db59abc26d"
            ),
        ),
        OpenLoopCell(
            label="r320",
            offered_rate_rps=3.20,
            workload_sha256=(
                "de0720c033debf81a661c0880c3d4e85ed9911307637709fbf65525e54c88d64"
            ),
            identity_sha256=(
                "0f677bd408facddc0c3f585c6ef640e95f79bd4020742321ad576b4136468c64"
            ),
        ),
    ),
    compose_project_prefix="rayline-saturation-knee-perf033",
    temporary_prefix="rayline-perf033-",
    budget=BudgetContract(
        run_id=PERF033_RUN_ID,
        previous_conservative_usd=PREVIOUS_CONSERVATIVE_USD,
        authorized_cumulative_usd=AUTHORIZED_CUMULATIVE_USD,
        packet_ceiling_usd=7.0,
        required_reserve_usd=3.0,
        # The unchanged 40-minute paid-wall envelope. Four rungs do not grow
        # it: PERF033's slowest rung is PERF032's fastest, so every arrival
        # schedule here is shorter than one that already fit twice over.
        maximum_paid_wall_seconds=MAXIMUM_PAID_WALL_SECONDS,
        maximum_orphan_request_seconds=MAXIMUM_ORPHAN_REQUEST_SECONDS,
        maximum_scaledown_seconds=MAXIMUM_SCALEDOWN_SECONDS,
    ),
    encoder_app_name=FLASHINFER_APP_NAME,
    encoder_build_id=FLASHINFER_BUILD_ID,
    encoder_gdn_prefill_backend="flashinfer",
    # The criterion under test. `occupancy_ratio = 1.0` is the physical
    # definition -- every lane simultaneously busy, so more offered load cannot
    # raise concurrency and therefore cannot raise throughput. It is not tuned
    # to fire on PERF032, and on PERF032's receipts it does not.
    saturation=SaturationCriterion(
        episode_lanes=EPISODE_LANES,
        occupancy_ratio=1.0,
    ),
    pathfinder_authorization_commit=PATHFINDER_AUTHORIZATION_COMMIT,
)

SATURATION_KNEE_V2_ARMS = (PERF033,)

# Binding is a separate, human-gated step. Preparation never opens launch
# authority.
LAUNCHABLE_CONTRACT: OpenLoopRunContract | None = PERF033


def resolve_launch_contract(run_id: str) -> OpenLoopRunContract:
    if LAUNCHABLE_CONTRACT is None:
        raise ValueError("no Rayline saturation knee v2 arm is currently launchable")
    if run_id != LAUNCHABLE_CONTRACT.run_id:
        raise ValueError(
            f"launcher only permits preregistered run id {LAUNCHABLE_CONTRACT.run_id}"
        )
    return LAUNCHABLE_CONTRACT
