#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the aggregate-only staged control/replica-stop ARC packet."""

from __future__ import annotations

import bisect
import concurrent.futures
import hashlib
import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from rayline_open_loop_packet import MAX_EPISODE_LANES
from rayline_open_loop_probe import (
    INPUT_SCHEMA,
    _execute_open_loop,
    _latency,
    _validate_results,
    _warm_selector,
    poisson_schedule,
)
from rayline_parity_http_probe import Case, JSONClient, ProbeError

RECEIPT_SCHEMA = "rayline.vllm.replica-stop-stage.v1"
TURNS_PER_EPISODE = 4
PRELOAD_TURNS_PER_EPISODE = 2
EXPECTED_WARMUP_TURNS = 4
EXPECTED_MEASURED_EPISODES = 8
EXPECTED_PRELOAD_TURNS = 16
EXPECTED_POST_BOUNDARY_TURNS = 16

BoundaryCallback = Callable[[], Mapping[str, Any]]


def _lanes(cases: list[Case]) -> dict[str, list[Case]]:
    lanes: dict[str, list[Case]] = {}
    for case in cases:
        lanes.setdefault(case.episode_id, []).append(case)
    if len(lanes) != EXPECTED_MEASURED_EPISODES or any(
        len(lane) != TURNS_PER_EPISODE for lane in lanes.values()
    ):
        raise ProbeError("replica-stop packet episode shape differs")
    return lanes


def _preload(
    *,
    lanes: Mapping[str, list[Case]],
    select: Callable[[Case], tuple[str, float]],
    worker_map: Mapping[str, str],
    close_connection: Callable[[], None],
) -> list[dict[str, str]]:
    def run_lane(cases: list[Case]) -> list[dict[str, str]]:
        outcomes: list[dict[str, str]] = []
        for case in cases[:PRELOAD_TURNS_PER_EPISODE]:
            try:
                observed, _latency_value = select(case)
                canonical = worker_map.get(observed)
                if canonical is None:
                    raise ProbeError("preload selected an unmapped worker")
                outcomes.append({"case_id": case.case_id, "canonical": canonical})
            finally:
                close_connection()
        return outcomes

    outcomes: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_EPISODE_LANES
    ) as executor:
        futures = [executor.submit(run_lane, lane) for lane in lanes.values()]
        for future in concurrent.futures.as_completed(futures):
            outcomes.extend(future.result())
    if len(outcomes) != EXPECTED_PRELOAD_TURNS:
        raise ProbeError("replica-stop preload count differs")
    return outcomes


def _trace(outcomes: list[Mapping[str, str]]) -> str:
    ordered = sorted(
        ((outcome["case_id"], outcome["canonical"]) for outcome in outcomes),
        key=lambda item: item[0],
    )
    return hashlib.sha256(
        json.dumps(ordered, separators=(",", ":")).encode()
    ).hexdigest()


def _post_results(
    *,
    measured: list[Case],
    schedule: Mapping[str, float],
    outcomes: Mapping[str, Mapping[str, Any]],
    offered_rate_rps: float,
) -> dict[str, Any]:
    ordered = [outcomes[case.case_id] for case in measured]
    successes = [outcome for outcome in ordered if outcome["success"]]
    if not successes:
        raise ProbeError("replica-stop arm completed zero post-boundary decisions")
    completion_offsets = sorted(float(outcome["completion"]) for outcome in ordered)
    arrival_offsets = sorted(schedule.values())
    backlogs = [
        index + 1 - bisect.bisect_right(completion_offsets, arrival)
        for index, arrival in enumerate(arrival_offsets)
    ]
    starts = sorted(float(outcome["start"]) for outcome in ordered)
    duration = max(completion_offsets)
    achieved_start_rate = (
        (len(starts) - 1) / (starts[-1] - starts[0])
        if len(starts) > 1 and starts[-1] > starts[0]
        else offered_rate_rps
    )
    trace = json.dumps(
        [[outcome["case_id"], outcome["canonical"]] for outcome in successes],
        separators=(",", ":"),
    ).encode()
    results = {
        "scheduled": len(measured),
        "completed": len(successes),
        "failed": len(measured) - len(successes),
        "offered_rate_rps": offered_rate_rps,
        "realized_arrival_rate_rps": (
            (len(measured) - 1) / (arrival_offsets[-1] - arrival_offsets[0])
        ),
        "scheduled_span_seconds": arrival_offsets[-1] - arrival_offsets[0],
        "duration_seconds": duration,
        "completion_throughput_rps": len(successes) / duration,
        "achieved_start_rate_rps": achieved_start_rate,
        "service_latency_seconds": _latency(
            [float(outcome["service"]) for outcome in successes]
        ),
        "scheduled_latency_seconds": _latency(
            [
                float(outcome["completion"]) - float(outcome["scheduled"])
                for outcome in successes
            ]
        ),
        "start_lag_seconds": _latency(
            [
                float(outcome["start"]) - float(outcome["scheduled"])
                for outcome in successes
            ]
        ),
        "max_client_backlog": max(backlogs),
        "backlog_at_final_arrival": backlogs[-1],
        "drain_seconds_after_final_arrival": max(
            0.0, completion_offsets[-1] - arrival_offsets[-1]
        ),
        "selected_worker_trace_sha256": hashlib.sha256(trace).hexdigest(),
        "provider_calls": 0,
    }
    return _validate_results(
        results,
        case_count=EXPECTED_POST_BOUNDARY_TURNS,
        schema_version=INPUT_SCHEMA,
    )


def run_replica_stop_probe(
    *,
    client: JSONClient,
    warmup: list[Case],
    measured: list[Case],
    identity: Mapping[str, Any],
    worker_map: Mapping[str, str],
    run_id: str,
    offered_rate_rps: float,
    boundary_callback: BoundaryCallback,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    if len(warmup) != EXPECTED_WARMUP_TURNS:
        raise ProbeError("replica-stop warmup count differs")
    lanes = _lanes(measured)
    select = _warm_selector(
        arm="rayline_arc",
        protocol="openai_gateway",
        client=client,
        run_id=run_id,
        identity=identity,
        warmup=warmup,
        worker_map=worker_map,
    )
    preload = _preload(
        lanes=lanes,
        select=select,
        worker_map=worker_map,
        close_connection=client.close_thread_connection,
    )
    boundary = dict(boundary_callback())
    post = [
        case for lane in lanes.values() for case in lane[PRELOAD_TURNS_PER_EPISODE:]
    ]
    schedule = poisson_schedule(post, offered_rate_rps, int(identity["seed"]))
    post_lanes = {
        episode_id: lane[PRELOAD_TURNS_PER_EPISODE:]
        for episode_id, lane in lanes.items()
    }
    base = clock()
    outcomes = _execute_open_loop(
        lanes=post_lanes,
        schedule=schedule,
        select=select,
        worker_map=worker_map,
        base=base,
        clock=clock,
        sleeper=sleeper,
        close_connection=client.close_thread_connection,
    )
    return {
        "schema_version": RECEIPT_SCHEMA,
        "arm": "rayline_arc",
        "run_id": run_id,
        "identity": dict(identity),
        "stage": {
            "warmup_turns": len(warmup),
            "preload_turns": len(preload),
            "post_boundary_turns": len(post),
            "turns_per_episode": TURNS_PER_EPISODE,
            "preload_turns_per_episode": PRELOAD_TURNS_PER_EPISODE,
            "boundary_excluded_from_latency": True,
        },
        "preload": {
            "scheduled": EXPECTED_PRELOAD_TURNS,
            "completed": len(preload),
            "failed": EXPECTED_PRELOAD_TURNS - len(preload),
            "selected_worker_trace_sha256": _trace(preload),
        },
        "boundary": boundary,
        "results": _post_results(
            measured=post,
            schedule=schedule,
            outcomes=outcomes,
            offered_rate_rps=offered_rate_rps,
        ),
    }
