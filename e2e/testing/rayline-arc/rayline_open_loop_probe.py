#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run one aggregate-only Rayline open-loop offered-load cell."""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from rayline_open_loop_packet import (
    ARRIVAL_PROCESS,
    COORDINATED_OMISSION_POLICY,
    MAX_EPISODE_LANES,
    MEASURED_CASES,
    MEASUREMENT_SCOPE,
    WARMUP_CASES,
    WORKLOAD_SCHEMA,
)
from rayline_parity_comparator import ARMS, IDENTITY_FIELDS
from rayline_parity_http_probe import (
    MIN_CANONICAL_WORKERS,
    PROTOCOL_BY_ARM,
    TOPOLOGY_SCHEMA,
    Case,
    JSONClient,
    ProbeError,
    _exact_keys,
    _percentile,
    _read_json,
    _selector,
    _sha256,
    load_corpus,
)

INPUT_SCHEMA = "rayline.vllm.open-loop-input.v1"
RUN_ID_MAX_LENGTH = 128
SHA256_LENGTH = 64
LATENCY_FIELDS = ("p50", "p95", "p99")
RESULT_FIELDS = (
    "scheduled",
    "completed",
    "failed",
    "offered_rate_rps",
    "scheduled_span_seconds",
    "duration_seconds",
    "completion_throughput_rps",
    "achieved_start_rate_rps",
    "service_latency_seconds",
    "scheduled_latency_seconds",
    "start_lag_seconds",
    "max_client_backlog",
    "backlog_at_final_arrival",
    "drain_seconds_after_final_arrival",
    "selected_worker_trace_sha256",
    "provider_calls",
)


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number <= 0):
        raise ProbeError(f"{label} must be finite and non-negative")
    return number


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProbeError(f"{label} must be a non-negative integer")
    return value


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _validate_latency(raw: object, label: str) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise ProbeError(f"{label} must be an object")
    _exact_keys(raw, set(LATENCY_FIELDS), label)
    latency = {
        field: _finite(raw[field], f"{label}.{field}") for field in LATENCY_FIELDS
    }
    if not latency["p50"] <= latency["p95"] <= latency["p99"]:
        raise ProbeError(f"{label} percentiles must be monotonic")
    return latency


def _validate_identity(identity_raw: object) -> dict[str, Any]:
    if not isinstance(identity_raw, Mapping):
        raise ProbeError("identity must be an object")
    _exact_keys(identity_raw, set(IDENTITY_FIELDS), "identity")
    identity = dict(identity_raw)
    if identity["measurement_scope"] != MEASUREMENT_SCOPE:
        raise ProbeError("identity measurement scope differs")
    if identity["case_count"] != MEASURED_CASES:
        raise ProbeError("identity case count differs")
    return identity


def _validate_results(results_raw: object, *, case_count: int) -> dict[str, Any]:
    if not isinstance(results_raw, Mapping):
        raise ProbeError("results must be an object")
    _exact_keys(results_raw, set(RESULT_FIELDS), "results")
    results = dict(results_raw)
    for field in (
        "scheduled",
        "completed",
        "failed",
        "max_client_backlog",
        "backlog_at_final_arrival",
        "provider_calls",
    ):
        results[field] = _count(results[field], f"results.{field}")
    if results["scheduled"] != case_count:
        raise ProbeError("scheduled count differs from identity")
    if results["completed"] + results["failed"] != results["scheduled"]:
        raise ProbeError("open-loop result counts do not reconcile")
    for field in (
        "offered_rate_rps",
        "scheduled_span_seconds",
        "duration_seconds",
        "completion_throughput_rps",
        "achieved_start_rate_rps",
        "drain_seconds_after_final_arrival",
    ):
        results[field] = _finite(
            results[field],
            f"results.{field}",
            positive=field
            in {
                "offered_rate_rps",
                "duration_seconds",
                "completion_throughput_rps",
                "achieved_start_rate_rps",
            },
        )
    for field in (
        "service_latency_seconds",
        "scheduled_latency_seconds",
        "start_lag_seconds",
    ):
        results[field] = _validate_latency(results[field], f"results.{field}")
    trace = str(results["selected_worker_trace_sha256"])
    if len(trace) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in trace
    ):
        raise ProbeError("selected worker trace digest is invalid")
    results["selected_worker_trace_sha256"] = trace
    if results["max_client_backlog"] < results["backlog_at_final_arrival"]:
        raise ProbeError("final backlog exceeds maximum backlog")
    return results


def validate_receipt(raw: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        raw, {"schema_version", "arm", "run_id", "identity", "results"}, "receipt"
    )
    if raw["schema_version"] != INPUT_SCHEMA or raw["arm"] not in ARMS:
        raise ProbeError("unsupported open-loop receipt identity")
    run_id = str(raw["run_id"])
    if not run_id or len(run_id) > RUN_ID_MAX_LENGTH:
        raise ProbeError("run_id must contain 1 to 128 characters")
    identity = _validate_identity(raw["identity"])
    results = _validate_results(raw["results"], case_count=int(identity["case_count"]))
    return {
        "schema_version": INPUT_SCHEMA,
        "arm": str(raw["arm"]),
        "run_id": run_id,
        "identity": identity,
        "results": results,
    }


def load_open_loop_packet(
    *,
    arm: str,
    corpus_path: Path,
    workload_path: Path,
    topology_path: Path,
    identity_path: Path,
) -> tuple[list[Case], list[Case], dict[str, Any], dict[str, str], dict[str, Any]]:
    warmup, measured = load_corpus(corpus_path)
    workload = _read_json(workload_path, "workload")
    _exact_keys(
        workload,
        {
            "schema_version",
            "arrival_process",
            "coordinated_omission_policy",
            "offered_rate_rps",
            "max_episode_lanes",
            "warmup_cases",
            "measured_cases",
            "seed",
        },
        "workload",
    )
    if (
        workload["schema_version"] != WORKLOAD_SCHEMA
        or workload["arrival_process"] != ARRIVAL_PROCESS
        or workload["coordinated_omission_policy"] != COORDINATED_OMISSION_POLICY
        or workload["max_episode_lanes"] != MAX_EPISODE_LANES
        or workload["warmup_cases"] != len(warmup)
        or len(warmup) != WARMUP_CASES
        or workload["measured_cases"] != len(measured)
        or len(measured) != MEASURED_CASES
    ):
        raise ProbeError("open-loop workload contract differs")
    _finite(workload["offered_rate_rps"], "workload.offered_rate_rps", positive=True)
    identity = _read_json(identity_path, "identity")
    _exact_keys(identity, set(IDENTITY_FIELDS), "identity")
    if (
        identity.get("measurement_scope") != MEASUREMENT_SCOPE
        or identity.get("case_count") != len(measured)
        or identity.get("seed") != workload["seed"]
        or identity.get("corpus_sha256") != _sha256(corpus_path)
        or identity.get("workload_sha256") != _sha256(workload_path)
        or identity.get("worker_topology_sha256") != _sha256(topology_path)
    ):
        raise ProbeError("open-loop identity differs from packet")
    topology = _read_json(topology_path, "topology")
    _exact_keys(
        topology, {"schema_version", "canonical_workers", "arm_worker_maps"}, "topology"
    )
    maps = topology.get("arm_worker_maps")
    if (
        topology.get("schema_version") != TOPOLOGY_SCHEMA
        or not isinstance(maps, Mapping)
        or set(maps) != set(ARMS)
        or not isinstance(maps.get(arm), Mapping)
    ):
        raise ProbeError("open-loop topology differs")
    canonical = {str(worker) for worker in topology.get("canonical_workers", [])}
    worker_map = {str(key): str(value) for key, value in maps[arm].items()}
    if len(canonical) < MIN_CANONICAL_WORKERS or set(worker_map.values()) != canonical:
        raise ProbeError("open-loop topology map is incomplete")
    return warmup, measured, identity, worker_map, workload


def poisson_schedule(
    cases: list[Case], offered_rate_rps: float, seed: int
) -> dict[str, float]:
    lanes: dict[str, list[Case]] = {}
    for case in cases:
        lanes.setdefault(case.episode_id, []).append(case)
    ordered: list[Case] = []
    for turn in range(max(map(len, lanes.values()), default=0)):
        ordered.extend(lane[turn] for lane in lanes.values() if turn < len(lane))
    rng = random.Random(seed)
    offset = 0.0
    schedule: dict[str, float] = {}
    for index, case in enumerate(ordered):
        if index:
            offset += rng.expovariate(offered_rate_rps)
        schedule[case.case_id] = offset
    return schedule


def run_open_loop_probe(
    *,
    arm: str,
    protocol: str,
    client: JSONClient,
    warmup: list[Case],
    measured: list[Case],
    identity: Mapping[str, Any],
    worker_map: Mapping[str, str],
    run_id: str,
    offered_rate_rps: float,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    select = _warm_selector(
        arm=arm,
        protocol=protocol,
        client=client,
        run_id=run_id,
        identity=identity,
        warmup=warmup,
        worker_map=worker_map,
    )
    schedule = poisson_schedule(measured, offered_rate_rps, int(identity["seed"]))
    lanes: dict[str, list[Case]] = {}
    for case in measured:
        lanes.setdefault(case.episode_id, []).append(case)
    base = clock()
    outcomes = _execute_open_loop(
        lanes=lanes,
        schedule=schedule,
        select=select,
        worker_map=worker_map,
        base=base,
        clock=clock,
        sleeper=sleeper,
    )
    return _render_receipt(
        arm=arm,
        run_id=run_id,
        identity=identity,
        measured=measured,
        schedule=schedule,
        outcomes=outcomes,
        offered_rate_rps=offered_rate_rps,
    )


def _execute_open_loop(
    *,
    lanes: Mapping[str, list[Case]],
    schedule: Mapping[str, float],
    select: Callable[[Case], tuple[str, float]],
    worker_map: Mapping[str, str],
    base: float,
    clock: Callable[[], float],
    sleeper: Callable[[float], None],
) -> dict[str, dict[str, Any]]:

    def run_lane(cases: list[Case]) -> list[dict[str, Any]]:
        outcomes: list[dict[str, Any]] = []
        for case in cases:
            deadline = base + schedule[case.case_id]
            remaining = deadline - clock()
            if remaining > 0:
                sleeper(remaining)
            actual_start = clock()
            try:
                observed, service_latency = select(case)
                canonical = worker_map.get(observed)
                if canonical is None:
                    raise ProbeError("measured request selected an unmapped worker")
                success = True
            except (OSError, ProbeError):
                canonical = ""
                service_latency = 0.0
                success = False
            completed_at = clock()
            outcomes.append(
                {
                    "case_id": case.case_id,
                    "canonical": canonical,
                    "success": success,
                    "service": service_latency,
                    "start": actual_start - base,
                    "completion": completed_at - base,
                    "scheduled": schedule[case.case_id],
                }
            )
        return outcomes

    outcomes: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_EPISODE_LANES
    ) as executor:
        futures = [executor.submit(run_lane, lane) for lane in lanes.values()]
        for future in concurrent.futures.as_completed(futures):
            for outcome in future.result():
                outcomes[outcome["case_id"]] = outcome
    return outcomes


def _render_receipt(
    *,
    arm: str,
    run_id: str,
    identity: Mapping[str, Any],
    measured: list[Case],
    schedule: Mapping[str, float],
    outcomes: Mapping[str, Mapping[str, Any]],
    offered_rate_rps: float,
) -> dict[str, Any]:
    ordered = [outcomes[case.case_id] for case in measured]
    successes = [outcome for outcome in ordered if outcome["success"]]
    if not successes:
        raise ProbeError("arm completed zero measured decisions")
    completion_offsets = sorted(outcome["completion"] for outcome in ordered)
    arrival_offsets = sorted(schedule.values())
    backlogs = [
        index + 1 - bisect.bisect_right(completion_offsets, arrival)
        for index, arrival in enumerate(arrival_offsets)
    ]
    starts = sorted(outcome["start"] for outcome in ordered)
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
    receipt = {
        "schema_version": INPUT_SCHEMA,
        "arm": arm,
        "run_id": run_id,
        "identity": dict(identity),
        "results": {
            "scheduled": len(measured),
            "completed": len(successes),
            "failed": len(measured) - len(successes),
            "offered_rate_rps": offered_rate_rps,
            "scheduled_span_seconds": arrival_offsets[-1] - arrival_offsets[0],
            "duration_seconds": duration,
            "completion_throughput_rps": len(successes) / duration,
            "achieved_start_rate_rps": achieved_start_rate,
            "service_latency_seconds": _latency(
                [outcome["service"] for outcome in successes]
            ),
            "scheduled_latency_seconds": _latency(
                [outcome["completion"] - outcome["scheduled"] for outcome in successes]
            ),
            "start_lag_seconds": _latency(
                [outcome["start"] - outcome["scheduled"] for outcome in successes]
            ),
            "max_client_backlog": max(backlogs),
            "backlog_at_final_arrival": backlogs[-1],
            "drain_seconds_after_final_arrival": max(
                0.0, completion_offsets[-1] - arrival_offsets[-1]
            ),
            "selected_worker_trace_sha256": hashlib.sha256(trace).hexdigest(),
            "provider_calls": 0,
        },
    }
    return validate_receipt(receipt)


def _warm_selector(
    *,
    arm: str,
    protocol: str,
    client: JSONClient,
    run_id: str,
    identity: Mapping[str, Any],
    warmup: list[Case],
    worker_map: Mapping[str, str],
) -> Callable[[Case], tuple[str, float]]:
    if PROTOCOL_BY_ARM.get(arm) != protocol:
        raise ProbeError("arm/protocol pair is not registered")
    select = _selector(arm=arm, client=client, run_id=run_id, identity=identity)
    for case in warmup:
        observed, _latency_value = select(case)
        if observed not in worker_map:
            raise ProbeError("warmup selected an unmapped worker")
    return select


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--protocol", required=True, choices=PROTOCOL_BY_ARM.values())
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--authorization-env", default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    authorization = ""
    if args.authorization_env:
        token = os.environ.get(args.authorization_env, "")
        if not token:
            raise SystemExit("authorization environment variable is empty")
        authorization = f"Bearer {token}"
    try:
        warmup, measured, identity, worker_map, workload = load_open_loop_packet(
            arm=args.arm,
            corpus_path=args.corpus,
            workload_path=args.workload,
            topology_path=args.topology,
            identity_path=args.identity,
        )
        report = run_open_loop_probe(
            arm=args.arm,
            protocol=args.protocol,
            client=JSONClient(
                args.base_url,
                timeout_seconds=args.timeout_seconds,
                authorization=authorization,
            ),
            warmup=warmup,
            measured=measured,
            identity=identity,
            worker_map=worker_map,
            run_id=args.run_id,
            offered_rate_rps=float(workload["offered_rate_rps"]),
        )
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    except (OSError, ProbeError, ValueError) as error:
        raise SystemExit(f"invalid open-loop arm packet: {error}") from error


if __name__ == "__main__":
    main()
