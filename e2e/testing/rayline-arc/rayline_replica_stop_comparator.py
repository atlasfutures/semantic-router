#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Strict comparator for the PERF027 staged real-replica-stop packet."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from rayline_affinity_proxy import OUTAGE_STATS_SCHEMA, STATS_SCHEMA
from rayline_open_loop_probe import INPUT_SCHEMA, _validate_results
from rayline_parity_comparator import IDENTITY_FIELDS
from rayline_replica_stop_contract import (
    EXPECTED_AFFECTED_SESSIONS,
    EXPECTED_ALL_PRIMARY_SESSIONS,
    STOP_ARMS,
    UNAVAILABLE_APP_NAME,
    UNAVAILABLE_REPLICA,
)
from rayline_replica_stop_probe import (
    EXPECTED_POST_BOUNDARY_TURNS,
    EXPECTED_PRELOAD_TURNS,
    RECEIPT_SCHEMA,
)

REPORT_SCHEMA = "rayline.vllm.replica-stop-comparison.v1"
BOUNDARY_SCHEMA = "rayline.vllm.replica-stop-boundary.v1"
TELEMETRY_SCHEMA = "rayline.vllm.arc-telemetry.v1"
EXPECTED_POOLING_REQUESTS = 36
EXPECTED_SESSIONS = 9
EXPECTED_REPLICAS = 2
EXPECTED_SURVIVOR_CLOSES = 8
EXPECTED_UNAVAILABLE_CLOSE_SKIPS = 5


class ReplicaStopComparisonError(ValueError):
    """A staged receipt violates the frozen replica-stop contract."""


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ReplicaStopComparisonError(f"{label} keys differ")


def _vector(value: Mapping[str, Any], field: str) -> list[int]:
    raw = value[field]
    if (
        not isinstance(raw, list)
        or len(raw) != EXPECTED_REPLICAS
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in raw
        )
    ):
        raise ReplicaStopComparisonError(f"{field} differs")
    return list(raw)


def _duration(value: object, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplicaStopComparisonError(f"{label} differs")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        raise ReplicaStopComparisonError(f"{label} differs")
    return result


def _receipt(value: Mapping[str, Any], *, logical_arm: str) -> dict[str, Any]:
    _exact(
        value,
        {
            "schema_version",
            "arm",
            "run_id",
            "identity",
            "stage",
            "preload",
            "boundary",
            "results",
        },
        logical_arm,
    )
    if value["schema_version"] != RECEIPT_SCHEMA or value["arm"] != "rayline_arc":
        raise ReplicaStopComparisonError("staged receipt identity differs")
    identity = value["identity"]
    if not isinstance(identity, Mapping) or set(identity) != set(IDENTITY_FIELDS):
        raise ReplicaStopComparisonError("staged source identity differs")
    stage = value["stage"]
    if not isinstance(stage, Mapping) or dict(stage) != {
        "warmup_turns": 4,
        "preload_turns": 16,
        "post_boundary_turns": 16,
        "turns_per_episode": 4,
        "preload_turns_per_episode": 2,
        "boundary_excluded_from_latency": True,
    }:
        raise ReplicaStopComparisonError("staged phase contract differs")
    preload = value["preload"]
    if not isinstance(preload, Mapping):
        raise ReplicaStopComparisonError("preload receipt is missing")
    _exact(
        preload,
        {"scheduled", "completed", "failed", "selected_worker_trace_sha256"},
        "preload",
    )
    if (
        preload["scheduled"] != EXPECTED_PRELOAD_TURNS
        or preload["completed"] != EXPECTED_PRELOAD_TURNS
        or preload["failed"] != 0
    ):
        raise ReplicaStopComparisonError("preload completion differs")
    results = value["results"]
    if not isinstance(results, Mapping):
        raise ReplicaStopComparisonError("post-boundary results are missing")
    validated = _validate_results(
        results,
        case_count=EXPECTED_POST_BOUNDARY_TURNS,
        schema_version=INPUT_SCHEMA,
    )
    boundary = value["boundary"]
    if not isinstance(boundary, Mapping):
        raise ReplicaStopComparisonError("boundary receipt is missing")
    return {
        **value,
        "identity": dict(identity),
        "stage": dict(stage),
        "preload": dict(preload),
        "boundary": dict(boundary),
        "results": validated,
    }


def _boundary(value: Mapping[str, Any], *, stopped: bool) -> dict[str, Any]:
    if stopped:
        keys = {
            "schema_version",
            "action",
            "unavailable_replica",
            "unavailable_app_name",
            "stop_command_succeeded",
            "stop_command_seconds",
            "convergence_seconds",
            "unavailable_app_stopped",
            "unavailable_containers_remaining",
            "survivor_app_deployed",
            "survivor_containers_running",
        }
        _exact(value, keys, "replica-stop boundary")
        if (
            value["schema_version"] != BOUNDARY_SCHEMA
            or value["action"] != "stop_exact_app"
            or value["unavailable_replica"] != UNAVAILABLE_REPLICA
            or value["unavailable_app_name"] != UNAVAILABLE_APP_NAME
            or value["stop_command_succeeded"] is not True
            or value["unavailable_app_stopped"] is not True
            or value["unavailable_containers_remaining"] != 0
            or value["survivor_app_deployed"] is not True
            or value["survivor_containers_running"] != 1
        ):
            raise ReplicaStopComparisonError("replica-stop boundary differs")
        result = dict(value)
        result["stop_command_seconds"] = _duration(
            value["stop_command_seconds"], "stop command"
        )
        result["convergence_seconds"] = _duration(
            value["convergence_seconds"], "stop convergence", allow_zero=True
        )
        return result
    _exact(value, {"schema_version", "action", "elapsed_seconds"}, "control boundary")
    if (
        value["schema_version"] != BOUNDARY_SCHEMA
        or value["action"] != "control_no_stop"
        or _duration(value["elapsed_seconds"], "control boundary", allow_zero=True) != 0
    ):
        raise ReplicaStopComparisonError("control boundary differs")
    return dict(value)


def _affinity(value: Mapping[str, Any], *, stopped: bool) -> dict[str, Any]:
    common = {
        "schema_version",
        "replicas",
        "requests_by_replica",
        "session_pooling_requests_by_replica",
        "session_deletes_by_replica",
        "unique_sessions_by_replica",
        "affinity_mismatches",
    }
    extra = {
        "primary_sessions_by_replica",
        "unavailable_replica",
        "outage_confirmed",
        "unavailability_detections",
        "unavailable_http_responses",
        "unavailable_transport_errors",
        "outage_failover_pooling_requests",
        "outage_failover_sessions",
        "unavailable_close_skips",
        "survivor_close_requests",
        "session_rebuild_responses",
    }
    _exact(value, common | (extra if stopped else set()), "affinity")
    if value["replicas"] != EXPECTED_REPLICAS or value["affinity_mismatches"] != 0:
        raise ReplicaStopComparisonError("affinity identity differs")
    vectors = {
        field: _vector(value, field)
        for field in (
            "requests_by_replica",
            "session_pooling_requests_by_replica",
            "session_deletes_by_replica",
            "unique_sessions_by_replica",
        )
    }
    if not stopped:
        if (
            value["schema_version"] != STATS_SCHEMA
            or vectors["session_pooling_requests_by_replica"] != [20, 16]
            or vectors["session_deletes_by_replica"] != [5, 4]
            or vectors["unique_sessions_by_replica"]
            != list(EXPECTED_ALL_PRIMARY_SESSIONS)
            or vectors["requests_by_replica"] != [25, 20]
        ):
            raise ReplicaStopComparisonError("control affinity accounting differs")
        return {**value, **vectors}
    primary = _vector(value, "primary_sessions_by_replica")
    if (
        value["schema_version"] != OUTAGE_STATS_SCHEMA
        or primary != list(EXPECTED_ALL_PRIMARY_SESSIONS)
        or vectors["session_pooling_requests_by_replica"] != [12, 24]
        or vectors["session_deletes_by_replica"] != [0, 8]
        or vectors["unique_sessions_by_replica"] != [5, 8]
        or vectors["requests_by_replica"] != [12, 32]
        or value["unavailable_replica"] != UNAVAILABLE_REPLICA
        or value["outage_confirmed"] is not True
        or value["unavailability_detections"] != EXPECTED_AFFECTED_SESSIONS
        or value["unavailable_http_responses"] + value["unavailable_transport_errors"]
        != EXPECTED_AFFECTED_SESSIONS
        or value["outage_failover_pooling_requests"] != EXPECTED_AFFECTED_SESSIONS * 2
        or value["outage_failover_sessions"] != EXPECTED_AFFECTED_SESSIONS
        or value["unavailable_close_skips"] != EXPECTED_UNAVAILABLE_CLOSE_SKIPS
        or value["survivor_close_requests"] != EXPECTED_SURVIVOR_CLOSES
        or value["session_rebuild_responses"] != EXPECTED_AFFECTED_SESSIONS
    ):
        raise ReplicaStopComparisonError("replica-stop affinity accounting differs")
    return {**value, **vectors, "primary_sessions_by_replica": primary}


def _telemetry(value: Mapping[str, Any], *, stopped: bool) -> dict[str, Any]:
    expected_actions = (
        {"created": 13, "appended": 23, "rebuilt": 0, "reused": 0}
        if stopped
        else {"created": 9, "appended": 27, "rebuilt": 0, "reused": 0}
    )
    if (
        value.get("schema_version") != TELEMETRY_SCHEMA
        or value.get("session_actions") != expected_actions
    ):
        raise ReplicaStopComparisonError("replica-stop telemetry differs")
    tokens = value.get("tokens")
    if not isinstance(tokens, Mapping):
        raise ReplicaStopComparisonError("replica-stop token telemetry is missing")
    for field in ("appended", "retained", "full", "serialized"):
        item = tokens.get(field)
        if (
            not isinstance(item, Mapping)
            or item.get("count") != EXPECTED_POOLING_REQUESTS
            or isinstance(item.get("sum"), bool)
            or not isinstance(item.get("sum"), int)
            or item["sum"] < 0
        ):
            raise ReplicaStopComparisonError(f"replica-stop {field} telemetry differs")
    return dict(value)


def _identity(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(receipt["identity"][field] for field in IDENTITY_FIELDS)


def compare_replica_stop(
    raw_receipts: Mapping[str, Mapping[str, Any]],
    raw_affinity: Mapping[str, Mapping[str, Any]],
    raw_telemetry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = set(STOP_ARMS)
    if (
        set(raw_receipts) != expected
        or set(raw_affinity) != expected
        or set(raw_telemetry) != expected
    ):
        raise ReplicaStopComparisonError("replica-stop arm sets differ")
    receipts = {arm: _receipt(raw_receipts[arm], logical_arm=arm) for arm in STOP_ARMS}
    if (
        _identity(receipts[STOP_ARMS[0]]) != _identity(receipts[STOP_ARMS[1]])
        or receipts[STOP_ARMS[0]]["run_id"] != receipts[STOP_ARMS[1]]["run_id"]
    ):
        raise ReplicaStopComparisonError("replica-stop arm identity differs")
    boundaries = {
        STOP_ARMS[0]: _boundary(receipts[STOP_ARMS[0]]["boundary"], stopped=False),
        STOP_ARMS[1]: _boundary(receipts[STOP_ARMS[1]]["boundary"], stopped=True),
    }
    affinity = {
        STOP_ARMS[0]: _affinity(raw_affinity[STOP_ARMS[0]], stopped=False),
        STOP_ARMS[1]: _affinity(raw_affinity[STOP_ARMS[1]], stopped=True),
    }
    if (
        affinity[STOP_ARMS[0]]["unique_sessions_by_replica"]
        != affinity[STOP_ARMS[1]]["primary_sessions_by_replica"]
    ):
        raise ReplicaStopComparisonError("replica-stop primary placement differs")
    telemetry = {
        STOP_ARMS[0]: _telemetry(raw_telemetry[STOP_ARMS[0]], stopped=False),
        STOP_ARMS[1]: _telemetry(raw_telemetry[STOP_ARMS[1]], stopped=True),
    }
    preload_traces = {
        receipt["preload"]["selected_worker_trace_sha256"]
        for receipt in receipts.values()
    }
    post_traces = {
        receipt["results"]["selected_worker_trace_sha256"]
        for receipt in receipts.values()
    }
    all_completed = all(
        receipt["results"]["completed"] == EXPECTED_POST_BOUNDARY_TURNS
        and receipt["results"]["failed"] == 0
        and receipt["results"]["provider_calls"] == 0
        for receipt in receipts.values()
    )
    integrity = {
        "all_completed": all_completed,
        "preload_trace_match": len(preload_traces) == 1,
        "post_boundary_trace_match": len(post_traces) == 1,
        "provider_calls_zero": all(
            receipt["results"]["provider_calls"] == 0 for receipt in receipts.values()
        ),
    }
    passed = all(integrity.values())
    control = receipts[STOP_ARMS[0]]["results"]
    stopped = receipts[STOP_ARMS[1]]["results"]
    control_tokens = telemetry[STOP_ARMS[0]]["tokens"]
    stopped_tokens = telemetry[STOP_ARMS[1]]["tokens"]
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "integrity": integrity,
        "preload_trace_sha256": (
            next(iter(preload_traces)) if len(preload_traces) == 1 else None
        ),
        "post_boundary_trace_sha256": (
            next(iter(post_traces)) if len(post_traces) == 1 else None
        ),
        "boundaries": boundaries,
        "affinity": affinity,
        "telemetry": telemetry,
        "replica_stop_vs_control": {
            "completion_throughput_ratio": stopped["completion_throughput_rps"]
            / control["completion_throughput_rps"],
            "service_latency_ratio": {
                percentile: stopped["service_latency_seconds"][percentile]
                / control["service_latency_seconds"][percentile]
                for percentile in ("p50", "p95", "p99")
            },
            "scheduled_latency_ratio": {
                percentile: stopped["scheduled_latency_seconds"][percentile]
                / control["scheduled_latency_seconds"][percentile]
                for percentile in ("p50", "p95", "p99")
            },
            "drain_ratio": stopped["drain_seconds_after_final_arrival"]
            / control["drain_seconds_after_final_arrival"],
            "final_arrival_backlog_delta": stopped["backlog_at_final_arrival"]
            - control["backlog_at_final_arrival"],
            "appended_token_work_ratio": stopped_tokens["appended"]["sum"]
            / control_tokens["appended"]["sum"],
            "retained_token_ratio": stopped_tokens["retained"]["sum"]
            / control_tokens["retained"]["sum"],
        },
    }
