#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Strict comparator for the PERF025 forced affinity-loss packet."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rayline_failover_contract import FAILOVER_AFTER_POOLING, FAILOVER_ARMS
from rayline_open_loop_probe import validate_receipt
from rayline_parity_comparator import IDENTITY_FIELDS

REPORT_SCHEMA = "rayline.vllm.affinity-failover-comparison.v1"
STICKY_AFFINITY_SCHEMA = "rayline.vllm.episode-affinity-stats.v1"
FAILOVER_AFFINITY_SCHEMA = "rayline.vllm.episode-affinity-failover-stats.v1"
TELEMETRY_SCHEMA = "rayline.vllm.arc-telemetry.v1"
EXPECTED_POOLING_REQUESTS = 36
EXPECTED_SESSIONS = 9
EXPECTED_MEASURED_CASES = 32
EXPECTED_REPLICAS = 2
EXPECTED_FAILOVER_POOLING_REQUESTS = 18


class FailoverComparisonError(ValueError):
    """A failover receipt is missing or violates the frozen schema."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise FailoverComparisonError(f"{label} keys differ from the frozen schema")


def _integer_vector(
    value: Mapping[str, Any], field: str, *, replicas: int, label: str
) -> list[int]:
    raw = value[field]
    if (
        not isinstance(raw, list)
        or len(raw) != replicas
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in raw
        )
    ):
        raise FailoverComparisonError(f"{label} {field} differs")
    return list(raw)


def _validate_affinity(value: Mapping[str, Any], *, failover: bool) -> dict[str, Any]:
    label = "forced_failover" if failover else "sticky"
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
        "failover_after_pooling",
        "failover_pooling_requests",
        "failover_sessions",
        "close_fanout_requests",
        "session_rebuild_responses",
    }
    _exact_keys(value, common | (extra if failover else set()), label)
    expected_schema = FAILOVER_AFFINITY_SCHEMA if failover else STICKY_AFFINITY_SCHEMA
    if (
        value["schema_version"] != expected_schema
        or value["replicas"] != EXPECTED_REPLICAS
    ):
        raise FailoverComparisonError(f"{label} affinity identity differs")
    vectors = {
        field: _integer_vector(value, field, replicas=EXPECTED_REPLICAS, label=label)
        for field in (
            "requests_by_replica",
            "session_pooling_requests_by_replica",
            "session_deletes_by_replica",
            "unique_sessions_by_replica",
        )
    }
    expected_deletes = EXPECTED_SESSIONS * (2 if failover else 1)
    expected_unique = EXPECTED_SESSIONS * (2 if failover else 1)
    if (
        sum(vectors["session_pooling_requests_by_replica"]) != EXPECTED_POOLING_REQUESTS
        or sum(vectors["session_deletes_by_replica"]) != expected_deletes
        or sum(vectors["unique_sessions_by_replica"]) != expected_unique
        or sum(vectors["requests_by_replica"])
        != EXPECTED_POOLING_REQUESTS + expected_deletes
        or value["affinity_mismatches"] != 0
        or any(item == 0 for item in vectors["unique_sessions_by_replica"])
    ):
        raise FailoverComparisonError(f"{label} affinity accounting differs")
    if failover and (
        value["failover_after_pooling"] != FAILOVER_AFTER_POOLING
        or value["failover_pooling_requests"] != EXPECTED_FAILOVER_POOLING_REQUESTS
        or value["failover_sessions"] != EXPECTED_SESSIONS
        or value["close_fanout_requests"] != EXPECTED_SESSIONS * EXPECTED_REPLICAS
        or value["session_rebuild_responses"] != EXPECTED_SESSIONS
    ):
        raise FailoverComparisonError("forced failover accounting differs")
    return {**value, **vectors}


def _validate_telemetry(value: Mapping[str, Any], *, failover: bool) -> dict[str, Any]:
    if value.get("schema_version") != TELEMETRY_SCHEMA:
        raise FailoverComparisonError("ARC telemetry schema differs")
    actions = value.get("session_actions")
    expected = (
        {"created": 18, "appended": 18, "rebuilt": 0, "reused": 0}
        if failover
        else {"created": 9, "appended": 27, "rebuilt": 0, "reused": 0}
    )
    if actions != expected:
        raise FailoverComparisonError("ARC session-action accounting differs")
    tokens = value.get("tokens")
    if not isinstance(tokens, Mapping):
        raise FailoverComparisonError("ARC token telemetry is missing")
    for field in ("appended", "retained", "full", "serialized"):
        item = tokens.get(field)
        if (
            not isinstance(item, Mapping)
            or item.get("count") != EXPECTED_POOLING_REQUESTS
            or isinstance(item.get("sum"), bool)
            or not isinstance(item.get("sum"), int)
            or item["sum"] < 0
        ):
            raise FailoverComparisonError(f"ARC {field} telemetry differs")
    return dict(value)


def _identity(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(receipt["identity"][field] for field in IDENTITY_FIELDS)


def compare_failover(
    raw_receipts: Mapping[str, Mapping[str, Any]],
    raw_affinity: Mapping[str, Mapping[str, Any]],
    raw_telemetry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_arms = set(FAILOVER_ARMS)
    if (
        set(raw_receipts) != expected_arms
        or set(raw_affinity) != expected_arms
        or set(raw_telemetry) != expected_arms
    ):
        raise FailoverComparisonError("failover arm sets differ")
    receipts = {arm: validate_receipt(dict(raw_receipts[arm])) for arm in FAILOVER_ARMS}
    if any(receipt["arm"] != "rayline_arc" for receipt in receipts.values()):
        raise FailoverComparisonError("failover packet did not exercise ARC")
    if _identity(receipts[FAILOVER_ARMS[0]]) != _identity(receipts[FAILOVER_ARMS[1]]):
        raise FailoverComparisonError("failover arm identities differ")
    traces = {
        receipt["results"]["selected_worker_trace_sha256"]
        for receipt in receipts.values()
    }
    trace_match = len(traces) == 1
    all_completed = all(
        receipt["results"]["completed"] == EXPECTED_MEASURED_CASES
        and receipt["results"]["failed"] == 0
        and receipt["results"]["provider_calls"] == 0
        for receipt in receipts.values()
    )
    affinity = {
        FAILOVER_ARMS[0]: _validate_affinity(
            raw_affinity[FAILOVER_ARMS[0]], failover=False
        ),
        FAILOVER_ARMS[1]: _validate_affinity(
            raw_affinity[FAILOVER_ARMS[1]], failover=True
        ),
    }
    telemetry = {
        FAILOVER_ARMS[0]: _validate_telemetry(
            raw_telemetry[FAILOVER_ARMS[0]], failover=False
        ),
        FAILOVER_ARMS[1]: _validate_telemetry(
            raw_telemetry[FAILOVER_ARMS[1]], failover=True
        ),
    }
    sticky = receipts[FAILOVER_ARMS[0]]["results"]
    failed_over = receipts[FAILOVER_ARMS[1]]["results"]
    sticky_tokens = telemetry[FAILOVER_ARMS[0]]["tokens"]
    failover_tokens = telemetry[FAILOVER_ARMS[1]]["tokens"]
    integrity = {
        "all_completed": all_completed,
        "trace_match": trace_match,
        "provider_calls_zero": all(
            receipt["results"]["provider_calls"] == 0 for receipt in receipts.values()
        ),
    }
    passed = all(integrity.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "selected_worker_trace_sha256": next(iter(traces)) if trace_match else None,
        "integrity": integrity,
        "affinity": affinity,
        "telemetry": telemetry,
        "forced_failover_vs_sticky": {
            "completion_throughput_ratio": (
                failed_over["completion_throughput_rps"]
                / sticky["completion_throughput_rps"]
            ),
            "service_latency_ratio": {
                percentile: failed_over["service_latency_seconds"][percentile]
                / sticky["service_latency_seconds"][percentile]
                for percentile in ("p50", "p95", "p99")
            },
            "scheduled_latency_ratio": {
                percentile: failed_over["scheduled_latency_seconds"][percentile]
                / sticky["scheduled_latency_seconds"][percentile]
                for percentile in ("p50", "p95", "p99")
            },
            "drain_ratio": failed_over["drain_seconds_after_final_arrival"]
            / sticky["drain_seconds_after_final_arrival"],
            "final_arrival_backlog_delta": (
                failed_over["backlog_at_final_arrival"]
                - sticky["backlog_at_final_arrival"]
            ),
            "appended_token_work_ratio": failover_tokens["appended"]["sum"]
            / sticky_tokens["appended"]["sum"],
            "retained_token_ratio": failover_tokens["retained"]["sum"]
            / sticky_tokens["retained"]["sum"],
        },
    }
