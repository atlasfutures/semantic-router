#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Strict integrity and capacity comparator for the DYN006 paid cell."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from rayline_dynamic_stop_contract import (
    DRAINING_MEMBERSHIP_REVISION,
    DYN006_RUN_ID,
    DYNAMIC_STOP_ARMS,
    ENCODER_REPLICA_IDS,
    EXPECTED_AFFECTED_SESSIONS,
    EXPECTED_POST_STOP_OWNERS,
    EXPECTED_PRE_BOUNDARY_OWNERS,
    REGISTERED_MEMBERSHIP_REVISION,
    REMOVED_MEMBERSHIP_REVISION,
    SESSION_NAMESPACE,
    SURVIVOR_COUNT,
)
from rayline_dynamic_telemetry import DYNAMIC_TELEMETRY_SCHEMA
from rayline_open_loop_probe import (
    INPUT_SCHEMA,
    _validate_identity,
    _validate_results,
)
from rayline_parity_comparator import IDENTITY_FIELDS
from rayline_replica_stop_probe import (
    EXPECTED_POST_BOUNDARY_TURNS,
    EXPECTED_PRELOAD_TURNS,
    RECEIPT_SCHEMA,
)

REPORT_SCHEMA = "rayline.vllm.dynamic-capacity-stop-comparison.v1"
CONTROL_BOUNDARY_SCHEMA = "rayline.vllm.dynamic-control-boundary.v1"
STOP_BOUNDARY_SCHEMA = "rayline.vllm.dynamic-stop-boundary.v1"
EXPECTED_SELECTIONS = 47
EXPECTED_SESSIONS = 10
REPLICA_COUNT = len(ENCODER_REPLICA_IDS)
EXPECTED_RUN_ID = f"{DYN006_RUN_ID}:r030:{SESSION_NAMESPACE}"
TOKEN_KINDS = ("full", "serialized", "retained", "appended", "cached", "truncated")
SHA256_LENGTH = 64
MINIMUM_THROUGHPUT_RATIO = 0.75
MAXIMUM_LATENCY_RATIO = 2.0


class DynamicStopComparisonError(ValueError):
    """A DYN006 aggregate receipt violates its frozen contract."""


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise DynamicStopComparisonError(f"{label} keys differ")


def _vector(value: object, label: str) -> list[int]:
    if (
        not isinstance(value, list)
        or len(value) != REPLICA_COUNT
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise DynamicStopComparisonError(f"{label} differs")
    return list(value)


def _digest(value: object, label: str) -> str:
    rendered = str(value)
    if len(rendered) != SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in rendered
    ):
        raise DynamicStopComparisonError(f"{label} differs")
    return rendered


def _duration(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise DynamicStopComparisonError(f"{label} differs")
    return float(value)


def _receipt(value: Mapping[str, Any], logical_arm: str) -> dict[str, Any]:
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
    if (
        value["schema_version"] != RECEIPT_SCHEMA
        or value["arm"] != "rayline_arc"
        or value["run_id"] != EXPECTED_RUN_ID
    ):
        raise DynamicStopComparisonError("dynamic staged receipt identity differs")
    identity_raw = value["identity"]
    if not isinstance(identity_raw, Mapping) or set(identity_raw) != set(
        IDENTITY_FIELDS
    ):
        raise DynamicStopComparisonError("dynamic source identity differs")
    identity = _validate_identity(identity_raw)
    stage = value["stage"]
    if not isinstance(stage, Mapping) or dict(stage) != {
        "warmup_turns": 4,
        "preload_turns": 16,
        "post_boundary_turns": 16,
        "turns_per_episode": 4,
        "preload_turns_per_episode": 2,
        "boundary_excluded_from_latency": True,
    }:
        raise DynamicStopComparisonError("dynamic staged phase differs")
    preload = value["preload"]
    if not isinstance(preload, Mapping):
        raise DynamicStopComparisonError("dynamic preload receipt is missing")
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
        raise DynamicStopComparisonError("dynamic preload completion differs")
    preload_trace = _digest(
        preload["selected_worker_trace_sha256"],
        "dynamic preload trace",
    )
    results = _validate_results(
        value["results"],
        case_count=EXPECTED_POST_BOUNDARY_TURNS,
        schema_version=INPUT_SCHEMA,
    )
    if (
        results["completed"] != EXPECTED_POST_BOUNDARY_TURNS
        or results["failed"]
        or results["provider_calls"] != 0
    ):
        raise DynamicStopComparisonError("dynamic post-boundary completion differs")
    return {
        **dict(value),
        "identity": identity,
        "preload": {**dict(preload), "selected_worker_trace_sha256": preload_trace},
        "results": results,
    }


def _telemetry(value: Mapping[str, Any], *, treatment: bool) -> dict[str, Any]:
    _exact(
        value,
        {
            "schema_version",
            "component_ready",
            "session_actions",
            "tokens",
            "cache_miss_tokens",
            "replica_routes",
            "session_closes",
        },
        "dynamic telemetry",
    )
    if value.get("schema_version") != DYNAMIC_TELEMETRY_SCHEMA:
        raise DynamicStopComparisonError("dynamic telemetry schema differs")
    actions = value.get("session_actions")
    routes = value.get("replica_routes")
    closes = value.get("session_closes")
    tokens = value.get("tokens")
    if not all(isinstance(item, Mapping) for item in (actions, routes, closes, tokens)):
        raise DynamicStopComparisonError("dynamic telemetry aggregates are missing")
    _exact(actions, {"created", "appended", "rebuilt", "reused"}, "session actions")
    _exact(routes, {"direct", "failover"}, "replica routes")
    _exact(closes, {"closed", "unavailable", "failed"}, "session closes")
    _exact(tokens, set(TOKEN_KINDS), "token telemetry")
    expected_created = 10 + (EXPECTED_AFFECTED_SESSIONS if treatment else 0)
    expected_appended = EXPECTED_SELECTIONS - expected_created
    if (
        actions.get("created") != expected_created
        or actions.get("appended") != expected_appended
        or actions.get("rebuilt", 0) != 0
        or actions.get("reused", 0) != 0
    ):
        raise DynamicStopComparisonError("dynamic session actions differ")
    expected_failovers = EXPECTED_AFFECTED_SESSIONS if treatment else 0
    if (
        routes.get("direct") != EXPECTED_SELECTIONS - expected_failovers
        or routes.get("failover") != expected_failovers
    ):
        raise DynamicStopComparisonError("dynamic replica route accounting differs")
    if (
        closes.get("closed") != EXPECTED_SESSIONS
        or closes.get("unavailable") != expected_failovers
        or closes.get("failed") != 0
    ):
        raise DynamicStopComparisonError("dynamic close accounting differs")
    if any(
        not isinstance(tokens.get(kind), Mapping)
        or tokens[kind].get("count") != EXPECTED_SELECTIONS
        for kind in TOKEN_KINDS
    ):
        raise DynamicStopComparisonError("dynamic token telemetry differs")
    cache_miss = value.get("cache_miss_tokens")
    if (
        value.get("component_ready") != 1
        or not isinstance(cache_miss, Mapping)
        or set(cache_miss) != {"count", "sum"}
        or cache_miss.get("count") != EXPECTED_SELECTIONS
    ):
        raise DynamicStopComparisonError("dynamic readiness telemetry differs")
    return dict(value)


def _lifecycle(value: Mapping[str, Any], *, treatment: bool) -> dict[str, Any]:
    _exact(
        value,
        {
            "pre_boundary_owners",
            "post_boundary_owners",
            "capacity_registration",
            "final_membership",
            "episode_states_cleared",
        },
        "dynamic lifecycle",
    )
    if _vector(value["pre_boundary_owners"], "pre-boundary owners") != list(
        EXPECTED_PRE_BOUNDARY_OWNERS
    ):
        raise DynamicStopComparisonError("pre-boundary owner placement differs")
    expected_post = (
        list(EXPECTED_POST_STOP_OWNERS)
        if treatment
        else list(EXPECTED_PRE_BOUNDARY_OWNERS)
    )
    if _vector(value["post_boundary_owners"], "post-boundary owners") != expected_post:
        raise DynamicStopComparisonError("post-boundary owner placement differs")
    if value["capacity_registration"] != {
        "command": "register",
        "replica_id": "encoder-c",
        "revision": REGISTERED_MEMBERSHIP_REVISION,
        "active": 3,
        "draining": 0,
    }:
        raise DynamicStopComparisonError("capacity registration differs")
    expected_membership = {
        "revision": (
            REMOVED_MEMBERSHIP_REVISION if treatment else REGISTERED_MEMBERSHIP_REVISION
        ),
        "active": SURVIVOR_COUNT if treatment else REPLICA_COUNT,
        "draining": 0,
    }
    membership = value["final_membership"]
    if not isinstance(membership, Mapping) or any(
        membership.get(field) != expected
        for field, expected in expected_membership.items()
    ):
        raise DynamicStopComparisonError("final membership differs")
    expected_member_ids = ENCODER_REPLICA_IDS[1:] if treatment else ENCODER_REPLICA_IDS
    members = membership.get("members")
    if (
        membership.get("command") != "status"
        or not isinstance(members, list)
        or members
        != [{"id": replica_id, "state": "active"} for replica_id in expected_member_ids]
    ):
        raise DynamicStopComparisonError("final membership members differ")
    if value["episode_states_cleared"] != EXPECTED_SESSIONS:
        raise DynamicStopComparisonError("episode-state cleanup differs")
    return dict(value)


def _ratio(numerator: float, denominator: float, label: str) -> float:
    if not all(
        math.isfinite(value) and value > 0 for value in (numerator, denominator)
    ):
        raise DynamicStopComparisonError(f"{label} inputs differ")
    return numerator / denominator


def _validate_boundaries(
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
) -> None:
    control_boundary = control["boundary"]
    treatment_boundary = treatment["boundary"]
    if not isinstance(control_boundary, Mapping) or dict(control_boundary) != {
        "schema_version": CONTROL_BOUNDARY_SCHEMA,
        "action": "control_no_mutation",
    }:
        raise DynamicStopComparisonError("dynamic control boundary differs")
    if (
        not isinstance(treatment_boundary, Mapping)
        or treatment_boundary.get("schema_version") != STOP_BOUNDARY_SCHEMA
        or treatment_boundary.get("action") != "drain_then_stop_exact_app"
        or treatment_boundary.get("drain_revision") != DRAINING_MEMBERSHIP_REVISION
        or treatment_boundary.get("unavailable_replica_id") != "encoder-a"
        or treatment_boundary.get("unavailable_app_name")
        != "rayline-arc-session-encoder-a"
        or treatment_boundary.get("stop_command_succeeded") is not True
        or treatment_boundary.get("unavailable_app_stopped") is not True
        or treatment_boundary.get("unavailable_containers_remaining") != 0
        or treatment_boundary.get("survivor_apps_deployed") is not True
        or treatment_boundary.get("survivor_containers_running") != SURVIVOR_COUNT
    ):
        raise DynamicStopComparisonError("dynamic stop boundary differs")
    _exact(
        treatment_boundary,
        {
            "schema_version",
            "action",
            "drain_revision",
            "unavailable_replica_id",
            "unavailable_app_name",
            "stop_command_succeeded",
            "stop_command_seconds",
            "convergence_seconds",
            "unavailable_app_stopped",
            "unavailable_containers_remaining",
            "survivor_apps_deployed",
            "survivor_containers_running",
        },
        "dynamic stop boundary",
    )
    _duration(treatment_boundary["stop_command_seconds"], "stop command duration")
    _duration(treatment_boundary["convergence_seconds"], "stop convergence duration")


def _capacity_ratios(
    control_results: Mapping[str, Any],
    treatment_results: Mapping[str, Any],
) -> tuple[float, float, float, bool]:
    throughput_ratio = _ratio(
        treatment_results["completion_throughput_rps"],
        control_results["completion_throughput_rps"],
        "throughput ratio",
    )
    p50_ratio = _ratio(
        treatment_results["service_latency_seconds"]["p50"],
        control_results["service_latency_seconds"]["p50"],
        "p50 ratio",
    )
    p95_ratio = _ratio(
        treatment_results["service_latency_seconds"]["p95"],
        control_results["service_latency_seconds"]["p95"],
        "p95 ratio",
    )
    passed = (
        throughput_ratio >= MINIMUM_THROUGHPUT_RATIO
        and p50_ratio <= MAXIMUM_LATENCY_RATIO
        and p95_ratio <= MAXIMUM_LATENCY_RATIO
    )
    return throughput_ratio, p50_ratio, p95_ratio, passed


def compare_dynamic_stop(
    receipts: Mapping[str, Mapping[str, Any]],
    telemetry: Mapping[str, Mapping[str, Any]],
    lifecycle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if (
        set(receipts) != set(DYNAMIC_STOP_ARMS)
        or set(telemetry) != set(DYNAMIC_STOP_ARMS)
        or set(lifecycle) != set(DYNAMIC_STOP_ARMS)
    ):
        raise DynamicStopComparisonError("dynamic arm sets differ")
    control_name, treatment_name = DYNAMIC_STOP_ARMS
    control = _receipt(receipts[control_name], control_name)
    treatment = _receipt(receipts[treatment_name], treatment_name)
    if control["identity"] != treatment["identity"]:
        raise DynamicStopComparisonError("dynamic source identities differ")
    if (
        control["preload"]["selected_worker_trace_sha256"]
        != treatment["preload"]["selected_worker_trace_sha256"]
        or control["results"]["selected_worker_trace_sha256"]
        != treatment["results"]["selected_worker_trace_sha256"]
    ):
        raise DynamicStopComparisonError("dynamic selected-worker traces differ")
    _validate_boundaries(control, treatment)
    telemetry_out = {
        control_name: _telemetry(telemetry[control_name], treatment=False),
        treatment_name: _telemetry(telemetry[treatment_name], treatment=True),
    }
    lifecycle_out = {
        control_name: _lifecycle(lifecycle[control_name], treatment=False),
        treatment_name: _lifecycle(lifecycle[treatment_name], treatment=True),
    }
    control_results = control["results"]
    treatment_results = treatment["results"]
    throughput_ratio, p50_ratio, p95_ratio, capacity_passed = _capacity_ratios(
        control_results,
        treatment_results,
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if capacity_passed else "failed",
        "integrity_status": "passed",
        "capacity_gate": {
            "status": "passed" if capacity_passed else "failed",
            "minimum_throughput_ratio": MINIMUM_THROUGHPUT_RATIO,
            "maximum_latency_ratio": MAXIMUM_LATENCY_RATIO,
        },
        "arms": {name: receipts[name]["results"] for name in DYNAMIC_STOP_ARMS},
        "dynamic_stop_vs_control": {
            "completion_throughput_ratio": throughput_ratio,
            "service_latency_p50_ratio": p50_ratio,
            "service_latency_p95_ratio": p95_ratio,
        },
        "telemetry": telemetry_out,
        "lifecycle": lifecycle_out,
        "provider_calls": 0,
        "release_qualification_1000_executed": False,
    }
