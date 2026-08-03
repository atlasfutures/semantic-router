# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

comparator = importlib.import_module("rayline_dynamic_stop_comparator")
contract = importlib.import_module("rayline_dynamic_stop_contract")
packet = importlib.import_module("rayline_open_loop_packet")
parity = importlib.import_module("rayline_parity_comparator")
probe = importlib.import_module("rayline_replica_stop_probe")
telemetry_contract = importlib.import_module("rayline_dynamic_telemetry")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity() -> dict[str, object]:
    identity: dict[str, object] = {}
    for field in parity.IDENTITY_FIELDS:
        if field == "measurement_scope":
            identity[field] = packet.MEASUREMENT_SCOPE
        elif field == "case_count":
            identity[field] = 32
        elif field == "seed":
            identity[field] = 20260730
        elif field.endswith("sha256"):
            identity[field] = _digest(field)
        else:
            identity[field] = field
    return identity


def _results(throughput: float, latency: float) -> dict[str, object]:
    latencies = {"p50": latency, "p95": latency * 2, "p99": latency * 3}
    return {
        "scheduled": 16,
        "completed": 16,
        "failed": 0,
        "offered_rate_rps": 0.3,
        "realized_arrival_rate_rps": 0.3,
        "scheduled_span_seconds": 50.0,
        "duration_seconds": 60.0,
        "completion_throughput_rps": throughput,
        "achieved_start_rate_rps": 0.3,
        "service_latency_seconds": latencies,
        "scheduled_latency_seconds": latencies,
        "start_lag_seconds": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
        "max_client_backlog": 4,
        "backlog_at_final_arrival": 4,
        "drain_seconds_after_final_arrival": 10.0,
        "selected_worker_trace_sha256": _digest("post-trace"),
        "provider_calls": 0,
    }


def _receipt(
    *, treatment: bool, throughput: float, latency: float
) -> dict[str, object]:
    boundary = (
        {
            "schema_version": comparator.STOP_BOUNDARY_SCHEMA,
            "action": "drain_then_stop_exact_app",
            "drain_revision": contract.DRAINING_MEMBERSHIP_REVISION,
            "unavailable_replica_id": contract.UNAVAILABLE_REPLICA_ID,
            "unavailable_app_name": contract.UNAVAILABLE_APP_NAME,
            "stop_command_succeeded": True,
            "stop_command_seconds": 1.0,
            "convergence_seconds": 2.0,
            "unavailable_app_stopped": True,
            "unavailable_containers_remaining": 0,
            "survivor_apps_deployed": True,
            "survivor_containers_running": contract.SURVIVOR_COUNT,
        }
        if treatment
        else {
            "schema_version": comparator.CONTROL_BOUNDARY_SCHEMA,
            "action": "control_no_mutation",
        }
    )
    return {
        "schema_version": probe.RECEIPT_SCHEMA,
        "arm": "rayline_arc",
        "run_id": comparator.EXPECTED_RUN_ID,
        "identity": _identity(),
        "stage": {
            "warmup_turns": 4,
            "preload_turns": 16,
            "post_boundary_turns": 16,
            "turns_per_episode": 4,
            "preload_turns_per_episode": 2,
            "boundary_excluded_from_latency": True,
        },
        "preload": {
            "scheduled": 16,
            "completed": 16,
            "failed": 0,
            "selected_worker_trace_sha256": _digest("preload-trace"),
        },
        "boundary": boundary,
        "results": _results(throughput, latency),
    }


def _telemetry(*, treatment: bool) -> dict[str, object]:
    failovers = contract.EXPECTED_AFFECTED_SESSIONS if treatment else 0
    created = comparator.EXPECTED_SESSIONS + failovers
    return {
        "schema_version": telemetry_contract.DYNAMIC_TELEMETRY_SCHEMA,
        "component_ready": 1,
        "session_actions": {
            "created": created,
            "appended": comparator.EXPECTED_SELECTIONS - created,
            "rebuilt": 0,
            "reused": 0,
        },
        "replica_routes": {
            "direct": comparator.EXPECTED_SELECTIONS - failovers,
            "failover": failovers,
        },
        "session_closes": {
            "closed": comparator.EXPECTED_SESSIONS,
            "unavailable": failovers,
            "failed": 0,
        },
        "tokens": {
            name: {"count": comparator.EXPECTED_SELECTIONS, "sum": 100}
            for name in comparator.TOKEN_KINDS
        },
        "cache_miss_tokens": {
            "count": comparator.EXPECTED_SELECTIONS,
            "sum": 0,
        },
    }


def _lifecycle(*, treatment: bool) -> dict[str, object]:
    return {
        "pre_boundary_owners": list(contract.EXPECTED_PRE_BOUNDARY_OWNERS),
        "post_boundary_owners": list(
            contract.EXPECTED_POST_STOP_OWNERS
            if treatment
            else contract.EXPECTED_PRE_BOUNDARY_OWNERS
        ),
        "capacity_registration": {
            "command": "register",
            "replica_id": "encoder-c",
            "revision": contract.REGISTERED_MEMBERSHIP_REVISION,
            "active": 3,
            "draining": 0,
        },
        "final_membership": {
            "command": "status",
            "revision": (
                contract.REMOVED_MEMBERSHIP_REVISION
                if treatment
                else contract.REGISTERED_MEMBERSHIP_REVISION
            ),
            "active": contract.SURVIVOR_COUNT if treatment else 3,
            "draining": 0,
            "members": [
                {"id": replica_id, "state": "active"}
                for replica_id in (
                    contract.ENCODER_REPLICA_IDS[1:]
                    if treatment
                    else contract.ENCODER_REPLICA_IDS
                )
            ],
        },
        "episode_states_cleared": comparator.EXPECTED_SESSIONS,
    }


def _inputs():
    control, treatment = contract.DYNAMIC_STOP_ARMS
    return (
        {
            control: _receipt(treatment=False, throughput=0.20, latency=1.0),
            treatment: _receipt(treatment=True, throughput=0.18, latency=1.5),
        },
        {
            control: _telemetry(treatment=False),
            treatment: _telemetry(treatment=True),
        },
        {
            control: _lifecycle(treatment=False),
            treatment: _lifecycle(treatment=True),
        },
    )


def test_dynamic_stop_comparison_passes_capacity_and_integrity() -> None:
    report = comparator.compare_dynamic_stop(*_inputs())

    assert report["status"] == "passed"
    assert report["integrity_status"] == "passed"
    assert report["dynamic_stop_vs_control"][
        "completion_throughput_ratio"
    ] == pytest.approx(0.9)
    assert report["dynamic_stop_vs_control"][
        "service_latency_p95_ratio"
    ] == pytest.approx(1.5)


def test_dynamic_stop_comparison_rejects_owner_drift() -> None:
    receipts, telemetry, lifecycle = _inputs()
    broken = copy.deepcopy(lifecycle)
    broken[contract.DYNAMIC_STOP_ARMS[1]]["post_boundary_owners"] = [0, 5, 3]

    with pytest.raises(comparator.DynamicStopComparisonError, match="placement"):
        comparator.compare_dynamic_stop(receipts, telemetry, broken)


def test_dynamic_stop_comparison_reports_capacity_failure() -> None:
    receipts, telemetry, lifecycle = _inputs()
    receipts[contract.DYNAMIC_STOP_ARMS[1]]["results"] = _results(0.10, 2.5)

    report = comparator.compare_dynamic_stop(receipts, telemetry, lifecycle)

    assert report["status"] == "failed"
    assert report["integrity_status"] == "passed"
    assert report["capacity_gate"]["status"] == "failed"
