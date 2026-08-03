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

comparator = importlib.import_module("rayline_replica_stop_comparator")
contract = importlib.import_module("rayline_replica_stop_contract")
packet = importlib.import_module("rayline_open_loop_packet")
parity = importlib.import_module("rayline_parity_comparator")
probe = importlib.import_module("rayline_replica_stop_probe")


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


def _results(throughput: float) -> dict[str, object]:
    latency = {"p50": 1.0, "p95": 2.0, "p99": 3.0}
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
        "service_latency_seconds": latency,
        "scheduled_latency_seconds": latency,
        "start_lag_seconds": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
        "max_client_backlog": 4,
        "backlog_at_final_arrival": 4,
        "drain_seconds_after_final_arrival": 10.0,
        "selected_worker_trace_sha256": _digest("post-trace"),
        "provider_calls": 0,
    }


def _receipt(*, stopped: bool) -> dict[str, object]:
    boundary = (
        {
            "schema_version": comparator.BOUNDARY_SCHEMA,
            "action": "stop_exact_app",
            "unavailable_replica": 0,
            "unavailable_app_name": contract.UNAVAILABLE_APP_NAME,
            "stop_command_succeeded": True,
            "stop_command_seconds": 1.0,
            "convergence_seconds": 2.0,
            "unavailable_app_stopped": True,
            "unavailable_containers_remaining": 0,
            "survivor_app_deployed": True,
            "survivor_containers_running": 1,
        }
        if stopped
        else {
            "schema_version": comparator.BOUNDARY_SCHEMA,
            "action": "control_no_stop",
            "elapsed_seconds": 0.0,
        }
    )
    return {
        "schema_version": probe.RECEIPT_SCHEMA,
        "arm": "rayline_arc",
        "run_id": "run:r030:shared-replica-stop",
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
        "results": _results(0.25 if stopped else 0.2),
    }


def _affinity(*, stopped: bool) -> dict[str, object]:
    if not stopped:
        return {
            "schema_version": comparator.STATS_SCHEMA,
            "replicas": 2,
            "requests_by_replica": [25, 20],
            "session_pooling_requests_by_replica": [20, 16],
            "session_deletes_by_replica": [5, 4],
            "unique_sessions_by_replica": [5, 4],
            "affinity_mismatches": 0,
        }
    return {
        "schema_version": comparator.OUTAGE_STATS_SCHEMA,
        "replicas": 2,
        "requests_by_replica": [12, 32],
        "session_pooling_requests_by_replica": [12, 24],
        "session_deletes_by_replica": [0, 8],
        "unique_sessions_by_replica": [5, 8],
        "affinity_mismatches": 0,
        "primary_sessions_by_replica": [5, 4],
        "unavailable_replica": 0,
        "outage_confirmed": True,
        "unavailability_detections": 4,
        "unavailable_http_responses": 4,
        "unavailable_transport_errors": 0,
        "outage_failover_pooling_requests": 8,
        "outage_failover_sessions": 4,
        "unavailable_close_skips": 5,
        "survivor_close_requests": 8,
        "session_rebuild_responses": 4,
    }


def _telemetry(*, stopped: bool) -> dict[str, object]:
    return {
        "schema_version": comparator.TELEMETRY_SCHEMA,
        "session_actions": (
            {"created": 13, "appended": 23, "rebuilt": 0, "reused": 0}
            if stopped
            else {"created": 9, "appended": 27, "rebuilt": 0, "reused": 0}
        ),
        "tokens": {
            "appended": {"count": 36, "sum": 120 if stopped else 100},
            "retained": {"count": 36, "sum": 40 if stopped else 50},
            "full": {"count": 36, "sum": 200},
            "serialized": {"count": 36, "sum": 200},
        },
    }


def _inputs():
    control, stopped = contract.STOP_ARMS
    return (
        {control: _receipt(stopped=False), stopped: _receipt(stopped=True)},
        {control: _affinity(stopped=False), stopped: _affinity(stopped=True)},
        {control: _telemetry(stopped=False), stopped: _telemetry(stopped=True)},
    )


def test_replica_stop_comparison_passes_and_reports_post_boundary_cost() -> None:
    receipts, affinity, telemetry = _inputs()
    report = comparator.compare_replica_stop(receipts, affinity, telemetry)
    assert report["status"] == "passed"
    assert report["replica_stop_vs_control"]["completion_throughput_ratio"] == (
        pytest.approx(1.25)
    )
    assert report["replica_stop_vs_control"]["appended_token_work_ratio"] == (
        pytest.approx(1.2)
    )


def test_replica_stop_comparison_rejects_primary_placement_drift() -> None:
    receipts, affinity, telemetry = _inputs()
    broken = copy.deepcopy(affinity)
    broken[contract.STOP_ARMS[1]]["primary_sessions_by_replica"] = [4, 5]
    with pytest.raises(comparator.ReplicaStopComparisonError, match="affinity"):
        comparator.compare_replica_stop(receipts, broken, telemetry)


def test_replica_stop_comparison_rejects_missing_detection() -> None:
    receipts, affinity, telemetry = _inputs()
    broken = copy.deepcopy(affinity)
    broken[contract.STOP_ARMS[1]]["unavailability_detections"] = 3
    with pytest.raises(comparator.ReplicaStopComparisonError, match="affinity"):
        comparator.compare_replica_stop(receipts, broken, telemetry)
