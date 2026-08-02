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

comparator = importlib.import_module("rayline_failover_comparator")
contract = importlib.import_module("rayline_failover_contract")
packet = importlib.import_module("rayline_open_loop_packet")
parity = importlib.import_module("rayline_parity_comparator")
probe = importlib.import_module("rayline_open_loop_probe")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity() -> dict[str, object]:
    identity: dict[str, object] = {}
    digest_fields = {
        "corpus_sha256",
        "workload_sha256",
        "tokenizer_sha256",
        "worker_topology_sha256",
    }
    for field in parity.IDENTITY_FIELDS:
        if field == "measurement_scope":
            identity[field] = packet.MEASUREMENT_SCOPE
        elif field == "case_count":
            identity[field] = 32
        elif field == "seed":
            identity[field] = 20260730
        elif field in digest_fields:
            identity[field] = _digest(field)
        else:
            identity[field] = field
    return identity


def _receipt(arm: str, throughput: float) -> dict[str, object]:
    latency = {"p50": 1.0, "p95": 2.0, "p99": 3.0}
    return {
        "schema_version": probe.INPUT_SCHEMA,
        "arm": "rayline_arc",
        "run_id": f"run:{arm}",
        "identity": _identity(),
        "results": {
            "scheduled": 32,
            "completed": 32,
            "failed": 0,
            "offered_rate_rps": 0.3,
            "realized_arrival_rate_rps": 0.3,
            "scheduled_span_seconds": 31 / 0.3,
            "duration_seconds": 100.0,
            "completion_throughput_rps": throughput,
            "achieved_start_rate_rps": 0.3,
            "service_latency_seconds": latency,
            "scheduled_latency_seconds": latency,
            "start_lag_seconds": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
            "max_client_backlog": 8,
            "backlog_at_final_arrival": 8,
            "drain_seconds_after_final_arrival": 10.0,
            "selected_worker_trace_sha256": _digest("trace"),
            "provider_calls": 0,
        },
    }


def _affinity(*, failover: bool) -> dict[str, object]:
    if failover:
        return {
            "schema_version": comparator.FAILOVER_AFFINITY_SCHEMA,
            "replicas": 2,
            "requests_by_replica": [27, 27],
            "session_pooling_requests_by_replica": [18, 18],
            "session_deletes_by_replica": [9, 9],
            "unique_sessions_by_replica": [9, 9],
            "affinity_mismatches": 0,
            "failover_after_pooling": 2,
            "failover_pooling_requests": 18,
            "failover_sessions": 9,
            "close_fanout_requests": 18,
            "session_rebuild_responses": 9,
        }
    return {
        "schema_version": comparator.STICKY_AFFINITY_SCHEMA,
        "replicas": 2,
        "requests_by_replica": [20, 25],
        "session_pooling_requests_by_replica": [16, 20],
        "session_deletes_by_replica": [4, 5],
        "unique_sessions_by_replica": [4, 5],
        "affinity_mismatches": 0,
    }


def _telemetry(*, failover: bool) -> dict[str, object]:
    actions = (
        {"created": 18, "appended": 18, "rebuilt": 0, "reused": 0}
        if failover
        else {"created": 9, "appended": 27, "rebuilt": 0, "reused": 0}
    )
    appended = 150 if failover else 100
    retained = 25 if failover else 50
    return {
        "schema_version": comparator.TELEMETRY_SCHEMA,
        "session_actions": actions,
        "tokens": {
            "appended": {"count": 36, "sum": appended},
            "retained": {"count": 36, "sum": retained},
            "full": {"count": 36, "sum": 200},
            "serialized": {"count": 36, "sum": 200},
        },
    }


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    sticky, failed = contract.FAILOVER_ARMS
    return (
        {sticky: _receipt(sticky, 0.3), failed: _receipt(failed, 0.25)},
        {sticky: _affinity(failover=False), failed: _affinity(failover=True)},
        {sticky: _telemetry(failover=False), failed: _telemetry(failover=True)},
    )


def test_failover_comparison_passes_and_reports_rebuild_cost() -> None:
    receipts, affinity, telemetry = _inputs()

    report = comparator.compare_failover(receipts, affinity, telemetry)

    assert report["status"] == "passed"
    ratios = report["forced_failover_vs_sticky"]
    assert ratios["completion_throughput_ratio"] == pytest.approx(5 / 6)
    assert ratios["appended_token_work_ratio"] == pytest.approx(1.5)
    assert ratios["retained_token_ratio"] == pytest.approx(0.5)


def test_failover_comparison_rejects_missing_rebuild() -> None:
    receipts, affinity, telemetry = _inputs()
    broken = copy.deepcopy(affinity)
    broken[contract.FAILOVER_ARMS[1]]["session_rebuild_responses"] = 8

    with pytest.raises(comparator.FailoverComparisonError, match="failover"):
        comparator.compare_failover(receipts, broken, telemetry)


def test_failover_comparison_fails_trace_drift() -> None:
    receipts, affinity, telemetry = _inputs()
    receipts[contract.FAILOVER_ARMS[1]]["results"]["selected_worker_trace_sha256"] = (
        _digest("different")
    )

    report = comparator.compare_failover(receipts, affinity, telemetry)

    assert report["status"] == "failed"
