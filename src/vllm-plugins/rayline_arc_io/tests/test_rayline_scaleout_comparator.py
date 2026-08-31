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

comparator = importlib.import_module("rayline_scaleout_comparator")
packet = importlib.import_module("rayline_open_loop_packet")
parity = importlib.import_module("rayline_parity_comparator")
probe = importlib.import_module("rayline_open_loop_probe")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity(label: str) -> dict[str, object]:
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
            identity[field] = _digest(label if field == "workload_sha256" else field)
        else:
            identity[field] = field
    return identity


def _receipt(label: str, rate: float, suffix: str) -> dict[str, object]:
    latency = {"p50": 1.0, "p95": 2.0, "p99": 3.0}
    return {
        "schema_version": probe.INPUT_SCHEMA,
        "arm": "rayline_arc",
        "run_id": f"run:{label}:{suffix}",
        "identity": _identity(label),
        "results": {
            "scheduled": 32,
            "completed": 32,
            "failed": 0,
            "offered_rate_rps": rate,
            "realized_arrival_rate_rps": rate,
            "scheduled_span_seconds": 31 / rate,
            "duration_seconds": 100.0,
            "completion_throughput_rps": 0.2 if suffix == "single" else 0.3,
            "achieved_start_rate_rps": rate,
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


def _affinity(replicas: int) -> dict[str, object]:
    if replicas == 1:
        pooling, deletes, unique = [36], [9], [9]
    else:
        pooling, deletes, unique = [16, 20], [4, 5], [4, 5]
    return {
        "schema_version": comparator.AFFINITY_SCHEMA,
        "replicas": replicas,
        "requests_by_replica": [
            pooling[index] + deletes[index] for index in range(replicas)
        ],
        "session_pooling_requests_by_replica": pooling,
        "session_deletes_by_replica": deletes,
        "unique_sessions_by_replica": unique,
        "affinity_mismatches": 0,
    }


def _cells() -> tuple[dict[str, object], dict[str, object]]:
    receipts: dict[str, object] = {}
    affinity: dict[str, object] = {}
    for label, rate in (("r030", 0.3), ("r045", 0.45)):
        receipts[label] = {
            "arc_single": _receipt(label, rate, "single"),
            "arc_dual_affinity": _receipt(label, rate, "dual"),
        }
        affinity[label] = {
            "arc_single": _affinity(1),
            "arc_dual_affinity": _affinity(2),
        }
    return receipts, affinity


def test_scaleout_comparison_passes_and_reports_gain() -> None:
    receipts, affinity = _cells()

    report = comparator.compare_scaleout(receipts, affinity)

    assert report["status"] == "passed"
    assert report["dual_vs_single"]["r030"][
        "completion_throughput_ratio"
    ] == pytest.approx(1.5)
    assert report["affinity"]["r045"]["arc_dual_affinity"][
        "unique_sessions_by_replica"
    ] == [4, 5]


def test_scaleout_comparison_rejects_affinity_loss() -> None:
    receipts, affinity = _cells()
    broken = copy.deepcopy(affinity)
    broken["r030"]["arc_dual_affinity"]["affinity_mismatches"] = 1

    with pytest.raises(comparator.ScaleoutComparisonError, match="affinity"):
        comparator.compare_scaleout(receipts, broken)


def test_scaleout_comparison_fails_trace_integrity() -> None:
    receipts, affinity = _cells()
    receipts["r045"]["arc_dual_affinity"]["results"]["selected_worker_trace_sha256"] = (
        _digest("different")
    )

    report = comparator.compare_scaleout(receipts, affinity)

    assert report["status"] == "failed"
