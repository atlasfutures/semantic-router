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

comparator = importlib.import_module("rayline_open_loop_comparator")
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


def _receipt(label: str, rate: float, arm: str) -> dict[str, object]:
    latency = {"p50": 0.1, "p95": 0.2, "p99": 0.3}
    return {
        "schema_version": probe.INPUT_SCHEMA,
        "arm": arm,
        "run_id": f"run:{label}:{arm}",
        "identity": _identity(label),
        "results": {
            "scheduled": 32,
            "completed": 32,
            "failed": 0,
            "offered_rate_rps": rate,
            "realized_arrival_rate_rps": rate,
            "scheduled_span_seconds": 31 / rate,
            "duration_seconds": 101.0,
            "completion_throughput_rps": 32 / 101,
            "achieved_start_rate_rps": rate,
            "service_latency_seconds": latency,
            "scheduled_latency_seconds": latency,
            "start_lag_seconds": {"p50": 0.0, "p95": 0.0, "p99": 0.0},
            "max_client_backlog": 2,
            "backlog_at_final_arrival": 1,
            "drain_seconds_after_final_arrival": 1.0,
            "selected_worker_trace_sha256": _digest("trace"),
            "provider_calls": 0,
        },
    }


def _cells() -> dict[str, dict[str, dict[str, object]]]:
    return {
        packet.rate_label(rate): {
            arm: _receipt(packet.rate_label(rate), rate, arm)
            for arm in comparator.OPEN_LOOP_ARMS
        }
        for rate in packet.OFFERED_RATES
    }


def test_open_loop_comparison_passes_integrity_and_reports_knee() -> None:
    cells = _cells()
    cells["r045"]["rayline_arc"]["results"]["achieved_start_rate_rps"] = 0.3
    cells["r045"]["rayline_arc"]["results"]["backlog_at_final_arrival"] = 8
    cells["r045"]["rayline_arc"]["results"]["max_client_backlog"] = 9

    result = comparator.compare_open_loop(cells)

    assert result["status"] == "passed"
    assert result["cross_cell_trace_match"] is True
    assert result["first_overloaded_cell"]["rayline_remote"] is None
    assert result["first_overloaded_cell"]["rayline_arc"] == "r045"


def test_open_loop_comparison_fails_trace_integrity() -> None:
    cells = _cells()
    cells["r030"]["rayline_arc"]["results"]["selected_worker_trace_sha256"] = _digest(
        "different"
    )

    result = comparator.compare_open_loop(cells)

    assert result["status"] == "failed"
    assert result["cross_cell_trace_match"] is False


def test_open_loop_comparison_rejects_cross_arm_identity_mismatch() -> None:
    cells = _cells()
    broken = copy.deepcopy(cells)
    broken["r015"]["rayline_arc"]["identity"]["warm_state"] = "cold"

    with pytest.raises(comparator.OpenLoopComparisonError, match="identities differ"):
        comparator.compare_open_loop(broken)
