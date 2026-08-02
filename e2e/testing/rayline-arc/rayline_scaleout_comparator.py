#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Strict aggregate comparison for the PERF022 affinity scale-out packet."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rayline_open_loop_probe import validate_receipt
from rayline_parity_comparator import IDENTITY_FIELDS
from rayline_scaleout_contract import SCALEOUT_ARMS

REPORT_SCHEMA = "rayline.vllm.affinity-scaleout-comparison.v1"
AFFINITY_SCHEMA = "rayline.vllm.episode-affinity-stats.v1"
EXPECTED_REQUESTS = 36
EXPECTED_SESSIONS = 9
EXPECTED_MEASURED_CASES = 32


class ScaleoutComparisonError(ValueError):
    """A scale-out receipt is missing or violates the frozen schema."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ScaleoutComparisonError(f"{label} keys differ from the frozen schema")


def _validate_affinity(
    value: Mapping[str, Any], *, expected_replicas: int, label: str
) -> dict[str, Any]:
    expected = {
        "schema_version",
        "replicas",
        "requests_by_replica",
        "session_pooling_requests_by_replica",
        "session_deletes_by_replica",
        "unique_sessions_by_replica",
        "affinity_mismatches",
    }
    _exact_keys(value, expected, label)
    if value["schema_version"] != AFFINITY_SCHEMA:
        raise ScaleoutComparisonError(f"{label} schema differs")
    if value["replicas"] != expected_replicas:
        raise ScaleoutComparisonError(f"{label} replica count differs")
    vectors: dict[str, list[int]] = {}
    for field in (
        "requests_by_replica",
        "session_pooling_requests_by_replica",
        "session_deletes_by_replica",
        "unique_sessions_by_replica",
    ):
        raw = value[field]
        if (
            not isinstance(raw, list)
            or len(raw) != expected_replicas
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in raw
            )
        ):
            raise ScaleoutComparisonError(f"{label} {field} differs")
        vectors[field] = list(raw)
    if (
        sum(vectors["session_pooling_requests_by_replica"]) != EXPECTED_REQUESTS
        or sum(vectors["session_deletes_by_replica"]) != EXPECTED_SESSIONS
        or sum(vectors["unique_sessions_by_replica"]) != EXPECTED_SESSIONS
        or sum(vectors["requests_by_replica"]) != EXPECTED_REQUESTS + EXPECTED_SESSIONS
        or value["affinity_mismatches"] != 0
        or any(item == 0 for item in vectors["unique_sessions_by_replica"])
    ):
        raise ScaleoutComparisonError(f"{label} affinity accounting differs")
    return {**value, **vectors}


def _identity(receipt: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(receipt["identity"][field] for field in IDENTITY_FIELDS)


def compare_scaleout(
    raw_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
    raw_affinity: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if not raw_cells or set(raw_cells) != set(raw_affinity):
        raise ScaleoutComparisonError("scale-out cell sets differ")
    integrity: dict[str, Any] = {}
    affinity: dict[str, Any] = {}
    ratios: dict[str, Any] = {}
    trace: str | None = None
    passed = True
    for cell in sorted(raw_cells):
        arms = raw_cells[cell]
        if set(arms) != set(SCALEOUT_ARMS):
            raise ScaleoutComparisonError(f"{cell} arms differ")
        receipts = {arm: validate_receipt(dict(arms[arm])) for arm in SCALEOUT_ARMS}
        if any(receipt["arm"] != "rayline_arc" for receipt in receipts.values()):
            raise ScaleoutComparisonError(f"{cell} did not exercise ARC")
        if _identity(receipts[SCALEOUT_ARMS[0]]) != _identity(
            receipts[SCALEOUT_ARMS[1]]
        ):
            raise ScaleoutComparisonError(f"{cell} identities differ")
        cell_traces = {
            receipt["results"]["selected_worker_trace_sha256"]
            for receipt in receipts.values()
        }
        if trace is None:
            trace = next(iter(cell_traces))
        trace_match = len(cell_traces) == 1 and trace in cell_traces
        all_completed = all(
            receipt["results"]["completed"] == EXPECTED_MEASURED_CASES
            and receipt["results"]["failed"] == 0
            and receipt["results"]["provider_calls"] == 0
            for receipt in receipts.values()
        )
        integrity[cell] = {
            "all_completed": all_completed,
            "trace_match": trace_match,
            "provider_calls_zero": all(
                receipt["results"]["provider_calls"] == 0
                for receipt in receipts.values()
            ),
        }
        passed = passed and all(integrity[cell].values())
        affinity[cell] = {
            "arc_single": _validate_affinity(
                raw_affinity[cell]["arc_single"],
                expected_replicas=1,
                label=f"{cell}.arc_single",
            ),
            "arc_dual_affinity": _validate_affinity(
                raw_affinity[cell]["arc_dual_affinity"],
                expected_replicas=2,
                label=f"{cell}.arc_dual_affinity",
            ),
        }
        single = receipts["arc_single"]["results"]
        dual = receipts["arc_dual_affinity"]["results"]
        ratios[cell] = {
            "completion_throughput_ratio": (
                dual["completion_throughput_rps"] / single["completion_throughput_rps"]
            ),
            "service_latency_ratio": {
                percentile: dual["service_latency_seconds"][percentile]
                / single["service_latency_seconds"][percentile]
                for percentile in ("p50", "p95", "p99")
            },
            "scheduled_latency_ratio": {
                percentile: dual["scheduled_latency_seconds"][percentile]
                / single["scheduled_latency_seconds"][percentile]
                for percentile in ("p50", "p95", "p99")
            },
            "drain_ratio": dual["drain_seconds_after_final_arrival"]
            / single["drain_seconds_after_final_arrival"],
            "final_arrival_backlog_delta": (
                dual["backlog_at_final_arrival"] - single["backlog_at_final_arrival"]
            ),
        }
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "selected_worker_trace_sha256": trace,
        "integrity": integrity,
        "affinity": affinity,
        "dual_vs_single": ratios,
    }
