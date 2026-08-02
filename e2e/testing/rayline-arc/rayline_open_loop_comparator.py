#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare the fixed Remote-versus-ARC PERF020 open-loop cells."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rayline_open_loop_packet import MEASUREMENT_SCOPE, OFFERED_RATES, rate_label
from rayline_open_loop_probe import INPUT_SCHEMA, ProbeError, validate_receipt
from rayline_parity_comparator import IDENTITY_FIELDS

REPORT_SCHEMA = "rayline.vllm.open-loop-comparison.v1"
OPEN_LOOP_ARMS = ("rayline_remote", "rayline_arc")
MEASURED_CASES = 32
FINAL_BACKLOG_KNEE = 8
START_RATE_FLOOR_RATIO = 0.90
CELL_IDENTITY_FIELDS = tuple(
    field for field in IDENTITY_FIELDS if field != "workload_sha256"
)


class OpenLoopComparisonError(ValueError):
    """The supplied receipts cannot form the registered open-loop sweep."""


def _ratio(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 1.0 if candidate == 0 else math.inf
    return candidate / baseline


def _latency_ratios(
    candidate: Mapping[str, float], baseline: Mapping[str, float]
) -> dict[str, float]:
    return {
        field: _ratio(float(candidate[field]), float(baseline[field]))
        for field in ("p50", "p95", "p99")
    }


def _pairwise(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    candidate_results = candidate["results"]
    baseline_results = baseline["results"]
    return {
        "completion_throughput_ratio": _ratio(
            candidate_results["completion_throughput_rps"],
            baseline_results["completion_throughput_rps"],
        ),
        "achieved_start_rate_ratio": _ratio(
            candidate_results["achieved_start_rate_rps"],
            baseline_results["achieved_start_rate_rps"],
        ),
        "service_latency_ratio": _latency_ratios(
            candidate_results["service_latency_seconds"],
            baseline_results["service_latency_seconds"],
        ),
        "scheduled_latency_ratio": _latency_ratios(
            candidate_results["scheduled_latency_seconds"],
            baseline_results["scheduled_latency_seconds"],
        ),
        "start_lag_ratio": _latency_ratios(
            candidate_results["start_lag_seconds"],
            baseline_results["start_lag_seconds"],
        ),
        "max_client_backlog_delta": (
            candidate_results["max_client_backlog"]
            - baseline_results["max_client_backlog"]
        ),
        "final_arrival_backlog_delta": (
            candidate_results["backlog_at_final_arrival"]
            - baseline_results["backlog_at_final_arrival"]
        ),
    }


def _diagnostic(receipt: Mapping[str, Any]) -> dict[str, Any]:
    results = receipt["results"]
    maintained = (
        results["achieved_start_rate_rps"]
        >= START_RATE_FLOOR_RATIO * results["offered_rate_rps"]
    )
    bounded = results["backlog_at_final_arrival"] < FINAL_BACKLOG_KNEE
    return {
        "offered_rate_maintained": maintained,
        "final_arrival_backlog_bounded": bounded,
        "queue_latency_dominates_service_p95": (
            results["start_lag_seconds"]["p95"]
            > results["service_latency_seconds"]["p95"]
        ),
        "overloaded": not maintained or not bounded,
    }


def _validate_cells(
    raw_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[
    tuple[str, ...],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, bool]],
]:
    expected_labels = tuple(rate_label(rate) for rate in OFFERED_RATES)
    if set(raw_cells) != set(expected_labels):
        raise OpenLoopComparisonError("open-loop cells differ from frozen rates")
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    integrity: dict[str, dict[str, bool]] = {}
    for label, rate in zip(expected_labels, OFFERED_RATES, strict=True):
        raw_receipts = raw_cells[label]
        if set(raw_receipts) != set(OPEN_LOOP_ARMS):
            raise OpenLoopComparisonError(f"{label} arms differ")
        cells[label] = {}
        for arm in OPEN_LOOP_ARMS:
            try:
                receipt = validate_receipt(raw_receipts[arm])
            except ProbeError as error:
                raise OpenLoopComparisonError(str(error)) from error
            if (
                receipt["schema_version"] != INPUT_SCHEMA
                or receipt["arm"] != arm
                or receipt["identity"]["measurement_scope"] != MEASUREMENT_SCOPE
                or receipt["identity"]["case_count"] != MEASURED_CASES
                or not math.isclose(
                    receipt["results"]["offered_rate_rps"], rate, rel_tol=1e-12
                )
            ):
                raise OpenLoopComparisonError(f"{label} receipt differs")
            cells[label][arm] = receipt
        if (
            cells[label]["rayline_remote"]["identity"]
            != cells[label]["rayline_arc"]["identity"]
        ):
            raise OpenLoopComparisonError(f"{label} identities differ")
        integrity[label] = {
            "all_completed": all(
                receipt["results"]["completed"] == MEASURED_CASES
                and receipt["results"]["failed"] == 0
                for receipt in cells[label].values()
            ),
            "provider_calls_zero": all(
                receipt["results"]["provider_calls"] == 0
                for receipt in cells[label].values()
            ),
            "trace_match": len(
                {
                    receipt["results"]["selected_worker_trace_sha256"]
                    for receipt in cells[label].values()
                }
            )
            == 1,
        }
    baseline_identity = cells[expected_labels[0]][OPEN_LOOP_ARMS[0]]["identity"]
    for label in expected_labels[1:]:
        candidate = cells[label][OPEN_LOOP_ARMS[0]]["identity"]
        mismatched = [
            field
            for field in CELL_IDENTITY_FIELDS
            if candidate[field] != baseline_identity[field]
        ]
        if mismatched:
            raise OpenLoopComparisonError(
                f"{label} fixed identities differ: {mismatched}"
            )
        if candidate["workload_sha256"] == baseline_identity["workload_sha256"]:
            raise OpenLoopComparisonError("rate workloads must have distinct digests")
    return expected_labels, cells, integrity


def _trace_summary(
    *,
    labels: tuple[str, ...],
    cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, str]], set[str], bool]:
    trace_digests = {
        label: {
            arm: cells[label][arm]["results"]["selected_worker_trace_sha256"]
            for arm in OPEN_LOOP_ARMS
        }
        for label in labels
    }
    distinct_traces = {
        digest for cell in trace_digests.values() for digest in cell.values()
    }
    return trace_digests, distinct_traces, len(distinct_traces) == 1


def compare_open_loop(
    raw_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    expected_labels, cells, integrity = _validate_cells(raw_cells)

    trace_digests, distinct_traces, cross_cell_trace_match = _trace_summary(
        labels=expected_labels, cells=cells
    )
    diagnostics = {
        arm: {label: _diagnostic(cells[label][arm]) for label in expected_labels}
        for arm in OPEN_LOOP_ARMS
    }
    first_overloaded = {
        arm: next(
            (
                label
                for label in expected_labels
                if diagnostics[arm][label]["overloaded"]
            ),
            None,
        )
        for arm in OPEN_LOOP_ARMS
    }
    passed = (
        all(all(cell.values()) for cell in integrity.values())
        and cross_cell_trace_match
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "integrity": integrity,
        "cross_cell_trace_match": cross_cell_trace_match,
        "selected_worker_trace_sha256": (
            next(iter(distinct_traces)) if cross_cell_trace_match else None
        ),
        "selected_worker_trace_sha256_by_cell": trace_digests,
        "run_ids": {
            label: {arm: cells[label][arm]["run_id"] for arm in OPEN_LOOP_ARMS}
            for label in expected_labels
        },
        "arc_vs_remote": {
            label: _pairwise(
                cells[label]["rayline_arc"], cells[label]["rayline_remote"]
            )
            for label in expected_labels
        },
        "load_diagnostics": diagnostics,
        "first_overloaded_cell": first_overloaded,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for rate in OFFERED_RATES:
        label = rate_label(rate)
        for arm in OPEN_LOOP_ARMS:
            parser.add_argument(
                f"--{label}-{arm.replace('_', '-')}", required=True, type=Path
            )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_cells = {
        rate_label(rate): {
            arm: json.loads(getattr(args, f"{rate_label(rate)}_{arm}").read_text())
            for arm in OPEN_LOOP_ARMS
        }
        for rate in OFFERED_RATES
    }
    try:
        report = compare_open_loop(raw_cells)
    except (OSError, json.JSONDecodeError, OpenLoopComparisonError) as error:
        raise SystemExit(f"invalid open-loop sweep: {error}") from error
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
