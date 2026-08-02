#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare the fixed Remote-versus-ARC PERF017 concurrency cells."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rayline_parity_comparator import (
    IDENTITY_FIELDS,
    INPUT_SCHEMA,
    INPUT_TOKEN_BUCKETS,
    ReceiptError,
    validate_receipt,
)

REPORT_SCHEMA = "rayline.vllm.concurrency-sweep-comparison.v1"
MEASUREMENT_SCOPE = "architecture_decision_concurrency_sweep"
CONCURRENCY_CELLS = (1, 4, 8)
SWEEP_ARMS = ("rayline_remote", "rayline_arc")
MEASURED_CASES = 32
CELL_IDENTITY_FIELDS = tuple(
    field for field in IDENTITY_FIELDS if field != "workload_sha256"
)


class SweepComparisonError(ValueError):
    """The supplied receipts cannot form the registered sweep."""


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
    buckets: dict[str, Any] = {}
    for name in INPUT_TOKEN_BUCKETS:
        candidate_bucket = candidate_results["latency_by_input_tokens"][name]
        baseline_bucket = baseline_results["latency_by_input_tokens"][name]
        if candidate_bucket["scheduled"] != baseline_bucket["scheduled"]:
            raise SweepComparisonError(f"input bucket {name} schedules differ")
        if (
            candidate_bucket["selection_latency_seconds"] is None
            or baseline_bucket["selection_latency_seconds"] is None
        ):
            raise SweepComparisonError(f"input bucket {name} must be non-empty")
        buckets[name] = {
            "cases": candidate_bucket["scheduled"],
            "latency_ratio": _latency_ratios(
                candidate_bucket["selection_latency_seconds"],
                baseline_bucket["selection_latency_seconds"],
            ),
        }
    return {
        "throughput_ratio": _ratio(
            candidate_results["throughput_rps"],
            baseline_results["throughput_rps"],
        ),
        "latency_ratio": _latency_ratios(
            candidate_results["selection_latency_seconds"],
            baseline_results["selection_latency_seconds"],
        ),
        "latency_by_input_tokens": buckets,
    }


def _validate_cell(
    concurrency: int, raw_receipts: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if set(raw_receipts) != set(SWEEP_ARMS):
        raise SweepComparisonError(f"concurrency {concurrency} arms differ")
    receipts: dict[str, dict[str, Any]] = {}
    for arm in SWEEP_ARMS:
        try:
            receipt = validate_receipt(raw_receipts[arm])
        except ReceiptError as error:
            raise SweepComparisonError(str(error)) from error
        if receipt["schema_version"] != INPUT_SCHEMA or receipt["arm"] != arm:
            raise SweepComparisonError(f"concurrency {concurrency} receipt differs")
        if (
            receipt["identity"]["measurement_scope"] != MEASUREMENT_SCOPE
            or receipt["identity"]["case_count"] != MEASURED_CASES
        ):
            raise SweepComparisonError(f"concurrency {concurrency} identity differs")
        receipts[arm] = receipt
    remote_identity = receipts["rayline_remote"]["identity"]
    arc_identity = receipts["rayline_arc"]["identity"]
    if remote_identity != arc_identity:
        raise SweepComparisonError(f"concurrency {concurrency} identities differ")

    integrity = {
        "all_completed": all(
            receipt["results"]["completed"] == MEASURED_CASES
            and receipt["results"]["failed"] == 0
            for receipt in receipts.values()
        ),
        "provider_calls_zero": all(
            receipt["results"]["provider_calls"] == 0 for receipt in receipts.values()
        ),
        "trace_match": len(
            {
                receipt["results"]["selected_worker_trace_sha256"]
                for receipt in receipts.values()
            }
        )
        == 1,
    }
    return receipts, integrity


def compare_sweep(
    raw_cells: Mapping[int, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(raw_cells) != set(CONCURRENCY_CELLS):
        raise SweepComparisonError("concurrency cells must be exactly 1, 4, and 8")
    cells: dict[int, dict[str, dict[str, Any]]] = {}
    integrity: dict[str, Any] = {}
    for concurrency in CONCURRENCY_CELLS:
        cells[concurrency], integrity[f"c{concurrency}"] = _validate_cell(
            concurrency, raw_cells[concurrency]
        )

    baseline_identity = cells[CONCURRENCY_CELLS[0]][SWEEP_ARMS[0]]["identity"]
    for concurrency in CONCURRENCY_CELLS[1:]:
        candidate = cells[concurrency][SWEEP_ARMS[0]]["identity"]
        mismatched = [
            field
            for field in CELL_IDENTITY_FIELDS
            if candidate[field] != baseline_identity[field]
        ]
        if mismatched:
            raise SweepComparisonError(
                f"concurrency {concurrency} fixed identities differ: {mismatched}"
            )
        if candidate["workload_sha256"] == baseline_identity["workload_sha256"]:
            raise SweepComparisonError(
                "concurrency workloads must have distinct digests"
            )

    trace_digests = {
        f"c{concurrency}": {
            arm: cells[concurrency][arm]["results"]["selected_worker_trace_sha256"]
            for arm in SWEEP_ARMS
        }
        for concurrency in CONCURRENCY_CELLS
    }
    distinct_trace_digests = {
        digest for cell in trace_digests.values() for digest in cell.values()
    }
    cross_cell_trace_match = len(distinct_trace_digests) == 1
    pairwise = {
        f"c{concurrency}": _pairwise(
            cells[concurrency]["rayline_arc"],
            cells[concurrency]["rayline_remote"],
        )
        for concurrency in CONCURRENCY_CELLS
    }
    scaling: dict[str, Any] = {}
    for arm in SWEEP_ARMS:
        baseline = cells[1][arm]
        scaling[arm] = {
            f"c{concurrency}_vs_c1": _pairwise(cells[concurrency][arm], baseline)
            for concurrency in CONCURRENCY_CELLS[1:]
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
            next(iter(distinct_trace_digests)) if cross_cell_trace_match else None
        ),
        "selected_worker_trace_sha256_by_cell": trace_digests,
        "run_ids": {
            f"c{concurrency}": {
                arm: cells[concurrency][arm]["run_id"] for arm in SWEEP_ARMS
            }
            for concurrency in CONCURRENCY_CELLS
        },
        "arc_vs_remote": pairwise,
        "scaling": scaling,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for concurrency in CONCURRENCY_CELLS:
        for arm in SWEEP_ARMS:
            parser.add_argument(
                f"--c{concurrency}-{arm.replace('_', '-')}",
                required=True,
                type=Path,
            )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_cells = {
        concurrency: {
            arm: json.loads(getattr(args, f"c{concurrency}_{arm}").read_text())
            for arm in SWEEP_ARMS
        }
        for concurrency in CONCURRENCY_CELLS
    }
    try:
        report = compare_sweep(raw_cells)
    except (OSError, json.JSONDecodeError, SweepComparisonError) as error:
        raise SystemExit(f"invalid concurrency sweep: {error}") from error
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
