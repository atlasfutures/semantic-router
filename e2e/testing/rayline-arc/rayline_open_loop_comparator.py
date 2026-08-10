#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compare the fixed Remote-versus-ARC PERF020 open-loop cells."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rayline_open_loop_contract import SaturationCriterion
from rayline_open_loop_packet import (
    MEASUREMENT_SCOPE,
    OFFERED_RATES,
    OpenLoopPacketError,
    rate_label,
    resolve_offered_rates,
)
from rayline_open_loop_probe import INPUT_SCHEMA, ProbeError, validate_receipt
from rayline_parity_comparator import IDENTITY_FIELDS

REPORT_SCHEMA = "rayline.vllm.open-loop-comparison.v2"
SATURATION_REPORT_SCHEMA = "rayline.vllm.open-loop-comparison.v3"
OPEN_LOOP_ARMS = ("rayline_remote", "rayline_arc")
DEFAULT_MEASURED_CASES = 32
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
        >= START_RATE_FLOOR_RATIO * results["realized_arrival_rate_rps"]
    )
    bounded = results["backlog_at_final_arrival"] < FINAL_BACKLOG_KNEE
    return {
        "offered_rate_maintained": maintained,
        "realized_arrival_rate_rps": results["realized_arrival_rate_rps"],
        "final_arrival_backlog_bounded": bounded,
        "queue_latency_dominates_service_p95": (
            results["start_lag_seconds"]["p95"]
            > results["service_latency_seconds"]["p95"]
        ),
        "overloaded": not maintained or not bounded,
    }


def _saturation(
    receipt: Mapping[str, Any], criterion: SaturationCriterion
) -> dict[str, Any]:
    """Decide whether one cell ran out of the only capacity this rig has.

    The deciding term is peak lane occupancy. Everything else here is recorded
    and deliberately does not vote.

    `completion_ratio` in particular must not decide. It is
    `completed / duration` over `(scheduled - 1) / span`, and `duration` is
    `span + drain` where `drain` is never smaller than the service time of the
    last request to finish. So the ratio falls as the offered rate rises even
    when nothing queues at all: shrinking the arrival span raises the weight of
    a fixed service tail. `unqueued_completion_ratio` is what that same cell
    would report with zero queueing, given its own measured tail, and it is
    recorded next to the observed ratio so the two can be read together
    instead of the observed one being mistaken for a deficit.
    """

    results = receipt["results"]
    peak = results["max_client_backlog"] / criterion.episode_lanes
    tail = float(results["service_latency_seconds"]["p95"])
    span = float(results["scheduled_span_seconds"])
    completed = int(results["completed"])
    finite_sample_gain = completed / (completed - 1) if completed > 1 else 1.0
    return {
        "peak_lane_occupancy": peak,
        "terminal_lane_occupancy": (
            results["backlog_at_final_arrival"] / criterion.episode_lanes
        ),
        "episode_lanes": criterion.episode_lanes,
        "completion_ratio": _ratio(
            results["completion_throughput_rps"],
            results["realized_arrival_rate_rps"],
        ),
        "unqueued_completion_ratio": finite_sample_gain * _ratio(span, span + tail),
        "drain_service_tail_multiple": _ratio(
            results["drain_seconds_after_final_arrival"], tail
        ),
        "saturated": peak >= criterion.occupancy_ratio,
    }


def _resolve_rungs(offered_rates: Sequence[float] | None) -> tuple[float, ...]:
    """Resolve the rung set this comparison is being held to.

    `None` means the frozen PERF020/PERF021 ladder, so callers that predate
    parameterised rungs keep byte-identical behaviour. The packet module owns
    what makes a rung set well-formed, so validation is deferred to it rather
    than duplicated here — a second copy is exactly how the ladder and the
    comparison drifted apart in the first place.
    """

    try:
        return resolve_offered_rates(offered_rates)
    except OpenLoopPacketError as error:
        raise OpenLoopComparisonError(str(error)) from error


def _validate_cells(
    raw_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
    offered_rates: Sequence[float] | None = None,
    case_count: int = DEFAULT_MEASURED_CASES,
) -> tuple[
    tuple[str, ...],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, bool]],
]:
    rungs = _resolve_rungs(offered_rates)
    expected_labels = tuple(rate_label(rate) for rate in rungs)
    if set(raw_cells) != set(expected_labels):
        raise OpenLoopComparisonError(
            "open-loop cells differ from the contracted rates"
        )
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    integrity: dict[str, dict[str, bool]] = {}
    for label, rate in zip(expected_labels, rungs, strict=True):
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
                or receipt["identity"]["case_count"] != case_count
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
                receipt["results"]["completed"] == case_count
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
    offered_rates: Sequence[float] | None = None,
    case_count: int = DEFAULT_MEASURED_CASES,
    saturation: SaturationCriterion | None = None,
) -> dict[str, Any]:
    """Compare one open-loop sweep's receipts against the rungs it contracted for.

    `offered_rates` and `case_count` are the run's own ladder and corpus size.
    They default to the frozen PERF020/PERF021 shape so those closed runs
    compare exactly as they did, but a caller that ran something else must say
    so: validating a four-rung run against a three-rung module default rejects
    a sweep that executed correctly, and a module case count writes
    `passed: False` over a run that completed every case it had.

    `saturation` is the run's saturation criterion. `None` keeps the frozen
    report shape byte-for-byte, which is what lets the closed runs' receipts
    still replay into the reports they recorded.
    """

    expected_labels, cells, integrity = _validate_cells(
        raw_cells, offered_rates, case_count
    )

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
    report: dict[str, Any] = {
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
    if saturation is None:
        return report
    saturation_cells = {
        arm: {
            label: _saturation(cells[label][arm], saturation)
            for label in expected_labels
        }
        for arm in OPEN_LOOP_ARMS
    }
    report["schema_version"] = SATURATION_REPORT_SCHEMA
    report["saturation"] = saturation_cells
    report["saturation_criterion"] = {
        "episode_lanes": saturation.episode_lanes,
        "occupancy_ratio": saturation.occupancy_ratio,
    }
    report["first_saturated_cell"] = {
        arm: next(
            (
                label
                for label in expected_labels
                if saturation_cells[arm][label]["saturated"]
            ),
            None,
        )
        for arm in OPEN_LOOP_ARMS
    }
    return report


def _parse_rate_list(raw: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _parse_args() -> tuple[argparse.Namespace, tuple[float, ...]]:
    """Build the per-cell receipt flags from the rung set being compared.

    The flags cannot be fixed at import time, because which cells exist is a
    property of the run, not of this module. `--offered-rates` is therefore
    read first and the receipt flags are derived from it.
    """

    rate_parser = argparse.ArgumentParser(add_help=False)
    rate_parser.add_argument(
        "--offered-rates",
        type=_parse_rate_list,
        default=None,
        help=(
            "Ascending comma-separated offered arrival rates in requests per "
            "second, matching the rungs the run contracted for. Defaults to "
            f"the frozen {','.join(str(rate) for rate in OFFERED_RATES)} ladder."
        ),
    )
    known, _ = rate_parser.parse_known_args()
    rungs = _resolve_rungs(known.offered_rates)
    parser = argparse.ArgumentParser(parents=[rate_parser])
    for rate in rungs:
        label = rate_label(rate)
        for arm in OPEN_LOOP_ARMS:
            parser.add_argument(
                f"--{label}-{arm.replace('_', '-')}", required=True, type=Path
            )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--episode-lanes",
        type=int,
        default=None,
        help=(
            "The packet's concurrent episode lane count. Supplying it enables "
            "the saturation criterion and the v3 report; omitting it keeps the "
            "frozen v2 report the closed runs recorded."
        ),
    )
    parser.add_argument("--occupancy-ratio", type=float, default=1.0)
    parser.add_argument("--case-count", type=int, default=DEFAULT_MEASURED_CASES)
    return parser.parse_args(), rungs


def main() -> None:
    try:
        args, rungs = _parse_args()
    except OpenLoopComparisonError as error:
        raise SystemExit(f"invalid open-loop sweep: {error}") from error
    raw_cells = {
        rate_label(rate): {
            arm: json.loads(getattr(args, f"{rate_label(rate)}_{arm}").read_text())
            for arm in OPEN_LOOP_ARMS
        }
        for rate in rungs
    }
    try:
        criterion = (
            None
            if args.episode_lanes is None
            else SaturationCriterion(
                episode_lanes=args.episode_lanes,
                occupancy_ratio=args.occupancy_ratio,
            )
        )
        report = compare_open_loop(raw_cells, rungs, args.case_count, criterion)
    except (
        OSError,
        json.JSONDecodeError,
        OpenLoopComparisonError,
        ValueError,
    ) as error:
        raise SystemExit(f"invalid open-loop sweep: {error}") from error
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
