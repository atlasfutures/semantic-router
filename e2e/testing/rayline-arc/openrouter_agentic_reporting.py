#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-only reporting and metric gates for the agentic benchmark."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from modal_fullstack_canary import _failure_total, _metric_values, _summary
from openrouter_agentic_workload import (
    MODAL_REFERENCE,
    PROVIDER_NAMES,
    SCENARIOS,
    WORKERS,
)
from openrouter_fullstack_canary import (
    PROVIDER_ATTEMPT_METRIC,
    PROVIDER_EXHAUSTION_METRIC,
    PROVIDER_LOGICAL_METRIC,
    PROVIDER_RETRY_METRIC,
    SESSION_METRIC,
    _metric_total,
)


def flatten(phases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for phase in phases for result in phase["results"]]


def result_report(results: list[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    completion_tokens = sum(int(result["completion_tokens"]) for result in results)
    attempts = sum(int(result["external_attempts"]) for result in results)
    upstream = [
        float(result["envoy_upstream_service_seconds"])
        for result in results
        if result["envoy_upstream_service_seconds"] is not None
    ]
    return {
        "requests": len(results),
        "wall_seconds": wall_seconds,
        "requests_per_second": len(results) / wall_seconds,
        "output_tokens_per_second": completion_tokens / wall_seconds,
        "time_to_first_token": _summary(
            [float(result["time_to_first_token_seconds"]) for result in results]
        ),
        "end_to_end_latency": _summary(
            [float(result["total_seconds"]) for result in results]
        ),
        "envoy_upstream_service_time": _summary(upstream) if upstream else None,
        "prompt_tokens": sum(int(result["prompt_tokens"]) for result in results),
        "completion_tokens": completion_tokens,
        "external_attempts": attempts,
        "retries": attempts - len(results),
        "cost_usd": sum(float(result["cost_usd"]) for result in results),
        "providers": sorted({str(result["provider"]) for result in results}),
        "models": sorted({str(result["response_model"]) for result in results}),
    }


def _per_model_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "requests": 0,
            "time_to_first_token": None,
            "end_to_end_latency": None,
            "envoy_upstream_service_time": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "external_attempts": 0,
            "retries": 0,
            "cost_usd": 0.0,
            "providers": [],
            "models": [],
        }
    report = result_report(
        results,
        sum(float(result["total_seconds"]) for result in results),
    )
    for key in ("wall_seconds", "requests_per_second", "output_tokens_per_second"):
        report.pop(key)
    return report


def path_reports(
    phases: list[dict[str, Any]],
    *,
    paths: tuple[str, ...],
    concurrency_levels: tuple[int, ...],
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for path in paths:
        path_phases = [phase for phase in phases if phase["path"] == path]
        by_concurrency: dict[str, Any] = {}
        for concurrency in concurrency_levels:
            matching = [
                phase for phase in path_phases if phase["concurrency"] == concurrency
            ]
            results = flatten(matching)
            by_concurrency[str(concurrency)] = result_report(
                results,
                sum(float(phase["wall_seconds"]) for phase in matching),
            )
        all_results = flatten(path_phases)
        per_model = {
            worker: _per_model_report(
                [
                    result
                    for result in all_results
                    if result["selected_worker"] == worker
                ]
            )
            for worker in WORKERS
        }
        reports[path] = {
            "by_concurrency": by_concurrency,
            "per_model": per_model,
        }
    return reports


def comparison(
    reports: dict[str, Any], concurrency_levels: tuple[int, ...]
) -> dict[str, Any]:
    result: dict[str, Any] = {"pure_modal_reference": MODAL_REFERENCE}
    for concurrency in concurrency_levels:
        key = str(concurrency)
        static = reports["gateway_static"]["by_concurrency"][key]
        arc = reports["arc"]["by_concurrency"][key]
        result[f"concurrency_{concurrency}"] = {
            "arc_to_static_throughput_ratio": (
                arc["requests_per_second"] / static["requests_per_second"]
            ),
            "arc_to_static_output_throughput_ratio": (
                arc["output_tokens_per_second"] / static["output_tokens_per_second"]
            ),
            "arc_minus_static_ttft_p50_seconds": (
                arc["time_to_first_token"]["p50_seconds"]
                - static["time_to_first_token"]["p50_seconds"]
            ),
            "arc_minus_static_ttft_p95_seconds": (
                arc["time_to_first_token"]["p95_seconds"]
                - static["time_to_first_token"]["p95_seconds"]
            ),
            "arc_minus_static_e2e_p95_seconds": (
                arc["end_to_end_latency"]["p95_seconds"]
                - static["end_to_end_latency"]["p95_seconds"]
            ),
        }
    return result


def _readiness_report(
    key_readiness: dict[str, Any], key_readiness_requests: int
) -> dict[str, Any]:
    return {
        "requests": key_readiness_requests,
        "model": key_readiness["response_model"],
        "provider": key_readiness["provider"],
        "completion_tokens": key_readiness["completion_tokens"],
        "external_attempts": key_readiness["external_attempts"],
        "retries": key_readiness["external_attempts"] - 1,
        "cost_usd": key_readiness["cost_usd"],
    }


def _workload_report(
    *,
    selected_cases: list[dict[str, Any]],
    measured_requests: int,
    selected_case_count: int,
    minimum_active_workers: int,
    minimum_selected_cases_per_active_worker: int,
    max_completion_tokens: int,
    concurrency_levels: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "scenarios": sorted(SCENARIOS),
        "selected_case_counts_by_scenario": {
            scenario: sum(case["scenario"] == scenario for case in selected_cases)
            for scenario in SCENARIOS
        },
        "selected_case_counts_by_worker": {
            worker: sum(case["expected_worker"] == worker for case in selected_cases)
            for worker in WORKERS
        },
        "selected_case_count": selected_case_count,
        "minimum_active_workers": minimum_active_workers,
        "minimum_selected_cases_per_active_worker": (
            minimum_selected_cases_per_active_worker
        ),
        "max_completion_tokens": max_completion_tokens,
        "concurrency_levels": concurrency_levels,
        "measured_requests": measured_requests,
    }


def _coverage_report(coverage: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "requests": len(coverage),
        "selection_counts": {
            worker: sum(result["selected_worker"] == worker for result in coverage)
            for worker in WORKERS
        },
        "cost_usd": sum(float(result["cost_usd"]) for result in coverage),
    }


def _endpoint_report(endpoint_probes: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = sum(int(result["external_attempts"]) for result in endpoint_probes)
    return {
        "requests": len(endpoint_probes),
        "all_reachable": len(endpoint_probes) == len(WORKERS),
        "workers": {
            worker: {
                "reachable": any(
                    result["selected_worker"] == worker for result in endpoint_probes
                ),
                "model": model,
                "provider": next(
                    result["provider"]
                    for result in endpoint_probes
                    if result["selected_worker"] == worker
                ),
                "provider_order": PROVIDER_NAMES[worker],
            }
            for worker, model in WORKERS.items()
        },
        "external_attempts": attempts,
        "retries": attempts - len(endpoint_probes),
        "cost_usd": sum(float(result["cost_usd"]) for result in endpoint_probes),
    }


def build_report(
    *,
    run_id: str,
    transport_preflight: dict[str, Any],
    encoder_warmup: dict[str, Any],
    key_readiness: dict[str, Any],
    endpoint_probes: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    router_metrics: dict[str, int],
    all_results: list[dict[str, Any]],
    external_attempts: int,
    provider_cost: float,
    paths: tuple[str, ...],
    concurrency_levels: tuple[int, ...],
    key_readiness_requests: int,
    selected_case_count: int,
    minimum_active_workers: int,
    minimum_selected_cases_per_active_worker: int,
    max_completion_tokens: int,
    maximum_provider_requests: int,
    maximum_external_attempts: int,
    maximum_reported_provider_cost_usd: float,
) -> dict[str, Any]:
    measured_results = flatten(phases)
    reports = path_reports(
        phases,
        paths=paths,
        concurrency_levels=concurrency_levels,
    )
    return {
        "schema_version": "rayline.arc.openrouter-agentic-benchmark.v4",
        "run_id": run_id,
        "status": "passed",
        "models": WORKERS,
        "provider_orders": PROVIDER_NAMES,
        "provider_fallbacks": True,
        "reasoning_enabled": False,
        "encoder_warmup": encoder_warmup,
        "pre_encoder_transport_preflight": transport_preflight,
        "openrouter_key_readiness": _readiness_report(
            key_readiness, key_readiness_requests
        ),
        "workload": _workload_report(
            selected_cases=selected_cases,
            measured_requests=len(measured_results),
            selected_case_count=selected_case_count,
            minimum_active_workers=minimum_active_workers,
            minimum_selected_cases_per_active_worker=(
                minimum_selected_cases_per_active_worker
            ),
            max_completion_tokens=max_completion_tokens,
            concurrency_levels=concurrency_levels,
        ),
        "coverage": _coverage_report(coverage),
        "endpoint_reachability": _endpoint_report(endpoint_probes),
        "paths": reports,
        "comparison": comparison(reports, concurrency_levels),
        "router_metrics": router_metrics,
        "actual_provider_requests": (
            len(all_results) + int(transport_preflight["provider_requests"])
        ),
        "maximum_provider_requests": maximum_provider_requests,
        "actual_external_attempts": external_attempts,
        "maximum_external_attempts": maximum_external_attempts,
        "reported_provider_cost_usd": provider_cost,
        "maximum_reported_provider_cost_usd": maximum_reported_provider_cost_usd,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
        "limitations": [
            "small diagnostic sample, not a production SLO qualification",
            "synthetic public tool outputs and routing anchors",
            "pure-Modal reference used different generation models and prompt lengths",
        ],
    }


def _metric_delta(after: float, before: float) -> int:
    value = after - before
    if value < 0 or not value.is_integer():
        raise RuntimeError("router metric reset or became non-integral")
    return int(value)


def validate_router_metrics(
    *,
    before: str,
    after: str,
    coverage: list[dict[str, Any]],
    phases: list[dict[str, Any]],
) -> dict[str, int]:
    arc_results = [
        result
        for phase in phases
        if phase["path"] == "arc"
        for result in phase["results"]
    ]
    logical_expected = len(coverage) + len(arc_results)
    logical = _metric_delta(
        _metric_total(after, PROVIDER_LOGICAL_METRIC),
        _metric_total(before, PROVIDER_LOGICAL_METRIC),
    )
    attempts = _metric_delta(
        _metric_total(after, PROVIDER_ATTEMPT_METRIC),
        _metric_total(before, PROVIDER_ATTEMPT_METRIC),
    )
    retries = _metric_delta(
        _metric_total(after, PROVIDER_RETRY_METRIC),
        _metric_total(before, PROVIDER_RETRY_METRIC),
    )
    exhaustions = _metric_delta(
        _metric_total(after, PROVIDER_EXHAUSTION_METRIC),
        _metric_total(before, PROVIDER_EXHAUSTION_METRIC),
    )
    failures = _failure_total(after) - _failure_total(before)
    sessions_before = _metric_values(before, SESSION_METRIC)
    sessions_after = _metric_values(after, SESSION_METRIC)
    sessions = int(sessions_after.get("created", 0) - sessions_before.get("created", 0))
    if logical != logical_expected:
        raise RuntimeError("router logical-request metric diverged")
    if attempts != sum(
        int(result["external_attempts"]) for result in [*coverage, *arc_results]
    ):
        raise RuntimeError("router attempt metric diverged")
    if retries != attempts - logical or exhaustions or failures:
        raise RuntimeError("router recorded unexpected retry or selection failure")
    if sessions < logical_expected:
        raise RuntimeError("router session-create metric omitted agentic requests")
    return {
        "logical_requests": logical,
        "external_attempts": attempts,
        "retries": retries,
        "retry_exhaustions": exhaustions,
        "selection_failures": int(failures),
        "session_creates": sessions,
    }
