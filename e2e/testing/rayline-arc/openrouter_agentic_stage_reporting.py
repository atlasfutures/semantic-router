#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-only report for c4 ARC attribution and model stratification."""

from __future__ import annotations

from typing import Any

from openrouter_agentic_reporting import (
    _coverage_report,
    _endpoint_report,
    _per_model_report,
    _readiness_report,
    _workload_report,
    flatten,
    result_report,
)
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS

PURE_MODAL_STAGE_REFERENCE = {
    "concurrency": 4,
    "static_router_mean_seconds": 0.000146,
    "arc_router_mean_seconds": 0.597,
    "arc_encoder_mean_seconds": 0.595,
    "arc_to_static_throughput_ratio": 0.755,
    "arc_minus_static_e2e_p95_seconds": 0.596,
    "generation": "two Modal L4 Qwen3.5-0.8B workers",
}


def _phase_report(phase: dict[str, Any]) -> dict[str, Any]:
    results = phase["results"]
    aggregate = result_report(results, float(phase["wall_seconds"]))
    aggregate["per_model"] = {
        worker: _per_model_report(
            [result for result in results if result["selected_worker"] == worker]
        )
        for worker in WORKERS
    }
    aggregate["stage_attribution"] = phase["stage_attribution"]
    if phase.get("protected_encoder_stage") is not None:
        aggregate["protected_encoder_stage"] = phase["protected_encoder_stage"]
    return aggregate


def _natural_stage_report(
    *,
    selected_cases: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    concurrency: int,
    repetitions: int,
) -> dict[str, Any]:
    by_path = {phase["path"]: _phase_report(phase) for phase in phases}
    static = by_path["gateway_static"]
    arc = by_path["arc"]
    return {
        "claim_scope": (
            "natural ARC mix with identical static and ARC payloads; no forced model"
        ),
        "workload": _workload_report(
            selected_cases=selected_cases,
            measured_requests=len(flatten(phases)),
            selected_case_count=len(selected_cases),
            minimum_active_workers=2,
            minimum_selected_cases_per_active_worker=2,
            max_completion_tokens=96,
            concurrency_levels=(concurrency,),
        ),
        "repetitions_per_case": repetitions,
        "paths": by_path,
        "comparison": {
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
            "arc_minus_static_e2e_p50_seconds": (
                arc["end_to_end_latency"]["p50_seconds"]
                - static["end_to_end_latency"]["p50_seconds"]
            ),
            "arc_minus_static_e2e_p95_seconds": (
                arc["end_to_end_latency"]["p95_seconds"]
                - static["end_to_end_latency"]["p95_seconds"]
            ),
            "pure_modal_reference": PURE_MODAL_STAGE_REFERENCE,
        },
    }


def _stratified_control_report(
    *,
    unique_cases: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    concurrency: int,
    repetitions: int,
) -> dict[str, Any]:
    return {
        "claim_scope": (
            "equal-model direct/static transport control; semantic ARC selection is "
            "not exercised and no natural-mix claim is made"
        ),
        "concurrency": concurrency,
        "unique_cases_per_model": {
            worker: sum(case["expected_worker"] == worker for case in unique_cases)
            for worker in WORKERS
        },
        "repetitions_per_case": repetitions,
        "measured_requests": len(flatten(phases)),
        "paths": {phase["path"]: _phase_report(phase) for phase in phases},
    }


def build_stage_report(
    *,
    run_id: str,
    transport_preflight: dict[str, Any],
    encoder_warmup: dict[str, Any],
    key_readiness: dict[str, Any],
    endpoint_probes: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    natural_phases: list[dict[str, Any]],
    stratified_cases: list[dict[str, Any]],
    stratified_phases: list[dict[str, Any]],
    router_metrics: dict[str, int],
    all_results: list[dict[str, Any]],
    external_attempts: int,
    provider_cost: float,
    concurrency: int,
    repetitions: int,
    maximum_provider_requests: int,
    maximum_external_attempts: int,
    maximum_reported_provider_cost_usd: float,
) -> dict[str, Any]:
    return {
        "schema_version": "rayline.arc.openrouter-agentic-stage-attribution.v1",
        "run_id": run_id,
        "status": "passed",
        "models": WORKERS,
        "provider_orders": PROVIDER_NAMES,
        "provider_fallbacks": False,
        "reasoning_enabled": False,
        "encoder_warmup": encoder_warmup,
        "pre_encoder_transport_preflight": transport_preflight,
        "openrouter_key_readiness": _readiness_report(key_readiness, 1),
        "endpoint_reachability": _endpoint_report(endpoint_probes),
        "coverage": _coverage_report(coverage),
        "natural_arc_stage_attribution": _natural_stage_report(
            selected_cases=selected_cases,
            phases=natural_phases,
            concurrency=concurrency,
            repetitions=repetitions,
        ),
        "stratified_model_control": _stratified_control_report(
            unique_cases=stratified_cases,
            phases=stratified_phases,
            concurrency=concurrency,
            repetitions=repetitions,
        ),
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
            "single-router c4 diagnostic, not a production SLO qualification",
            "12 requests per path in each natural or stratified control cell",
            "stratified controls bypass semantic ARC selection",
            "pure-Modal reference used different generation models and prompt lengths",
        ],
    }
