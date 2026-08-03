#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded c4 ARC stage attribution plus equal-model OpenRouter controls."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import openrouter_agentic_benchmark as base
from modal_fullstack_canary import _read_metrics
from openrouter_agentic_preflight_contract import (
    MAX_EXTERNAL_ATTEMPTS as TRANSPORT_PREFLIGHT_EXTERNAL_ATTEMPTS,
)
from openrouter_agentic_preflight_contract import (
    MAX_PROVIDER_REQUESTS as TRANSPORT_PREFLIGHT_REQUESTS,
)
from openrouter_agentic_reporting import flatten, validate_router_metrics
from openrouter_agentic_stage_metrics import (
    encoder_client_from_environment,
    encoder_stage_delta,
    read_encoder_snapshot,
    router_snapshot,
    router_stage_delta,
)
from openrouter_agentic_stage_reporting import build_stage_report
from openrouter_agentic_workload import WORKERS
from openrouter_agentic_workload import candidate_case as _candidate_case

CONCURRENCY = 4
REPETITIONS = 2
UNIQUE_STRATIFIED_CASES_PER_MODEL = 2
NATURAL_PATHS = ("gateway_static", "arc")
STRATIFIED_PATHS = ("direct", "gateway_static")
NATURAL_REQUESTS_PER_PATH = base.SELECTED_CASE_COUNT * REPETITIONS
STRATIFIED_UNIQUE_CASES = len(WORKERS) * UNIQUE_STRATIFIED_CASES_PER_MODEL
STRATIFIED_REQUESTS_PER_PATH = STRATIFIED_UNIQUE_CASES * REPETITIONS
MEASURED_REQUESTS = (
    len(NATURAL_PATHS) * NATURAL_REQUESTS_PER_PATH
    + len(STRATIFIED_PATHS) * STRATIFIED_REQUESTS_PER_PATH
)
MAX_BENCHMARK_PROVIDER_REQUESTS = (
    base.KEY_READINESS_REQUESTS
    + base.ENDPOINT_PROBE_REQUESTS
    + base.MAX_COVERAGE_REQUESTS
    + MEASURED_REQUESTS
)
MAX_BENCHMARK_EXTERNAL_ATTEMPTS = (
    MAX_BENCHMARK_PROVIDER_REQUESTS * base.MAX_DATA_PLANE_ATTEMPTS
    + base.ENDPOINT_PROBE_REQUESTS
)
MAX_PROVIDER_REQUESTS = MAX_BENCHMARK_PROVIDER_REQUESTS + TRANSPORT_PREFLIGHT_REQUESTS
MAX_EXTERNAL_ATTEMPTS = (
    MAX_BENCHMARK_EXTERNAL_ATTEMPTS + TRANSPORT_PREFLIGHT_EXTERNAL_ATTEMPTS
)
MAX_REPORTED_PROVIDER_COST_USD = 0.50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _stratified_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for worker in WORKERS:
        for index in range(UNIQUE_STRATIFIED_CASES_PER_MODEL):
            case = dict(_candidate_case(index))
            case["expected_worker"] = worker
            cases.append(case)
    return cases


def _profiled_phase(
    *,
    path: str,
    cases: list[dict[str, Any]],
    wave: int,
    gateway_url: str,
    metrics_url: str,
    openrouter_key: str,
    run_id: str,
    timeout_seconds: float,
    encoder_client: Any | None = None,
) -> dict[str, Any]:
    router_before = router_snapshot(_read_metrics(metrics_url, timeout_seconds))
    encoder_before = (
        read_encoder_snapshot(encoder_client) if encoder_client is not None else None
    )
    phase = base._run_batch(
        path=path,
        cases=cases,
        concurrency=CONCURRENCY,
        wave=wave,
        gateway_url=gateway_url,
        openrouter_key=openrouter_key,
        run_id=run_id,
        timeout_seconds=timeout_seconds,
    )
    router_after = router_snapshot(_read_metrics(metrics_url, timeout_seconds))
    phase["stage_attribution"] = router_stage_delta(
        before=router_before,
        after=router_after,
        path=path,
        results=phase["results"],
    )
    if encoder_before is not None:
        encoder_after = read_encoder_snapshot(encoder_client)
        phase["protected_encoder_stage"] = encoder_stage_delta(
            before=encoder_before,
            after=encoder_after,
            requests=len(phase["results"]),
        )
    return phase


def _natural_phases(
    *,
    selected_cases: list[dict[str, Any]],
    gateway_url: str,
    metrics_url: str,
    openrouter_key: str,
    run_id: str,
    timeout_seconds: float,
    encoder_client: Any,
) -> list[dict[str, Any]]:
    measured_cases = selected_cases * REPETITIONS
    return [
        _profiled_phase(
            path=path,
            cases=measured_cases,
            wave=0,
            gateway_url=gateway_url,
            metrics_url=metrics_url,
            openrouter_key=openrouter_key,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            encoder_client=encoder_client if path == "arc" else None,
        )
        for path in NATURAL_PATHS
    ]


def _stratified_phases(
    *,
    unique_cases: list[dict[str, Any]],
    gateway_url: str,
    metrics_url: str,
    openrouter_key: str,
    run_id: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    measured_cases = unique_cases * REPETITIONS
    return [
        _profiled_phase(
            path=path,
            cases=measured_cases,
            wave=1,
            gateway_url=gateway_url,
            metrics_url=metrics_url,
            openrouter_key=openrouter_key,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
        )
        for path in STRATIFIED_PATHS
    ]


def _bounded_totals(
    *,
    key_readiness: dict[str, Any],
    endpoint_probes: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    phases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, float]:
    measured = flatten(phases)
    if len(measured) != MEASURED_REQUESTS:
        raise RuntimeError("stage diagnostic measured request count diverged")
    all_results = [key_readiness, *endpoint_probes, *coverage, *measured]
    if len(all_results) != MAX_BENCHMARK_PROVIDER_REQUESTS:
        raise RuntimeError("stage diagnostic provider request count diverged")
    attempts = sum(int(result["external_attempts"]) for result in all_results)
    if attempts > MAX_BENCHMARK_EXTERNAL_ATTEMPTS:
        raise RuntimeError("stage diagnostic external-attempt bound was exceeded")
    cost = sum(float(result["cost_usd"]) for result in all_results)
    if cost > MAX_REPORTED_PROVIDER_COST_USD:
        raise RuntimeError("stage diagnostic provider cost exceeded its gate")
    return all_results, attempts, cost


def _encode_report(report: dict[str, Any], openrouter_key: str) -> str:
    encoded = base._encode_private_report(report, openrouter_key)
    for name in (
        "RAYLINE_ARC_E2E_MODAL_KEY",
        "RAYLINE_ARC_E2E_MODAL_SECRET",
    ):
        protected = os.environ.get(name, "")
        if protected and protected in encoded:
            raise RuntimeError("Modal credential entered the stage report")
    return encoded


def main() -> None:
    args = _parse_args()
    openrouter_key = os.environ.get("OPENROUTER_EPHEMERAL_API_KEY", "")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_EPHEMERAL_API_KEY is required")
    transport_preflight = base._transport_preflight_from_environment()
    print("agentic stage encoder warmup: starting", file=sys.stderr, flush=True)
    encoder_warmup = base.warm_encoder_from_environment(
        timeout_seconds=args.timeout_seconds,
        connection_factory=base._connection,
    )
    metrics_before = _read_metrics(args.metrics_url, args.timeout_seconds)
    key_readiness = base._probe_key_readiness(
        gateway_url=args.gateway_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    endpoint_probes = base._probe_endpoints(
        gateway_url=args.gateway_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    selected_cases, coverage = base._discover_cases(
        gateway_url=args.gateway_url,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    base._bind_expected_workers(selected_cases, coverage)
    encoder_client = encoder_client_from_environment(args.timeout_seconds)
    natural_phases = _natural_phases(
        selected_cases=selected_cases,
        gateway_url=args.gateway_url,
        metrics_url=args.metrics_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
        encoder_client=encoder_client,
    )
    stratified_cases = _stratified_cases()
    stratified_phases = _stratified_phases(
        unique_cases=stratified_cases,
        gateway_url=args.gateway_url,
        metrics_url=args.metrics_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    phases = [*natural_phases, *stratified_phases]
    all_results, external_attempts, provider_cost = _bounded_totals(
        key_readiness=key_readiness,
        endpoint_probes=endpoint_probes,
        coverage=coverage,
        phases=phases,
    )
    external_attempts += int(transport_preflight["external_attempts"])
    provider_cost += float(transport_preflight["cost_usd"])
    if external_attempts > MAX_EXTERNAL_ATTEMPTS:
        raise RuntimeError("stage packet external-attempt bound was exceeded")
    if provider_cost > MAX_REPORTED_PROVIDER_COST_USD:
        raise RuntimeError("stage packet provider cost exceeded its gate")
    router_metrics = validate_router_metrics(
        before=metrics_before,
        after=_read_metrics(args.metrics_url, args.timeout_seconds),
        coverage=coverage,
        phases=phases,
    )
    report = build_stage_report(
        run_id=args.run_id,
        transport_preflight=transport_preflight,
        encoder_warmup=encoder_warmup,
        key_readiness=key_readiness,
        endpoint_probes=endpoint_probes,
        coverage=coverage,
        selected_cases=selected_cases,
        natural_phases=natural_phases,
        stratified_cases=stratified_cases,
        stratified_phases=stratified_phases,
        router_metrics=router_metrics,
        all_results=all_results,
        external_attempts=external_attempts,
        provider_cost=provider_cost,
        concurrency=CONCURRENCY,
        repetitions=REPETITIONS,
        maximum_provider_requests=MAX_PROVIDER_REQUESTS,
        maximum_external_attempts=MAX_EXTERNAL_ATTEMPTS,
        maximum_reported_provider_cost_usd=MAX_REPORTED_PROVIDER_COST_USD,
    )
    print(_encode_report(report, openrouter_key))


if __name__ == "__main__":
    main()
