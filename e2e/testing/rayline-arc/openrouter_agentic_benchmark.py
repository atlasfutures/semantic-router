#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded agentic direct/static-gateway/Rayline ARC OpenRouter benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import sys
import time
from typing import Any

from modal_encoder_warmup import warm_encoder_from_environment
from modal_fullstack_canary import (
    _episode_id,
    _read_metrics,
)
from modal_fullstack_inputs import CANDIDATE_PROMPTS
from modal_http import connection_for_url as _connection
from modal_http import request_following_result_redirects
from openrouter_agentic_reporting import (
    comparison,
    flatten,
    path_reports,
    validate_router_metrics,
)
from openrouter_agentic_workload import (
    PROVIDER_NAMES,
    PROVIDER_SLUGS,
    SCENARIOS,
    WORKERS,
)
from openrouter_agentic_workload import candidate_case as _candidate_case
from openrouter_fullstack_canary import (
    OpenRouterHTTPError,
    _attempt_count,
    _http_error,
)

HTTP_OK = 200
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PATHS = ("direct", "gateway_static", "arc")
CONCURRENCY_LEVELS = (1, 4)
SERIAL_WAVES = 2
CONCURRENT_REPETITIONS = 2
SELECTED_CASE_COUNT = 6
MIN_ACTIVE_WORKERS = 2
MIN_SELECTED_CASES_PER_ACTIVE_WORKER = 2
MAX_COVERAGE_REQUESTS = 24
ENDPOINT_PROBE_REQUESTS = len(WORKERS)
MAX_COMPLETION_TOKENS = 96
MAX_REPORTED_PROVIDER_COST_USD = 0.50
MAX_DATA_PLANE_ATTEMPTS = 2
RETRYABLE_STATUS_CODES = frozenset({429, 503})
MEASURED_REQUESTS_PER_PATH = (
    SELECTED_CASE_COUNT * SERIAL_WAVES + SELECTED_CASE_COUNT * CONCURRENT_REPETITIONS
)
MAX_MEASURED_REQUESTS = len(PATHS) * MEASURED_REQUESTS_PER_PATH
MAX_PROVIDER_REQUESTS = (
    ENDPOINT_PROBE_REQUESTS + MAX_COVERAGE_REQUESTS + MAX_MEASURED_REQUESTS
)
MAX_EXTERNAL_ATTEMPTS = MAX_PROVIDER_REQUESTS * MAX_DATA_PLANE_ATTEMPTS


def _response_model_matches(response_model: str, expected_model: str) -> bool:
    return response_model == expected_model or response_model.startswith(
        f"{expected_model}-"
    )


def _request_payload(
    *, path: str, case: dict[str, Any], expected_worker: str
) -> dict[str, Any]:
    if path == "direct":
        model = WORKERS[expected_worker]
    elif path == "gateway_static":
        model = expected_worker
    elif path == "arc":
        model = "auto"
    else:
        raise ValueError(f"unsupported benchmark path: {path}")
    payload: dict[str, Any] = {
        "model": model,
        "messages": case["messages"],
        "tools": case["tools"],
        "tool_choice": "none",
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if path != "arc":
        payload.update(
            {
                "provider": {
                    "order": [PROVIDER_SLUGS[expected_worker]],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "reasoning": {"enabled": False, "effort": "none"},
            }
        )
    return payload


def _request_headers(
    *, path: str, openrouter_key: str, episode_id: str
) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if path == "direct":
        headers["authorization"] = f"Bearer {openrouter_key}"
    if path == "arc":
        headers["x-rayline-episode-id"] = episode_id
    return headers


def _event_emits_token(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    choices = event.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        tool_calls = delta.get("tool_calls")
        if (isinstance(content, str) and content) or (
            isinstance(tool_calls, list) and tool_calls
        ):
            return True
    return False


def _read_stream(connection: Any, response: Any, started: float) -> dict[str, Any]:
    first_event_seconds: float | None = None
    first_token_seconds: float | None = None
    data_events = 0
    saw_done = False
    usage: dict[str, Any] | None = None
    provider = ""
    response_model = ""
    try:
        while line := response.readline():
            stripped = line.decode(errors="replace").strip()
            if not stripped.startswith("data:"):
                continue
            data = stripped.removeprefix("data:").strip()
            if data == "[DONE]":
                saw_done = True
                break
            event = json.loads(data)
            data_events += 1
            elapsed = time.perf_counter() - started
            first_event_seconds = first_event_seconds or elapsed
            if isinstance(event, dict):
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                provider = str(event.get("provider") or provider)
                response_model = str(event.get("model") or response_model)
            if first_token_seconds is None and _event_emits_token(event):
                first_token_seconds = elapsed
    finally:
        connection.close()
    return {
        "first_event_seconds": first_event_seconds,
        "first_token_seconds": first_token_seconds,
        "total_seconds": time.perf_counter() - started,
        "data_events": data_events,
        "saw_done": saw_done,
        "usage": usage,
        "provider": provider,
        "response_model": response_model,
    }


def _completed_usage(stream: dict[str, Any]) -> dict[str, Any]:
    if (
        not stream["saw_done"]
        or not stream["data_events"]
        or stream["first_event_seconds"] is None
    ):
        raise RuntimeError("agentic streaming response was incomplete")
    if stream["first_token_seconds"] is None:
        raise RuntimeError("agentic streaming response emitted no content token")
    usage = stream["usage"]
    if not isinstance(usage, dict) or not isinstance(usage.get("cost"), (int, float)):
        raise TypeError("agentic streaming response omitted OpenRouter usage cost")
    return usage


def _upstream_seconds(path: str, upstream_millis: str) -> float | None:
    if not upstream_millis:
        if path != "direct":
            raise RuntimeError("gateway response omitted Envoy upstream service time")
        return None
    try:
        return float(upstream_millis) / 1000.0
    except ValueError as error:
        raise RuntimeError("Envoy upstream service time was invalid") from error


def _validate_frozen_target(
    *,
    path: str,
    expected_worker: str,
    selected_worker: str,
    response_model: str,
    provider: str,
    attempts: int,
) -> None:
    expected_model = WORKERS[expected_worker]
    if not _response_model_matches(response_model, expected_model):
        raise RuntimeError("agentic response model did not match the frozen target")
    if provider != PROVIDER_NAMES[expected_worker]:
        raise RuntimeError("agentic response used the wrong pinned provider")
    if path == "arc" and selected_worker != expected_worker:
        raise RuntimeError("ARC selection changed for a frozen agentic case")
    if path == "gateway_static" and selected_worker not in {"", expected_worker}:
        raise RuntimeError("static gateway selected the wrong worker")
    if attempts > MAX_DATA_PLANE_ATTEMPTS:
        raise RuntimeError("agentic request exceeded the data-plane attempt bound")


def _stream_request_once(
    *,
    path: str,
    case: dict[str, Any],
    expected_worker: str,
    gateway_url: str,
    openrouter_key: str,
    episode_id: str,
    timeout_seconds: float,
    started: float,
) -> dict[str, Any]:
    direct = path == "direct"
    base_url = OPENROUTER_BASE_URL if direct else f"{gateway_url.rstrip('/')}/v1"
    headers = _request_headers(
        path=path,
        openrouter_key=openrouter_key,
        episode_id=episode_id,
    )
    connection, response = request_following_result_redirects(
        connection_factory=_connection,
        method="POST",
        url=f"{base_url}/chat/completions",
        body=json.dumps(
            _request_payload(
                path=path,
                case=case,
                expected_worker=expected_worker,
            ),
            separators=(",", ":"),
        ).encode(),
        headers=headers,
        timeout_seconds=timeout_seconds,
    )
    selected_worker = response.getheader("x-vsr-selected-model", "")
    attempts = _attempt_count(response.getheader("x-envoy-attempt-count"))
    upstream_millis = response.getheader("x-envoy-upstream-service-time", "")
    if response.status != HTTP_OK:
        body = response.read()
        retry_after = response.getheader("retry-after")
        connection.close()
        raise _http_error(
            endpoint=f"{path} streaming endpoint",
            status_code=response.status,
            body=body,
            retry_after=retry_after,
        )

    stream = _read_stream(connection, response, started)
    usage = _completed_usage(stream)
    _validate_frozen_target(
        path=path,
        expected_worker=expected_worker,
        selected_worker=selected_worker,
        response_model=stream["response_model"],
        provider=stream["provider"],
        attempts=attempts,
    )
    return {
        "path": path,
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "selected_worker": expected_worker,
        "response_model": stream["response_model"],
        "provider": stream["provider"],
        "external_attempts": attempts,
        "time_to_first_event_seconds": stream["first_event_seconds"],
        "time_to_first_token_seconds": stream["first_token_seconds"],
        "total_seconds": stream["total_seconds"],
        "envoy_upstream_service_seconds": _upstream_seconds(path, upstream_millis),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost_usd": float(usage["cost"]),
        "data_events": stream["data_events"],
    }


def _stream_request(
    *,
    path: str,
    case: dict[str, Any],
    expected_worker: str,
    gateway_url: str,
    openrouter_key: str,
    episode_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    maximum_attempts = MAX_DATA_PLANE_ATTEMPTS if path == "direct" else 1
    for attempt in range(1, maximum_attempts + 1):
        try:
            result = _stream_request_once(
                path=path,
                case=case,
                expected_worker=expected_worker,
                gateway_url=gateway_url,
                openrouter_key=openrouter_key,
                episode_id=episode_id,
                timeout_seconds=timeout_seconds,
                started=started,
            )
        except OpenRouterHTTPError as error:
            if (
                attempt == maximum_attempts
                or error.status_code not in RETRYABLE_STATUS_CODES
            ):
                raise
            time.sleep(error.retry_after_seconds)
            continue
        if path == "direct":
            result["external_attempts"] = attempt
        return result
    raise AssertionError("bounded direct retry loop did not return or raise")


def _choose_cases(
    candidates: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    active_workers = [
        worker
        for worker in WORKERS
        if len(candidates[worker]) >= MIN_SELECTED_CASES_PER_ACTIVE_WORKER
    ]
    if len(active_workers) < MIN_ACTIVE_WORKERS:
        return None
    allocations = [
        counts
        for counts in itertools.product(
            *(
                range(
                    MIN_SELECTED_CASES_PER_ACTIVE_WORKER,
                    min(len(candidates[worker]), SELECTED_CASE_COUNT) + 1,
                )
                for worker in active_workers
            )
        )
        if sum(counts) == SELECTED_CASE_COUNT
    ]
    allocations.sort(key=lambda counts: (max(counts) - min(counts), counts))
    for allocation in allocations:
        combinations = [
            list(itertools.combinations(candidates[worker], count))
            for worker, count in zip(active_workers, allocation, strict=True)
        ]
        for worker_groups in itertools.product(*combinations):
            selected = [case for group in worker_groups for case in group]
            if {case["scenario"] for case in selected} == set(SCENARIOS):
                return selected
    return None


def _probe_endpoints(
    *, gateway_url: str, openrouter_key: str, run_id: str, timeout_seconds: float
) -> list[dict[str, Any]]:
    print("agentic endpoint probes: starting", file=sys.stderr, flush=True)
    return [
        _stream_request(
            path="gateway_static",
            case=_candidate_case(index),
            expected_worker=worker,
            gateway_url=gateway_url,
            openrouter_key=openrouter_key,
            episode_id=_episode_id(run_id, f"agentic-endpoint-{worker}"),
            timeout_seconds=timeout_seconds,
        )
        for index, worker in enumerate(WORKERS)
    ]


def _coverage_request(
    *,
    case: dict[str, Any],
    gateway_url: str,
    episode_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    # Discovery cannot supply an expected worker. Run the same stream path with
    # a provisional target, then validate model/provider from the returned ARC
    # worker. The implementation is intentionally separate from measured calls.
    payload = {
        "model": "auto",
        "messages": case["messages"],
        "tools": case["tools"],
        "tool_choice": "none",
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    connection, response = request_following_result_redirects(
        connection_factory=_connection,
        method="POST",
        url=f"{gateway_url.rstrip('/')}/v1/chat/completions",
        body=json.dumps(payload, separators=(",", ":")).encode(),
        headers=_request_headers(
            path="arc",
            openrouter_key="",
            episode_id=episode_id,
        ),
        timeout_seconds=timeout_seconds,
    )
    selected_worker = response.getheader("x-vsr-selected-model", "")
    attempts = _attempt_count(response.getheader("x-envoy-attempt-count"))
    if response.status != HTTP_OK:
        body = response.read()
        retry_after = response.getheader("retry-after")
        connection.close()
        raise _http_error(
            endpoint="agentic coverage endpoint",
            status_code=response.status,
            body=body,
            retry_after=retry_after,
        )
    stream = _read_stream(connection, response, started)
    usage = _completed_usage(stream)
    if selected_worker not in WORKERS:
        raise RuntimeError("agentic coverage response was incomplete")
    if attempts > MAX_DATA_PLANE_ATTEMPTS:
        raise RuntimeError("agentic coverage exceeded the data-plane attempt bound")
    if stream["provider"] != PROVIDER_NAMES[selected_worker]:
        raise RuntimeError("agentic coverage used the wrong pinned provider")
    if not _response_model_matches(stream["response_model"], WORKERS[selected_worker]):
        raise RuntimeError("agentic coverage returned the wrong model")
    return {
        "path": "arc",
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "selected_worker": selected_worker,
        "response_model": stream["response_model"],
        "provider": stream["provider"],
        "external_attempts": attempts,
        "time_to_first_event_seconds": stream["first_event_seconds"],
        "time_to_first_token_seconds": stream["first_token_seconds"],
        "total_seconds": stream["total_seconds"],
        "envoy_upstream_service_seconds": None,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "cost_usd": float(usage["cost"]),
        "data_events": stream["data_events"],
    }


def _discover_cases(
    *, gateway_url: str, run_id: str, timeout_seconds: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {worker: [] for worker in WORKERS}
    results: list[dict[str, Any]] = []
    for index in range(MAX_COVERAGE_REQUESTS):
        case = _candidate_case(index)
        result = _coverage_request(
            case=case,
            gateway_url=gateway_url,
            episode_id=_episode_id(run_id, f"agentic-coverage-{index}"),
            timeout_seconds=timeout_seconds,
        )
        candidates[result["selected_worker"]].append(case)
        results.append(result)
        print(
            f"agentic coverage: {index + 1} request(s), "
            f"counts={[len(candidates[worker]) for worker in WORKERS]}",
            file=sys.stderr,
            flush=True,
        )
    selected = _choose_cases(candidates)
    if selected is None:
        raise RuntimeError("agentic candidates did not cover two active workers")
    return selected, results


def _run_batch(
    *,
    path: str,
    cases: list[dict[str, Any]],
    concurrency: int,
    wave: int,
    gateway_url: str,
    openrouter_key: str,
    run_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    phase = f"{path}-c{concurrency}-w{wave}"
    print(f"agentic benchmark {phase}: starting", file=sys.stderr, flush=True)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _stream_request,
                path=path,
                case=case,
                expected_worker=case["expected_worker"],
                gateway_url=gateway_url,
                openrouter_key=openrouter_key,
                episode_id=_episode_id(run_id, f"{phase}-{index}"),
                timeout_seconds=timeout_seconds,
            )
            for index, case in enumerate(cases)
        ]
        results = [future.result() for future in futures]
    return {
        "path": path,
        "concurrency": concurrency,
        "wave": wave,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
    }


def _rotated_paths(offset: int) -> tuple[str, ...]:
    shift = offset % len(PATHS)
    return (*PATHS[shift:], *PATHS[:shift])


def _run_measured(
    *,
    selected_cases: list[dict[str, Any]],
    gateway_url: str,
    openrouter_key: str,
    run_id: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for wave in range(SERIAL_WAVES):
        for path in _rotated_paths(wave):
            phases.append(
                _run_batch(
                    path=path,
                    cases=selected_cases,
                    concurrency=1,
                    wave=wave,
                    gateway_url=gateway_url,
                    openrouter_key=openrouter_key,
                    run_id=run_id,
                    timeout_seconds=timeout_seconds,
                )
            )
    concurrent_cases = selected_cases * CONCURRENT_REPETITIONS
    for path in _rotated_paths(SERIAL_WAVES):
        phases.append(
            _run_batch(
                path=path,
                cases=concurrent_cases,
                concurrency=4,
                wave=0,
                gateway_url=gateway_url,
                openrouter_key=openrouter_key,
                run_id=run_id,
                timeout_seconds=timeout_seconds,
            )
        )
    return phases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _bind_expected_workers(
    selected_cases: list[dict[str, Any]], coverage: list[dict[str, Any]]
) -> None:
    selected_by_id = {case["case_id"]: case for case in selected_cases}
    for result in coverage:
        case = selected_by_id.get(result["case_id"])
        if case is not None:
            case["expected_worker"] = result["selected_worker"]
    if any("expected_worker" not in case for case in selected_cases):
        raise RuntimeError("selected agentic case lost its coverage worker")


def _bounded_totals(
    endpoint_probes: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    phases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, float]:
    measured_results = flatten(phases)
    all_results = [*endpoint_probes, *coverage, *measured_results]
    if len(measured_results) != MAX_MEASURED_REQUESTS:
        raise RuntimeError("agentic measured request count diverged")
    if len(all_results) > MAX_PROVIDER_REQUESTS:
        raise RuntimeError("agentic provider request bound was exceeded")
    external_attempts = sum(int(result["external_attempts"]) for result in all_results)
    if external_attempts > MAX_EXTERNAL_ATTEMPTS:
        raise RuntimeError("agentic external-attempt bound was exceeded")
    provider_cost = sum(float(result["cost_usd"]) for result in all_results)
    if provider_cost > MAX_REPORTED_PROVIDER_COST_USD:
        raise RuntimeError("agentic provider cost exceeded its reported-cost gate")
    return all_results, external_attempts, provider_cost


def _build_report(
    *,
    run_id: str,
    encoder_warmup: dict[str, Any],
    endpoint_probes: list[dict[str, Any]],
    selected_cases: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    router_metrics: dict[str, int],
    all_results: list[dict[str, Any]],
    external_attempts: int,
    provider_cost: float,
) -> dict[str, Any]:
    measured_results = flatten(phases)
    reports = path_reports(
        phases,
        paths=PATHS,
        concurrency_levels=CONCURRENCY_LEVELS,
    )
    return {
        "schema_version": "rayline.arc.openrouter-agentic-benchmark.v2",
        "run_id": run_id,
        "status": "passed",
        "models": WORKERS,
        "pinned_providers": PROVIDER_NAMES,
        "provider_fallbacks": False,
        "reasoning_enabled": False,
        "encoder_warmup": encoder_warmup,
        "workload": {
            "scenarios": sorted(SCENARIOS),
            "selected_case_counts_by_scenario": {
                scenario: sum(case["scenario"] == scenario for case in selected_cases)
                for scenario in SCENARIOS
            },
            "selected_case_counts_by_worker": {
                worker: sum(
                    case["expected_worker"] == worker for case in selected_cases
                )
                for worker in WORKERS
            },
            "selected_case_count": SELECTED_CASE_COUNT,
            "minimum_active_workers": MIN_ACTIVE_WORKERS,
            "minimum_selected_cases_per_active_worker": (
                MIN_SELECTED_CASES_PER_ACTIVE_WORKER
            ),
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "measured_requests": len(measured_results),
        },
        "coverage": {
            "requests": len(coverage),
            "selection_counts": {
                worker: sum(result["selected_worker"] == worker for result in coverage)
                for worker in WORKERS
            },
            "cost_usd": sum(float(result["cost_usd"]) for result in coverage),
        },
        "endpoint_reachability": {
            "requests": len(endpoint_probes),
            "all_reachable": len(endpoint_probes) == len(WORKERS),
            "workers": {
                worker: {
                    "reachable": any(
                        result["selected_worker"] == worker
                        for result in endpoint_probes
                    ),
                    "model": model,
                    "provider": PROVIDER_NAMES[worker],
                }
                for worker, model in WORKERS.items()
            },
            "external_attempts": sum(
                int(result["external_attempts"]) for result in endpoint_probes
            ),
            "retries": sum(
                int(result["external_attempts"]) for result in endpoint_probes
            )
            - len(endpoint_probes),
            "cost_usd": sum(float(result["cost_usd"]) for result in endpoint_probes),
        },
        "paths": reports,
        "comparison": comparison(reports, CONCURRENCY_LEVELS),
        "router_metrics": router_metrics,
        "actual_provider_requests": len(all_results),
        "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
        "actual_external_attempts": external_attempts,
        "maximum_external_attempts": MAX_EXTERNAL_ATTEMPTS,
        "reported_provider_cost_usd": provider_cost,
        "maximum_reported_provider_cost_usd": MAX_REPORTED_PROVIDER_COST_USD,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
        "limitations": [
            "small diagnostic sample, not a production SLO qualification",
            "synthetic public tool outputs and routing anchors",
            "pure-Modal reference used different generation models and prompt lengths",
        ],
    }


def _encode_private_report(report: dict[str, Any], openrouter_key: str) -> str:
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if openrouter_key in encoded:
        raise RuntimeError("OpenRouter credential entered the agentic report")
    for anchor in CANDIDATE_PROMPTS:
        if anchor in encoded:
            raise RuntimeError("agentic report included a routing anchor")
    return encoded


def main() -> None:
    args = _parse_args()
    openrouter_key = os.environ.get("OPENROUTER_EPHEMERAL_API_KEY", "")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_EPHEMERAL_API_KEY is required")
    print("agentic encoder warmup: starting", file=sys.stderr, flush=True)
    encoder_warmup = warm_encoder_from_environment(
        timeout_seconds=args.timeout_seconds,
        connection_factory=_connection,
    )
    metrics_before = _read_metrics(args.metrics_url, args.timeout_seconds)
    endpoint_probes = _probe_endpoints(
        gateway_url=args.gateway_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    selected_cases, coverage = _discover_cases(
        gateway_url=args.gateway_url,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    _bind_expected_workers(selected_cases, coverage)
    phases = _run_measured(
        selected_cases=selected_cases,
        gateway_url=args.gateway_url,
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    all_results, external_attempts, provider_cost = _bounded_totals(
        endpoint_probes, coverage, phases
    )
    router_metrics = validate_router_metrics(
        before=metrics_before,
        after=_read_metrics(args.metrics_url, args.timeout_seconds),
        coverage=coverage,
        phases=phases,
    )
    report = _build_report(
        run_id=args.run_id,
        encoder_warmup=encoder_warmup,
        endpoint_probes=endpoint_probes,
        selected_cases=selected_cases,
        coverage=coverage,
        phases=phases,
        router_metrics=router_metrics,
        all_results=all_results,
        external_attempts=external_attempts,
        provider_cost=provider_cost,
    )
    print(_encode_private_report(report, openrouter_key))


if __name__ == "__main__":
    main()
