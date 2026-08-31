#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the AGT014 agentic workload against the native Modal Rayline router."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

import openrouter_agentic_benchmark as base
from modal_fullstack_canary import _episode_id, _summary
from modal_http import connection_for_url as _connection
from modal_http import request_following_result_redirects
from openrouter_agentic_reporting import result_report
from openrouter_agentic_stage_benchmark import (
    CONCURRENCY,
    REPETITIONS,
    UNIQUE_STRATIFIED_CASES_PER_MODEL,
)
from openrouter_agentic_workload import PROVIDER_NAMES, WORKERS
from openrouter_agentic_workload import candidate_case as _candidate_case
from openrouter_fullstack_canary import _http_error

HTTP_OK = 200
MAX_COVERAGE_REQUESTS = 24
SELECTED_CASE_COUNT = 6
MAX_COMPLETION_TOKENS = 96
EXPECTED_NATIVE_REQUESTS = 63
EXPECTED_DIRECT_REQUESTS = 13
MAX_PROVIDER_REQUESTS = EXPECTED_NATIVE_REQUESTS + EXPECTED_DIRECT_REQUESTS
AGT013_REFERENCE = {
    "static_requests_per_second": 1.3477498579,
    "arc_requests_per_second": 0.8388886281,
    "arc_to_static_throughput_ratio": 0.6224364434,
    "static_e2e_p50_seconds": 2.261819,
    "static_e2e_p95_seconds": 4.942118,
    "arc_e2e_p50_seconds": 4.157048,
    "arc_e2e_p95_seconds": 7.374518,
    "arc_router_mean_seconds": 1.490476,
    "arc_encoder_mean_seconds": 1.485365,
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _native_payload(case: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "model": model,
        "messages": case["messages"],
        "tools": case["tools"],
        "tool_choice": "none",
        "max_tokens": MAX_COMPLETION_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _read_native_stream(
    connection: Any, response: Any, started: float
) -> dict[str, Any]:
    first_event: float | None = None
    first_token: float | None = None
    response_model = ""
    events = 0
    done = False
    try:
        while line := response.readline():
            stripped = line.decode(errors="replace").strip()
            if not stripped.startswith("data:"):
                continue
            data = stripped.removeprefix("data:").strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            events += 1
            elapsed = time.perf_counter() - started
            first_event = first_event or elapsed
            response_model = str(event.get("model") or response_model)
            if first_token is None and base._event_emits_token(event):
                first_token = elapsed
    finally:
        connection.close()
    if not done or not events or first_event is None or first_token is None:
        raise RuntimeError("native Modal streaming shim returned an incomplete stream")
    return {
        "first_event_seconds": first_event,
        "first_token_seconds": first_token,
        "total_seconds": time.perf_counter() - started,
        "response_model": response_model,
        "data_events": events,
    }


def _native_request(
    *,
    path: str,
    case: dict[str, Any],
    expected_worker: str | None,
    router_url: str,
    router_token: str,
    episode_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if path not in {"modal_static", "modal_arc"}:
        raise ValueError(f"unsupported native path {path}")
    model = f"rayline/{expected_worker}" if path == "modal_static" else "rayline/router"
    started = time.perf_counter()
    connection, response = request_following_result_redirects(
        connection_factory=_connection,
        method="POST",
        url=f"{router_url.rstrip('/')}/v1/chat/completions",
        body=json.dumps(_native_payload(case, model), separators=(",", ":")).encode(),
        headers={
            "authorization": f"Bearer {router_token}",
            "content-type": "application/json",
            "x-rayline-episode-id": episode_id,
        },
        timeout_seconds=timeout_seconds,
    )
    selected_worker = response.getheader("x-rayline-worker", "")
    request_id = response.getheader("x-rayline-request-id", "")
    if response.status != HTTP_OK:
        body = response.read()
        retry_after = response.getheader("retry-after")
        connection.close()
        raise _http_error(
            endpoint=path,
            status_code=response.status,
            body=body,
            retry_after=retry_after,
        )
    stream = _read_native_stream(connection, response, started)
    if selected_worker not in WORKERS or not request_id:
        raise RuntimeError("native Modal response omitted routing identity")
    if expected_worker is not None and selected_worker != expected_worker:
        raise RuntimeError("native Modal selection changed for a frozen case")
    return {
        "path": path,
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "request_id": request_id,
        "selected_worker": selected_worker,
        "response_model": stream["response_model"],
        "time_to_first_event_seconds": stream["first_event_seconds"],
        "time_to_first_token_seconds": stream["first_token_seconds"],
        "total_seconds": stream["total_seconds"],
        "data_events": stream["data_events"],
    }


def _batch(
    *,
    path: str,
    cases: list[dict[str, Any]],
    router_url: str,
    router_token: str,
    run_id: str,
    phase: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [
            pool.submit(
                _native_request,
                path=path,
                case=case,
                expected_worker=case.get("expected_worker"),
                router_url=router_url,
                router_token=router_token,
                episode_id=_episode_id(run_id, f"{phase}-{index}"),
                timeout_seconds=timeout_seconds,
            )
            for index, case in enumerate(cases)
        ]
        results = [future.result() for future in futures]
    return {
        "path": path,
        "concurrency": CONCURRENCY,
        "phase": phase,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
    }


def _coverage(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {worker: [] for worker in WORKERS}
    results = []
    for index in range(MAX_COVERAGE_REQUESTS):
        case = _candidate_case(index)
        result = _native_request(
            path="modal_arc",
            case=case,
            expected_worker=None,
            router_url=args.router_url,
            router_token=args.router_token,
            episode_id=_episode_id(args.run_id, f"modal-native-coverage-{index}"),
            timeout_seconds=args.timeout_seconds,
        )
        candidates[result["selected_worker"]].append(case)
        results.append(result)
        print(
            "native Modal coverage: "
            f"{index + 1}/{MAX_COVERAGE_REQUESTS} "
            f"{[len(candidates[worker]) for worker in WORKERS]}",
            flush=True,
        )
    selected = base._choose_cases(candidates)
    if selected is None or len(selected) != SELECTED_CASE_COUNT:
        raise RuntimeError(
            "native Modal coverage did not produce the frozen six-case cell"
        )
    selected_by_id = {case["case_id"]: case for case in selected}
    for result in results:
        if result["case_id"] in selected_by_id:
            selected_by_id[result["case_id"]]["expected_worker"] = result[
                "selected_worker"
            ]
    return selected, results


def _stratified_cases() -> list[dict[str, Any]]:
    result = []
    for worker in WORKERS:
        for index in range(UNIQUE_STRATIFIED_CASES_PER_MODEL):
            case = dict(_candidate_case(index))
            case["expected_worker"] = worker
            result.append(case)
    return result


def _client_phase_summary(phase: dict[str, Any]) -> dict[str, Any]:
    results = phase["results"]
    return {
        "requests": len(results),
        "wall_seconds": phase["wall_seconds"],
        "requests_per_second": len(results) / phase["wall_seconds"],
        "observed_first_token": _summary(
            [result["time_to_first_token_seconds"] for result in results]
        ),
        "end_to_end_latency": _summary([result["total_seconds"] for result in results]),
    }


def _native_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *report["endpoint_probes"],
        *report["coverage"],
        *[
            result
            for phase in [
                *report["natural_phases"],
                report["stratified_phases"][1],
            ]
            for result in phase["results"]
        ],
    ]


def read_decisions(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [
        row for row in rows if row.get("schema_version") == "rayline-router.decision.v3"
    ]


def _decision_cost(row: dict[str, Any]) -> float:
    basis = row.get("settlement_cost_basis")
    if isinstance(basis, dict) and isinstance(
        basis.get("provider_charged_usd"), (int, float)
    ):
        return float(basis["provider_charged_usd"])
    return float(row.get("estimated_cost") or 0)


def _enrich_native_results(
    report: dict[str, Any], decisions: list[dict[str, Any]]
) -> None:
    by_request = {str(row.get("request_id") or ""): row for row in decisions}
    if len(decisions) != EXPECTED_NATIVE_REQUESTS or len(by_request) != len(decisions):
        raise RuntimeError("native Modal decision log count or identity diverged")
    for result in _native_results(report):
        row = by_request.get(str(result["request_id"]))
        if row is None or row.get("error"):
            raise RuntimeError(
                "native Modal client request did not join a clean decision"
            )
        worker = str(result["selected_worker"])
        if (
            row.get("selected_worker") != worker
            or row.get("worker_model") != WORKERS[worker]
        ):
            raise RuntimeError("native Modal decision execution identity diverged")
        provider = str(
            row.get("served_provider") or row.get("openrouter_provider_name") or ""
        )
        if provider not in PROVIDER_NAMES[worker]:
            raise RuntimeError(
                "native Modal decision used a provider outside its order"
            )
        transport = (
            row.get("transport") if isinstance(row.get("transport"), dict) else {}
        )
        attempts = transport.get("attempts")
        attempt_count = len(attempts) if isinstance(attempts, list) else 1
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        result.update(
            {
                "response_model": str(row.get("served_model") or row["worker_model"]),
                "provider": provider,
                "external_attempts": attempt_count,
                "prompt_tokens": int(row.get("input_tokens") or 0),
                "completion_tokens": int(row.get("output_tokens") or 0),
                "cost_usd": _decision_cost(row),
                "decision_latency_seconds": float(row.get("decision_latency_ms") or 0)
                / 1000,
                "embedding_latency_seconds": float(
                    features.get("embedding_latency_ms") or 0
                )
                / 1000,
                "q_latency_seconds": float(features.get("q_latency_ms") or 0) / 1000,
                "encode_mode": str(features.get("encode_mode") or "not_applicable"),
                "serialized_tokens": int(features.get("serialized_tokens") or 0),
            }
        )


def _native_phase_report(phase: dict[str, Any]) -> dict[str, Any]:
    results = phase["results"]
    output_tokens = sum(result["completion_tokens"] for result in results)
    attempts = sum(result["external_attempts"] for result in results)
    modes = sorted({result["encode_mode"] for result in results})

    def stage_summary(field: str) -> dict[str, float]:
        values = [float(result[field]) for result in results]
        return {**_summary(values), "mean_seconds": sum(values) / len(values)}

    return {
        "requests": len(results),
        "wall_seconds": phase["wall_seconds"],
        "requests_per_second": len(results) / phase["wall_seconds"],
        "output_tokens_per_second": output_tokens / phase["wall_seconds"],
        "observed_first_token_after_buffering": _summary(
            [result["time_to_first_token_seconds"] for result in results]
        ),
        "end_to_end_latency": _summary([result["total_seconds"] for result in results]),
        "decision_latency": stage_summary("decision_latency_seconds"),
        "embedding_latency": stage_summary("embedding_latency_seconds"),
        "q_latency": stage_summary("q_latency_seconds"),
        "prompt_tokens": sum(result["prompt_tokens"] for result in results),
        "completion_tokens": output_tokens,
        "external_attempts": attempts,
        "retries": attempts - len(results),
        "cost_usd": sum(result["cost_usd"] for result in results),
        "providers": sorted({result["provider"] for result in results}),
        "models": sorted({result["response_model"] for result in results}),
        "encode_modes": {
            mode: sum(result["encode_mode"] == mode for result in results)
            for mode in modes
        },
    }


def finalize_report(
    *,
    client_report: dict[str, Any],
    decisions: list[dict[str, Any]],
    actual_openrouter_cost_usd: float,
    deployment: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    _enrich_native_results(client_report, decisions)
    natural = {
        phase["path"]: _native_phase_report(phase)
        for phase in client_report["natural_phases"]
    }
    static = natural["modal_static"]
    arc = natural["modal_arc"]
    direct_phase, stratified_static_phase = client_report["stratified_phases"]
    direct = result_report(direct_phase["results"], direct_phase["wall_seconds"])
    stratified_static = _native_phase_report(stratified_static_phase)
    native_results = _native_results(client_report)
    actual_requests = len(native_results) + 1 + len(direct_phase["results"])
    if actual_requests != MAX_PROVIDER_REQUESTS:
        raise RuntimeError("final native Modal provider request count diverged")
    return {
        "schema_version": "rayline.openrouter-modal-native.stage-attribution.v1",
        "run_id": client_report["run_id"],
        "status": "passed",
        "deployment": deployment,
        "checkpoint": checkpoint,
        "models": WORKERS,
        "provider_orders": PROVIDER_NAMES,
        "workload": {
            "coverage_requests": len(client_report["coverage"]),
            "coverage_selection_counts": {
                worker: sum(
                    result["selected_worker"] == worker
                    for result in client_report["coverage"]
                )
                for worker in WORKERS
            },
            "selected_cases": client_report["selected_cases"],
            "concurrency": CONCURRENCY,
            "repetitions": REPETITIONS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        },
        "natural_paths": natural,
        "natural_comparison": {
            "arc_to_static_throughput_ratio": (
                arc["requests_per_second"] / static["requests_per_second"]
            ),
            "arc_to_static_output_throughput_ratio": (
                arc["output_tokens_per_second"] / static["output_tokens_per_second"]
            ),
            "arc_minus_static_e2e_p50_seconds": (
                arc["end_to_end_latency"]["p50_seconds"]
                - static["end_to_end_latency"]["p50_seconds"]
            ),
            "arc_minus_static_e2e_p95_seconds": (
                arc["end_to_end_latency"]["p95_seconds"]
                - static["end_to_end_latency"]["p95_seconds"]
            ),
            "agt013_remote_vllm_reference": AGT013_REFERENCE,
        },
        "stratified_control": {
            "direct": direct,
            "modal_static": stratified_static,
        },
        "actual_provider_requests": actual_requests,
        "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
        "actual_external_attempts": (
            sum(result["external_attempts"] for result in native_results)
            + int(client_report["key_readiness"]["external_attempts"])
            + sum(
                int(result["external_attempts"]) for result in direct_phase["results"]
            )
        ),
        "reported_provider_cost_usd": (
            sum(result["cost_usd"] for result in native_results)
            + float(client_report["key_readiness"]["cost_usd"])
            + sum(float(result["cost_usd"]) for result in direct_phase["results"])
        ),
        "actual_openrouter_key_cost_usd": actual_openrouter_cost_usd,
        "automatic_prefix_cache_enabled": True,
        "release_qualification_1000_executed": False,
        "limitations": [
            "single Modal L40S router and concurrency-four diagnostic, not an SLO",
            "OpenRouter provider variance remains outside router control",
            "native Modal buffers provider completion before emitting OpenAI SSE, so its observed first-token metric is not provider TTFT",
            "native Modal holds one per-token lock across routing, provider completion, and persistence, so this single-token c4 workload is serialized",
            "the checkpoint reproduces the AGT013 synthetic policy head; it is not the private production C82 checkpoint",
        ],
    }


def _run_client_phases(
    args: argparse.Namespace, router_token: str, openrouter_key: str
) -> dict[str, Any]:
    key_readiness = base._probe_key_readiness(
        gateway_url="",
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    endpoint_probes = [
        _native_request(
            path="modal_static",
            case=_candidate_case(index),
            expected_worker=worker,
            router_url=args.router_url,
            router_token=router_token,
            episode_id=_episode_id(args.run_id, f"modal-native-endpoint-{worker}"),
            timeout_seconds=args.timeout_seconds,
        )
        for index, worker in enumerate(WORKERS)
    ]
    args.router_token = router_token
    selected, coverage = _coverage(args)
    natural_cases = selected * REPETITIONS
    natural = [
        _batch(
            path=path,
            cases=natural_cases,
            router_url=args.router_url,
            router_token=router_token,
            run_id=args.run_id,
            phase=f"natural-{path}",
            timeout_seconds=args.timeout_seconds,
        )
        for path in ("modal_static", "modal_arc")
    ]
    stratified_cases = _stratified_cases() * REPETITIONS
    direct = base._run_batch(
        path="direct",
        cases=stratified_cases,
        concurrency=CONCURRENCY,
        wave=0,
        gateway_url="",
        openrouter_key=openrouter_key,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    direct["phase"] = "stratified-direct"
    stratified_static = _batch(
        path="modal_static",
        cases=stratified_cases,
        router_url=args.router_url,
        router_token=router_token,
        run_id=args.run_id,
        phase="stratified-modal-static",
        timeout_seconds=args.timeout_seconds,
    )
    native_count = (
        len(endpoint_probes)
        + len(coverage)
        + sum(len(phase["results"]) for phase in [*natural, stratified_static])
    )
    direct_count = 1 + len(direct["results"])
    if (
        native_count != EXPECTED_NATIVE_REQUESTS
        or direct_count != EXPECTED_DIRECT_REQUESTS
    ):
        raise RuntimeError("native Modal benchmark request contract diverged")
    return {
        "key_readiness": key_readiness,
        "endpoint_probes": endpoint_probes,
        "coverage": coverage,
        "selected": selected,
        "natural": natural,
        "direct": direct,
        "stratified_static": stratified_static,
        "native_count": native_count,
        "direct_count": direct_count,
    }


def _client_report(args: argparse.Namespace, phases: dict[str, Any]) -> dict[str, Any]:
    natural = phases["natural"]
    direct = phases["direct"]
    stratified_static = phases["stratified_static"]
    return {
        "schema_version": "rayline.openrouter-modal-native.client.v1",
        "run_id": args.run_id,
        "status": "client_passed",
        "models": WORKERS,
        "provider_orders": PROVIDER_NAMES,
        "maximum_provider_requests": MAX_PROVIDER_REQUESTS,
        "native_requests": phases["native_count"],
        "direct_requests": phases["direct_count"],
        "key_readiness": phases["key_readiness"],
        "endpoint_probes": phases["endpoint_probes"],
        "coverage": phases["coverage"],
        "selected_cases": [
            {
                "case_id": case["case_id"],
                "scenario": case["scenario"],
                "expected_worker": case["expected_worker"],
            }
            for case in phases["selected"]
        ],
        "natural_phases": natural,
        "stratified_phases": [direct, stratified_static],
        "client_summary": {
            phase["phase"]: _client_phase_summary(phase)
            for phase in [*natural, direct, stratified_static]
        },
        "ttft_contract": (
            "native Modal currently buffers the upstream completion before its "
            "OpenAI SSE shim; observed_first_token is response-after-buffering, "
            "not provider TTFT"
        ),
    }


def main() -> None:
    args = _args()
    router_token = os.environ.get("RAYLINE_MODAL_NATIVE_ROUTER_TOKEN", "")
    openrouter_key = os.environ.get("OPENROUTER_EPHEMERAL_API_KEY", "")
    if not router_token or not openrouter_key:
        raise SystemExit("native Modal router and OpenRouter credentials are required")
    report = _client_report(
        args, _run_client_phases(args, router_token, openrouter_key)
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    for protected in (openrouter_key, router_token):
        if protected in encoded:
            raise RuntimeError("credential entered native Modal client report")
    Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
