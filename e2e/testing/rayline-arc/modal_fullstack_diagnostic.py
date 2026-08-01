#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded direct/static-gateway/Rayline ARC latency diagnostic."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import sys
import time
from typing import Any

from modal_encoder_warmup import warm_encoder_from_environment
from modal_fullstack_benchmark import (
    _balanced_targets,
    _metric_deltas,
    _metric_snapshots,
    _WorkerMetricSampler,
)
from modal_fullstack_canary import (
    WORKERS,
    _cover_workers,
    _direct_baselines,
    _episode_id,
    _metric_values,
    _nonstream_chat,
    _read_metrics,
    _summary,
    _validate_metrics,
)
from modal_fullstack_diagnostic_metrics import (
    ROUTER_HISTOGRAMS,
    _histogram_report,
    _merge_histograms,
    _merge_worker_deltas,
    _router_metric_deltas,
    _router_metric_snapshot,
    _validate_phase_metrics,
    _worker_metric_report,
)
from modal_http import connection_for_url as _connection

PATHS = ("direct", "gateway_static", "arc")
CONCURRENCY_LEVELS = (1, 4)
WAVES_PER_LEVEL = 2
DIRECT_BASELINE_REQUESTS = 8
MAX_COVERAGE_REQUESTS = 24
REQUESTS_PER_PATH = sum(CONCURRENCY_LEVELS) * WAVES_PER_LEVEL
MEASURED_GENERATION_REQUESTS = len(PATHS) * REQUESTS_PER_PATH
MAX_GENERATION_REQUESTS = (
    DIRECT_BASELINE_REQUESTS + MAX_COVERAGE_REQUESTS + MEASURED_GENERATION_REQUESTS
)
EXPECTED_REQUESTS_PER_WORKER = MEASURED_GENERATION_REQUESTS // len(WORKERS)
PUBLIC_GATEWAY_AUTHORIZATION = "Bearer public-modal-fullstack-diagnostic"
SESSION_METRIC = "llm_rayline_arc_session_actions_total"
FIXED_SEED = 20260801
FIXED_MAX_TOKENS = 32
MAX_RESOURCE_ENVELOPE_USD = 3.0574728
CUMULATIVE_BEFORE_USD = 11.17499122

EXECUTION_FIELDS = {
    "worker-a": {
        "temperature": 0.2,
        "extra_fields": {
            "seed": FIXED_SEED,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
    "worker-b": {
        "temperature": 0.3,
        "extra_fields": {
            "seed": FIXED_SEED,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    },
}


def _one_request(
    *,
    path: str,
    target_worker: str,
    request_index: int,
    gateway_url: str,
    worker_urls: dict[str, str],
    selected_prompts: dict[str, str],
    worker_authorization: str,
    run_id: str,
    phase_label: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    execution = EXECUTION_FIELDS[target_worker]
    common = {
        "prompt": selected_prompts[target_worker],
        "timeout_seconds": timeout_seconds,
        "max_tokens": FIXED_MAX_TOKENS,
        "temperature": execution["temperature"],
        "extra_fields": execution["extra_fields"],
    }
    if path == "direct":
        result = _nonstream_chat(
            base_url=worker_urls[target_worker],
            model=WORKERS[target_worker],
            authorization=worker_authorization,
            **common,
        )
    elif path == "gateway_static":
        result = _nonstream_chat(
            base_url=gateway_url,
            model=target_worker,
            authorization=PUBLIC_GATEWAY_AUTHORIZATION,
            **common,
        )
    elif path == "arc":
        result = _nonstream_chat(
            base_url=gateway_url,
            model="auto",
            authorization=PUBLIC_GATEWAY_AUTHORIZATION,
            episode_id=_episode_id(
                run_id,
                f"diagnostic-{phase_label}-{request_index}",
            ),
            **common,
        )
        if result["selected_worker"] != target_worker:
            raise RuntimeError("ARC selection changed for a frozen diagnostic prompt")
    else:
        raise ValueError(f"unsupported diagnostic path: {path}")

    if result["response_model"] != WORKERS[target_worker]:
        raise RuntimeError("worker response model did not match diagnostic target")
    if path != "direct":
        if result["envoy_upstream_service_seconds"] is None:
            raise RuntimeError("gateway response omitted Envoy upstream service time")
        if result["envoy_attempt_count"] != 1:
            raise RuntimeError("self-hosted diagnostic observed an upstream retry")
    return {**result, "selected_worker": target_worker}


def _run_wave(
    *,
    path: str,
    concurrency: int,
    wave: int,
    gateway_url: str,
    metrics_url: str,
    worker_urls: dict[str, str],
    selected_prompts: dict[str, str],
    worker_authorization: str,
    run_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    phase_label = f"{path}-c{concurrency}-w{wave}"
    print(f"diagnostic {phase_label}: starting", file=sys.stderr, flush=True)
    worker_before = _metric_snapshots(
        worker_urls=worker_urls,
        authorization=worker_authorization,
        timeout_seconds=timeout_seconds,
    )
    router_before = _router_metric_snapshot(_read_metrics(metrics_url, timeout_seconds))
    targets = _balanced_targets(concurrency, wave)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _one_request,
                path=path,
                target_worker=target,
                request_index=index,
                gateway_url=gateway_url,
                worker_urls=worker_urls,
                selected_prompts=selected_prompts,
                worker_authorization=worker_authorization,
                run_id=run_id,
                phase_label=phase_label,
                timeout_seconds=timeout_seconds,
            )
            for index, target in enumerate(targets)
        ]
        results = [future.result() for future in futures]
    wall_seconds = time.perf_counter() - started
    router_after = _router_metric_snapshot(_read_metrics(metrics_url, timeout_seconds))
    worker_after = _metric_snapshots(
        worker_urls=worker_urls,
        authorization=worker_authorization,
        timeout_seconds=timeout_seconds,
    )
    worker_deltas = _metric_deltas(worker_before, worker_after)
    router_deltas = _router_metric_deltas(router_before, router_after)
    _validate_phase_metrics(
        path=path,
        requests=len(results),
        worker_deltas=worker_deltas,
        router_deltas=router_deltas,
    )

    latencies = [float(result["latency_seconds"]) for result in results]
    upstream = (
        [float(result["envoy_upstream_service_seconds"]) for result in results]
        if path != "direct"
        else []
    )
    residual = (
        [
            max(0.0, float(result["latency_seconds"]) - upstream_seconds)
            for result, upstream_seconds in zip(results, upstream, strict=True)
        ]
        if path != "direct"
        else []
    )
    return {
        "path": path,
        "concurrency": concurrency,
        "wave": wave,
        "requests": len(results),
        "wall_seconds": wall_seconds,
        "latencies": latencies,
        "upstream_service": upstream,
        "gateway_residual": residual,
        "completion_tokens": {
            worker: sum(
                int(result["completion_tokens"])
                for result in results
                if result["selected_worker"] == worker
            )
            for worker in WORKERS
        },
        "selection_counts": {
            worker: sum(result["selected_worker"] == worker for result in results)
            for worker in WORKERS
        },
        "worker_metric_deltas": worker_deltas,
        "router_metric_deltas": router_deltas,
    }


def _merge_router_deltas(phases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        label: _merge_histograms(
            [phase["router_metric_deltas"][label] for phase in phases]
        )
        for label in ROUTER_HISTOGRAMS
    }


def _phase_report(phase: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": phase["path"],
        "concurrency": phase["concurrency"],
        "wave": phase["wave"],
        "requests": phase["requests"],
        "wall_seconds": phase["wall_seconds"],
        "requests_per_second": phase["requests"] / phase["wall_seconds"],
        "client_latency": _summary(phase["latencies"]),
        "envoy_upstream_service_time": (
            _summary(phase["upstream_service"]) if phase["upstream_service"] else None
        ),
        "gateway_residual": (
            _summary(phase["gateway_residual"]) if phase["gateway_residual"] else None
        ),
        "completion_tokens": phase["completion_tokens"],
        "selection_counts": phase["selection_counts"],
        "worker_metrics": _worker_metric_report(phase["worker_metric_deltas"]),
        "router_metrics": {
            label: _histogram_report(value)
            for label, value in phase["router_metric_deltas"].items()
        },
    }


def _aggregate_phases(phases: list[dict[str, Any]]) -> dict[str, Any]:
    worker_deltas = _merge_worker_deltas(
        [phase["worker_metric_deltas"] for phase in phases]
    )
    router_deltas = _merge_router_deltas(phases)
    requests = sum(int(phase["requests"]) for phase in phases)
    wall_seconds = sum(float(phase["wall_seconds"]) for phase in phases)
    latencies = [value for phase in phases for value in phase["latencies"]]
    upstream = [value for phase in phases for value in phase["upstream_service"]]
    residual = [value for phase in phases for value in phase["gateway_residual"]]
    return {
        "path": phases[0]["path"],
        "concurrency": phases[0]["concurrency"],
        "requests": requests,
        "sum_wave_wall_seconds": wall_seconds,
        "requests_per_sum_wave_second": requests / wall_seconds,
        "client_latency": _summary(latencies),
        "envoy_upstream_service_time": _summary(upstream) if upstream else None,
        "gateway_residual": _summary(residual) if residual else None,
        "completion_tokens": {
            worker: sum(int(phase["completion_tokens"][worker]) for phase in phases)
            for worker in WORKERS
        },
        "selection_counts": {
            worker: sum(int(phase["selection_counts"][worker]) for phase in phases)
            for worker in WORKERS
        },
        "worker_metrics": _worker_metric_report(worker_deltas),
        "router_metrics": {
            label: _histogram_report(value) for label, value in router_deltas.items()
        },
    }


def _aggregates(phases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _aggregate_phases(
            [
                phase
                for phase in phases
                if phase["path"] == path and int(phase["concurrency"]) == concurrency
            ]
        )
        for concurrency in CONCURRENCY_LEVELS
        for path in PATHS
    ]


def _comparison(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for concurrency in CONCURRENCY_LEVELS:
        by_path = {
            item["path"]: item
            for item in aggregates
            if int(item["concurrency"]) == concurrency
        }
        direct = by_path["direct"]
        static = by_path["gateway_static"]
        arc = by_path["arc"]
        direct_rps = float(direct["requests_per_sum_wave_second"])
        static_rps = float(static["requests_per_sum_wave_second"])
        arc_rps = float(arc["requests_per_sum_wave_second"])
        direct_p95 = float(direct["client_latency"]["p95_seconds"])
        static_p95 = float(static["client_latency"]["p95_seconds"])
        arc_p95 = float(arc["client_latency"]["p95_seconds"])
        rows.append(
            {
                "concurrency": concurrency,
                "requests_per_path": direct["requests"],
                "gateway_static_to_direct_throughput_ratio": static_rps / direct_rps,
                "arc_to_gateway_static_throughput_ratio": arc_rps / static_rps,
                "arc_to_direct_throughput_ratio": arc_rps / direct_rps,
                "gateway_static_p95_over_direct_seconds": static_p95 - direct_p95,
                "arc_p95_over_gateway_static_seconds": arc_p95 - static_p95,
                "arc_p95_over_direct_seconds": arc_p95 - direct_p95,
                "direct_worker_e2e_mean_seconds": direct["worker_metrics"][
                    "e2e_request_latency"
                ]["mean_seconds"],
                "gateway_static_router_mean_seconds": static["router_metrics"][
                    "routing"
                ]["mean_seconds"],
                "arc_router_mean_seconds": arc["router_metrics"]["routing"][
                    "mean_seconds"
                ],
                "arc_encoder_mean_seconds": arc["router_metrics"]["encoder"][
                    "mean_seconds"
                ],
            }
        )
    return rows


def _validate_packet(
    phases: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
) -> None:
    if sum(int(phase["requests"]) for phase in phases) != MEASURED_GENERATION_REQUESTS:
        raise RuntimeError("diagnostic measured request count changed")
    for worker in WORKERS:
        by_path = {
            path: sum(
                int(item["completion_tokens"][worker])
                for item in aggregates
                if item["path"] == path
            )
            for path in PATHS
        }
        if len(set(by_path.values())) != 1:
            raise RuntimeError(
                f"completion-token parity failed for {worker}: {by_path}"
            )
    worker_deltas = _merge_worker_deltas(
        [phase["worker_metric_deltas"] for phase in phases]
    )
    for worker in WORKERS:
        if not math.isclose(
            worker_deltas[worker]["request_success"],
            EXPECTED_REQUESTS_PER_WORKER,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"measured worker balance changed on {worker}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--worker-a-url", required=True)
    parser.add_argument("--worker-b-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser.parse_args()


def _build_report(
    *,
    args: argparse.Namespace,
    direct_baselines: dict[str, list[dict[str, Any]]],
    encoder_warmup: dict[str, Any],
    coverage_results: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    aggregates: list[dict[str, Any]],
    sampled_metrics: dict[str, dict[str, Any]],
    session_delta: dict[str, float],
    session_actions: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": "rayline.arc.modal-fullstack-diagnostic.v1",
        "run_id": args.run_id,
        "status": "passed",
        "workload": {
            "paths": PATHS,
            "concurrency_levels": CONCURRENCY_LEVELS,
            "waves_per_level": WAVES_PER_LEVEL,
            "fixed_seed": FIXED_SEED,
            "fixed_max_tokens": FIXED_MAX_TOKENS,
            "measured_generation_requests": MEASURED_GENERATION_REQUESTS,
            "actual_generation_requests": (
                DIRECT_BASELINE_REQUESTS
                + len(coverage_results)
                + MEASURED_GENERATION_REQUESTS
            ),
            "maximum_generation_requests": MAX_GENERATION_REQUESTS,
            "measured_requests_per_worker": EXPECTED_REQUESTS_PER_WORKER,
        },
        "budget": {
            "maximum_resource_envelope_usd": MAX_RESOURCE_ENVELOPE_USD,
            "cumulative_before_usd": CUMULATIVE_BEFORE_USD,
            "cumulative_if_full_envelope_usd": (
                CUMULATIVE_BEFORE_USD + MAX_RESOURCE_ENVELOPE_USD
            ),
            "provider_spend_usd": 0,
        },
        "encoder_warmup": encoder_warmup,
        "direct_baseline": {
            worker: {
                "samples": len(results),
                "latency": _summary(
                    [float(result["latency_seconds"]) for result in results]
                ),
            }
            for worker, results in direct_baselines.items()
        },
        "coverage_requests": len(coverage_results),
        "phases": [_phase_report(phase) for phase in phases],
        "aggregates": aggregates,
        "comparison": _comparison(aggregates),
        "worker_metric_samples": sampled_metrics,
        "measured_session_actions": session_delta,
        "session_actions_created_total": session_actions["created"],
        "selection_failures": 0,
        "provider_calls": 0,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
        "interpretation_boundary": (
            "gateway_residual is client elapsed minus Envoy's upstream-service-time "
            "header; it is an aggregate residual estimate, not an exact span"
        ),
    }


def main() -> None:
    args = _parse_args()
    worker_api_key = os.environ.get("RAYLINE_ARC_WORKER_API_KEY", "")
    if not worker_api_key:
        raise SystemExit("RAYLINE_ARC_WORKER_API_KEY is required")
    worker_authorization = f"Bearer {worker_api_key}"
    worker_urls = {
        "worker-a": args.worker_a_url,
        "worker-b": args.worker_b_url,
    }

    direct_baselines = _direct_baselines(
        worker_urls=worker_urls,
        authorization=worker_authorization,
        timeout_seconds=args.timeout_seconds,
    )
    print("diagnostic encoder warmup: starting", file=sys.stderr, flush=True)
    encoder_warmup = warm_encoder_from_environment(
        timeout_seconds=args.timeout_seconds,
        connection_factory=_connection,
    )
    selected_prompts, coverage_results = _cover_workers(
        gateway_url=args.gateway_url,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    session_before = _metric_values(
        _read_metrics(args.metrics_url, args.timeout_seconds),
        SESSION_METRIC,
    )
    phases: list[dict[str, Any]] = []
    with _WorkerMetricSampler(
        worker_urls=worker_urls,
        authorization=worker_authorization,
        timeout_seconds=args.timeout_seconds,
    ) as sampler:
        for concurrency in CONCURRENCY_LEVELS:
            for wave in range(WAVES_PER_LEVEL):
                path_order = PATHS if wave % 2 == 0 else tuple(reversed(PATHS))
                for path in path_order:
                    phases.append(
                        _run_wave(
                            path=path,
                            concurrency=concurrency,
                            wave=wave,
                            gateway_url=args.gateway_url,
                            metrics_url=args.metrics_url,
                            worker_urls=worker_urls,
                            selected_prompts=selected_prompts,
                            worker_authorization=worker_authorization,
                            run_id=args.run_id,
                            timeout_seconds=args.timeout_seconds,
                        )
                    )
    sampled_metrics = sampler.report()
    aggregates = _aggregates(phases)
    _validate_packet(phases, aggregates)

    session_after = _metric_values(
        _read_metrics(args.metrics_url, args.timeout_seconds),
        SESSION_METRIC,
    )
    session_delta = {
        action: session_after.get(action, 0.0) - session_before.get(action, 0.0)
        for action in set(session_before) | set(session_after)
    }
    if session_delta.get("created", 0.0) != REQUESTS_PER_PATH or any(
        value != 0 for action, value in session_delta.items() if action != "created"
    ):
        raise RuntimeError("measured ARC session actions changed")
    session_actions = _validate_metrics(
        metrics_url=args.metrics_url,
        timeout_seconds=args.timeout_seconds,
        expected_creates=len(coverage_results) + REQUESTS_PER_PATH,
    )

    report = _build_report(
        args=args,
        direct_baselines=direct_baselines,
        encoder_warmup=encoder_warmup,
        coverage_results=coverage_results,
        phases=phases,
        aggregates=aggregates,
        sampled_metrics=sampled_metrics,
        session_delta=session_delta,
        session_actions=session_actions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
