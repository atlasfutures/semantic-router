#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded direct-vLLM versus Rayline ARC real-worker benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import threading
import time
from typing import Any, Self

from modal_encoder_warmup import warm_encoder_from_environment
from modal_fullstack_canary import (
    WORKERS,
    _cover_workers,
    _direct_baselines,
    _episode_id,
    _nonstream_chat,
    _summary,
    _validate_metrics,
)
from modal_http import connection_for_url as _connection

HTTP_OK = 200
CONCURRENCY_LEVELS = (1, 2, 4, 8)
WAVES_PER_LEVEL = 2
SOAK_CONCURRENCY = 4
SOAK_WAVES = 5
DIRECT_BASELINE_REQUESTS = 8
MAX_COVERAGE_REQUESTS = 24
LADDER_REQUESTS_PER_PATH = sum(CONCURRENCY_LEVELS) * WAVES_PER_LEVEL
SOAK_REQUESTS = SOAK_CONCURRENCY * SOAK_WAVES
BENCHMARK_REQUESTS = (2 * LADDER_REQUESTS_PER_PATH) + SOAK_REQUESTS
MAX_GENERATION_REQUESTS = (
    DIRECT_BASELINE_REQUESTS + MAX_COVERAGE_REQUESTS + BENCHMARK_REQUESTS
)
EXPECTED_REQUESTS_PER_WORKER = BENCHMARK_REQUESTS // len(WORKERS)
METRIC_SAMPLE_INTERVAL_SECONDS = 0.2
METRIC_RESET_TOLERANCE = 1e-9
PUBLIC_GATEWAY_AUTHORIZATION = "Bearer public-modal-fullstack-benchmark"

COUNTER_METRICS = {
    "request_success": "vllm:request_success_total",
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "preemptions": "vllm:num_preemptions_total",
}
HISTOGRAM_METRICS = {
    "time_to_first_token": "vllm:time_to_first_token_seconds",
    "e2e_request_latency": "vllm:e2e_request_latency_seconds",
    "request_queue_time": "vllm:request_queue_time_seconds",
}
GAUGE_METRICS = {
    "requests_running": "vllm:num_requests_running",
    "requests_waiting": "vllm:num_requests_waiting",
    "kv_cache_usage": "vllm:kv_cache_usage_perc",
}


def _metric_total(metrics: str, name: str) -> float:
    number = r"([0-9.eE+-]+)"
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+{number}$")
    return sum(
        float(match.group(1))
        for line in metrics.splitlines()
        if (match := pattern.match(line)) is not None
    )


def _worker_metric_snapshot(metrics: str) -> dict[str, float]:
    snapshot = {
        name: _metric_total(metrics, metric) for name, metric in COUNTER_METRICS.items()
    }
    for name, metric in HISTOGRAM_METRICS.items():
        snapshot[f"{name}_count"] = _metric_total(metrics, f"{metric}_count")
        snapshot[f"{name}_sum_seconds"] = _metric_total(metrics, f"{metric}_sum")
    for name, metric in GAUGE_METRICS.items():
        snapshot[name] = _metric_total(metrics, metric)
    return snapshot


def _read_worker_metrics(url: str, authorization: str, timeout_seconds: float) -> str:
    connection, prefix = _connection(url, timeout_seconds)
    try:
        connection.request(
            "GET",
            prefix or "/",
            headers={"authorization": authorization},
        )
        response = connection.getresponse()
        body = response.read().decode()
        if response.status != HTTP_OK:
            raise RuntimeError(
                f"worker metrics endpoint returned HTTP {response.status}"
            )
        return body
    finally:
        connection.close()


def _metric_snapshots(
    *, worker_urls: dict[str, str], authorization: str, timeout_seconds: float
) -> dict[str, dict[str, float]]:
    return {
        worker: _worker_metric_snapshot(
            _read_worker_metrics(
                f"{worker_urls[worker].rstrip('/')}/metrics",
                authorization,
                timeout_seconds,
            )
        )
        for worker in WORKERS
    }


class _WorkerMetricSampler:
    def __init__(
        self,
        *,
        worker_urls: dict[str, str],
        authorization: str,
        timeout_seconds: float,
    ) -> None:
        self.worker_urls = worker_urls
        self.authorization = authorization
        self.timeout_seconds = timeout_seconds
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples = dict.fromkeys(WORKERS, 0)
        self.errors = dict.fromkeys(WORKERS, 0)
        self.maximum = {worker: dict.fromkeys(GAUGE_METRICS, 0.0) for worker in WORKERS}

    def _run(self) -> None:
        while not self.stop.is_set():
            for worker in WORKERS:
                try:
                    metrics = _read_worker_metrics(
                        f"{self.worker_urls[worker].rstrip('/')}/metrics",
                        self.authorization,
                        self.timeout_seconds,
                    )
                    snapshot = _worker_metric_snapshot(metrics)
                    self.samples[worker] += 1
                    for name in GAUGE_METRICS:
                        self.maximum[worker][name] = max(
                            self.maximum[worker][name], snapshot[name]
                        )
                except (OSError, RuntimeError, ValueError):
                    self.errors[worker] += 1
            self.stop.wait(METRIC_SAMPLE_INTERVAL_SECONDS)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop.set()
        self.thread.join(timeout=max(5.0, self.timeout_seconds))
        if self.thread.is_alive():
            raise RuntimeError("worker metric sampler did not stop")

    def report(self) -> dict[str, dict[str, Any]]:
        for worker in WORKERS:
            if self.samples[worker] == 0:
                raise RuntimeError(f"worker metric sampler missed {worker}")
        return {
            worker: {
                "samples": self.samples[worker],
                "scrape_errors": self.errors[worker],
                "maximum": self.maximum[worker],
            }
            for worker in WORKERS
        }


def _balanced_targets(concurrency: int, wave: int) -> list[str]:
    workers = tuple(WORKERS)
    return [workers[(index + wave) % len(workers)] for index in range(concurrency)]


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
    if path == "direct":
        result = _nonstream_chat(
            base_url=worker_urls[target_worker],
            model=WORKERS[target_worker],
            prompt=selected_prompts[target_worker],
            authorization=worker_authorization,
            timeout_seconds=timeout_seconds,
        )
    elif path == "arc":
        result = _nonstream_chat(
            base_url=gateway_url,
            model="auto",
            prompt=selected_prompts[target_worker],
            authorization=PUBLIC_GATEWAY_AUTHORIZATION,
            timeout_seconds=timeout_seconds,
            episode_id=_episode_id(run_id, f"benchmark-{phase_label}-{request_index}"),
        )
        if result["selected_worker"] != target_worker:
            raise RuntimeError("ARC selection changed for a frozen benchmark prompt")
    else:
        raise ValueError(f"unsupported benchmark path: {path}")
    if result["response_model"] != WORKERS[target_worker]:
        raise RuntimeError("worker response model did not match the intended worker")
    return {**result, "selected_worker": target_worker}


def _run_wave(
    *,
    path: str,
    concurrency: int,
    wave: int,
    gateway_url: str,
    worker_urls: dict[str, str],
    selected_prompts: dict[str, str],
    worker_authorization: str,
    run_id: str,
    phase_label: str,
    timeout_seconds: float,
) -> dict[str, Any]:
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
    latencies = [float(result["latency_seconds"]) for result in results]
    return {
        "path": path,
        "concurrency": concurrency,
        "wave": wave,
        "requests": len(results),
        "wall_seconds": wall_seconds,
        "requests_per_second": len(results) / wall_seconds,
        "wall_to_sum_latency_ratio": wall_seconds / sum(latencies),
        "latency": _summary(latencies),
        "completion_tokens": sum(
            int(result["completion_tokens"]) for result in results
        ),
        "selection_counts": {
            worker: sum(result["selected_worker"] == worker for result in results)
            for worker in WORKERS
        },
    }


def _run_ladder(
    *,
    gateway_url: str,
    worker_urls: dict[str, str],
    selected_prompts: dict[str, str],
    worker_authorization: str,
    run_id: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    phases: list[dict[str, Any]] = []
    for concurrency in CONCURRENCY_LEVELS:
        for path in ("direct", "arc"):
            for wave in range(WAVES_PER_LEVEL):
                label = f"ladder-{path}-c{concurrency}-w{wave}"
                print(f"benchmark {label}: starting", file=sys.stderr, flush=True)
                phases.append(
                    _run_wave(
                        path=path,
                        concurrency=concurrency,
                        wave=wave,
                        gateway_url=gateway_url,
                        worker_urls=worker_urls,
                        selected_prompts=selected_prompts,
                        worker_authorization=worker_authorization,
                        run_id=run_id,
                        phase_label=label,
                        timeout_seconds=timeout_seconds,
                    )
                )
    return phases


def _run_soak(
    *,
    gateway_url: str,
    worker_urls: dict[str, str],
    selected_prompts: dict[str, str],
    worker_authorization: str,
    run_id: str,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    phases = []
    for wave in range(SOAK_WAVES):
        label = f"soak-arc-c{SOAK_CONCURRENCY}-w{wave}"
        print(f"benchmark {label}: starting", file=sys.stderr, flush=True)
        phases.append(
            _run_wave(
                path="arc",
                concurrency=SOAK_CONCURRENCY,
                wave=wave,
                gateway_url=gateway_url,
                worker_urls=worker_urls,
                selected_prompts=selected_prompts,
                worker_authorization=worker_authorization,
                run_id=run_id,
                phase_label=label,
                timeout_seconds=timeout_seconds,
            )
        )
    return phases


def _metric_deltas(
    before: dict[str, dict[str, float]], after: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    deltas: dict[str, dict[str, float]] = {}
    delta_names = [
        *COUNTER_METRICS,
        *(f"{name}_count" for name in HISTOGRAM_METRICS),
        *(f"{name}_sum_seconds" for name in HISTOGRAM_METRICS),
    ]
    for worker in WORKERS:
        deltas[worker] = {
            name: after[worker][name] - before[worker][name] for name in delta_names
        }
        if any(value < -METRIC_RESET_TOLERANCE for value in deltas[worker].values()):
            raise RuntimeError(f"vLLM metrics reset during the benchmark on {worker}")
    return deltas


def _validate_worker_metric_deltas(
    deltas: dict[str, dict[str, float]],
) -> None:
    for worker, values in deltas.items():
        if not math.isclose(
            values["request_success"], EXPECTED_REQUESTS_PER_WORKER, abs_tol=1e-9
        ):
            raise RuntimeError(f"vLLM success counter mismatch on {worker}")
        if values["preemptions"] != 0:
            raise RuntimeError(f"vLLM preempted a benchmark request on {worker}")
        if values["prompt_tokens"] <= 0 or values["generation_tokens"] <= 0:
            raise RuntimeError(f"vLLM token counters did not advance on {worker}")
        for name in HISTOGRAM_METRICS:
            if not math.isclose(
                values[f"{name}_count"],
                EXPECTED_REQUESTS_PER_WORKER,
                abs_tol=1e-9,
            ):
                raise RuntimeError(f"vLLM {name} histogram mismatch on {worker}")


def _aggregate_phases(phases: list[dict[str, Any]]) -> dict[str, Any]:
    total_requests = sum(int(phase["requests"]) for phase in phases)
    total_wall = sum(float(phase["wall_seconds"]) for phase in phases)
    return {
        "requests": total_requests,
        "sum_wave_wall_seconds": total_wall,
        "requests_per_sum_wave_second": total_requests / total_wall,
        "completion_tokens": sum(int(phase["completion_tokens"]) for phase in phases),
        "maximum_wave_p95_seconds": max(
            float(phase["latency"]["p95_seconds"]) for phase in phases
        ),
        "maximum_request_seconds": max(
            float(phase["latency"]["max_seconds"]) for phase in phases
        ),
        "selection_counts": {
            worker: sum(int(phase["selection_counts"][worker]) for phase in phases)
            for worker in WORKERS
        },
    }


def _ladder_comparison(phases: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    comparison = []
    for concurrency in CONCURRENCY_LEVELS:
        path_summaries = {
            path: _aggregate_phases(
                [
                    phase
                    for phase in phases
                    if phase["path"] == path
                    and int(phase["concurrency"]) == concurrency
                ]
            )
            for path in ("direct", "arc")
        }
        direct = path_summaries["direct"]
        arc = path_summaries["arc"]
        direct_rps = float(direct["requests_per_sum_wave_second"])
        arc_rps = float(arc["requests_per_sum_wave_second"])
        direct_p95 = float(direct["maximum_wave_p95_seconds"])
        arc_p95 = float(arc["maximum_wave_p95_seconds"])
        comparison.append(
            {
                "concurrency": concurrency,
                "requests_per_path": int(direct["requests"]),
                "direct_requests_per_second": direct_rps,
                "arc_requests_per_second": arc_rps,
                "arc_to_direct_throughput_ratio": arc_rps / direct_rps,
                "direct_maximum_wave_p95_seconds": direct_p95,
                "arc_maximum_wave_p95_seconds": arc_p95,
                "arc_p95_overhead_seconds": arc_p95 - direct_p95,
                "arc_to_direct_p95_ratio": arc_p95 / direct_p95,
            }
        )
    return comparison


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
    coverage_results: list[dict[str, Any]],
    encoder_warmup: dict[str, Any],
    ladder: list[dict[str, Any]],
    soak: list[dict[str, Any]],
    metric_deltas: dict[str, dict[str, float]],
    sampled_metrics: dict[str, dict[str, Any]],
    session_actions: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema_version": "rayline.arc.modal-fullstack-benchmark.v1",
        "run_id": args.run_id,
        "status": "passed",
        "workload": {
            "concurrency_levels": CONCURRENCY_LEVELS,
            "waves_per_level": WAVES_PER_LEVEL,
            "soak_concurrency": SOAK_CONCURRENCY,
            "soak_waves": SOAK_WAVES,
            "benchmark_requests": BENCHMARK_REQUESTS,
            "actual_generation_requests": (
                DIRECT_BASELINE_REQUESTS + len(coverage_results) + BENCHMARK_REQUESTS
            ),
            "maximum_generation_requests": MAX_GENERATION_REQUESTS,
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
        "ladder": ladder,
        "ladder_aggregate": _aggregate_phases(ladder),
        "ladder_comparison": _ladder_comparison(ladder),
        "soak": soak,
        "soak_aggregate": _aggregate_phases(soak),
        "worker_metric_deltas": metric_deltas,
        "worker_metric_samples": sampled_metrics,
        "session_actions_created": session_actions["created"],
        "selection_failures": 0,
        "provider_calls": 0,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
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
    print("benchmark encoder warmup: starting", file=sys.stderr, flush=True)
    encoder_warmup = warm_encoder_from_environment(
        timeout_seconds=args.timeout_seconds,
        connection_factory=_connection,
    )
    selected_prompts, coverage_results = _cover_workers(
        gateway_url=args.gateway_url,
        run_id=args.run_id,
        timeout_seconds=args.timeout_seconds,
    )
    before = _metric_snapshots(
        worker_urls=worker_urls,
        authorization=worker_authorization,
        timeout_seconds=args.timeout_seconds,
    )
    with _WorkerMetricSampler(
        worker_urls=worker_urls,
        authorization=worker_authorization,
        timeout_seconds=args.timeout_seconds,
    ) as sampler:
        ladder = _run_ladder(
            gateway_url=args.gateway_url,
            worker_urls=worker_urls,
            selected_prompts=selected_prompts,
            worker_authorization=worker_authorization,
            run_id=args.run_id,
            timeout_seconds=args.timeout_seconds,
        )
        soak = _run_soak(
            gateway_url=args.gateway_url,
            worker_urls=worker_urls,
            selected_prompts=selected_prompts,
            worker_authorization=worker_authorization,
            run_id=args.run_id,
            timeout_seconds=args.timeout_seconds,
        )
    sampled_metrics = sampler.report()
    after = _metric_snapshots(
        worker_urls=worker_urls,
        authorization=worker_authorization,
        timeout_seconds=args.timeout_seconds,
    )
    metric_deltas = _metric_deltas(before, after)
    _validate_worker_metric_deltas(metric_deltas)

    expected_creates = len(coverage_results) + LADDER_REQUESTS_PER_PATH + SOAK_REQUESTS
    session_actions = _validate_metrics(
        metrics_url=args.metrics_url,
        timeout_seconds=args.timeout_seconds,
        expected_creates=expected_creates,
    )
    report = _build_report(
        args=args,
        direct_baselines=direct_baselines,
        coverage_results=coverage_results,
        encoder_warmup=encoder_warmup,
        ladder=ladder,
        soak=soak,
        metric_deltas=metric_deltas,
        sampled_metrics=sampled_metrics,
        session_actions=session_actions,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
