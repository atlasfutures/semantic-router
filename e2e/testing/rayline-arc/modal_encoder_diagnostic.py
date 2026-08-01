#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Bounded protected-encoder concurrency and queue diagnostic.

The packet calls only the retained Rayline encoder. Inputs are fixed public
synthetic turns, and the report contains aggregate timings and counters only.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import threading
import time
from dataclasses import replace
from typing import Any

from modal_fullstack_inputs import CANDIDATE_PROMPTS
from modal_session_canary import CanaryClient, _episode_hash

CONCURRENCY_LEVELS = (1, 2, 4, 8)
WAVES_PER_LEVEL = 2
PHASES = ("create", "append")
MEASURED_REQUESTS = len(PHASES) * sum(CONCURRENCY_LEVELS) * WAVES_PER_LEVEL
APPEND_SETUP_REQUESTS = sum(CONCURRENCY_LEVELS) * WAVES_PER_LEVEL
SAME_EPISODE_CONTROL_REQUESTS = 2
MIN_ENGINE_CONCURRENT_REQUESTS = 2
MAX_POOLING_REQUESTS = (
    MEASURED_REQUESTS + APPEND_SETUP_REQUESTS + SAME_EPISODE_CONTROL_REQUESTS
)
METRICS_SCHEMA = "rayline.arc.session-metrics-response.v1"
REPORT_SCHEMA = "rayline.arc.modal-encoder-diagnostic.v1"
SAMPLER_INTERVAL_SECONDS = 0.02
ENGINE_METRIC_SETTLE_TIMEOUT_SECONDS = 10.0
MAX_RESOURCE_ENVELOPE_USD = 2.4996168
CUMULATIVE_BEFORE_USD = 14.23246402

FIRST_TURNS = [{"role": "user", "text": CANDIDATE_PROMPTS[0]}]
APPENDED_TURNS = [
    *FIRST_TURNS,
    {"role": "assistant", "text": "amber"},
    {"role": "user", "text": "Reply again with one word: amber."},
]
SAME_EPISODE_TURNS = [
    {
        "role": "user",
        "text": ("Public synthetic same-session contention evidence. " * 256),
    }
]

COORDINATOR_CUMULATIVE_FIELDS = (
    "tokenization_calls_total",
    "tokenization_seconds_total",
    "requests_started_total",
    "requests_succeeded_total",
    "requests_failed_total",
    "request_seconds_total",
    "session_lock_contentions_total",
    "session_lock_wait_seconds_total",
    "backend_calls_started_total",
    "backend_calls_succeeded_total",
    "backend_calls_failed_total",
    "backend_seconds_total",
    "backend_appended_tokens_total",
)
ENGINE_CUMULATIVE_FIELDS = (
    "queue_time_observations",
    "queue_time_seconds_total",
    "inference_time_observations",
    "inference_time_seconds_total",
    "e2e_time_observations",
    "e2e_time_seconds_total",
    "prompt_token_observations",
    "prompt_tokens_total",
)
SAMPLED_FIELDS = {
    "coordinator_requests_inflight": ("coordinator", "requests_inflight"),
    "coordinator_session_lock_waiters": (
        "coordinator",
        "session_lock_waiters",
    ),
    "coordinator_backend_inflight": ("coordinator", "backend_inflight"),
    "engine_requests_running": ("engine", "requests_running"),
    "engine_requests_waiting": ("engine", "requests_waiting"),
}


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise RuntimeError("encoder diagnostic latency set is empty")
    return {
        "p50_seconds": _percentile(values, 0.50),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": max(values),
        "mean_seconds": math.fsum(values) / len(values),
    }


def _read_metrics(client: CanaryClient) -> dict[str, Any]:
    metrics, _elapsed = client.request("GET", "/v1/rayline/arc/session/metrics")
    if metrics.get("schema_version") != METRICS_SCHEMA:
        raise RuntimeError("retained-session metrics schema diverged")
    coordinator = metrics.get("coordinator")
    engine = metrics.get("engine")
    if not isinstance(coordinator, dict) or not isinstance(engine, dict):
        raise TypeError("retained-session metrics omitted stage objects")
    if engine.get("available") is not True:
        raise RuntimeError("protected vLLM engine metrics are unavailable")
    return metrics


def _metric_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    section: str,
    fields: tuple[str, ...],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in fields:
        left = before[section].get(field)
        right = after[section].get(field)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise TypeError(f"non-numeric encoder metric: {section}.{field}")
        delta = float(right) - float(left)
        if delta < 0:
            raise RuntimeError(f"encoder metric moved backwards: {section}.{field}")
        result[field] = delta
    return result


def _read_completed_metrics(
    client: CanaryClient,
    before: dict[str, Any],
    expected_observations: int,
) -> dict[str, Any]:
    """Wait briefly for vLLM's completed-request logger to settle after close."""
    deadline = time.monotonic() + ENGINE_METRIC_SETTLE_TIMEOUT_SECONDS
    observed: dict[str, float] = {}
    while True:
        after = _read_metrics(client)
        observed = _metric_delta(
            before,
            after,
            "engine",
            ENGINE_CUMULATIVE_FIELDS,
        )
        completed = tuple(
            observed[field]
            for field in (
                "queue_time_observations",
                "inference_time_observations",
                "e2e_time_observations",
                "prompt_token_observations",
            )
        )
        if all(value == expected_observations for value in completed):
            return after
        if any(value > expected_observations for value in completed):
            raise RuntimeError("vLLM completed-request metric exceeded packet count")
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "vLLM completed-request metrics did not settle: "
                f"observed={completed}, expected={expected_observations}"
            )
        time.sleep(SAMPLER_INTERVAL_SECONDS)


def _merge_deltas(deltas: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(delta.keys() for delta in deltas))
    return {
        key: math.fsum(delta.get(key, 0.0) for delta in deltas) for key in sorted(keys)
    }


class _MetricsSampler:
    def __init__(self, client: CanaryClient) -> None:
        self._client = replace(client, timeout_seconds=min(2.0, client.timeout_seconds))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.samples = 0
        self.errors = 0
        self.maximum = dict.fromkeys(SAMPLED_FIELDS, 0.0)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("encoder metric sampler did not stop")
        if self.samples < 1 or self.errors:
            raise RuntimeError("encoder metric sampler did not remain healthy")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                metrics = _read_metrics(self._client)
                for field, (section, metric) in SAMPLED_FIELDS.items():
                    value = metrics[section].get(metric)
                    if isinstance(value, (int, float)):
                        self.maximum[field] = max(self.maximum[field], float(value))
                self.samples += 1
            except Exception:
                self.errors += 1
            self._stop.wait(SAMPLER_INTERVAL_SECONDS)

    def report(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "errors": self.errors,
            "maximum": dict(self.maximum),
        }


def _run_concurrent(
    client: CanaryClient,
    episode_ids: list[str],
    turns: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], float]:
    barrier = threading.Barrier(len(episode_ids))

    def encode(episode_id: str) -> dict[str, Any]:
        barrier.wait(timeout=5)
        return client.encode(episode_id, turns)

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(episode_ids)
    ) as executor:
        futures = [executor.submit(encode, episode_id) for episode_id in episode_ids]
        results = [future.result() for future in futures]
    return results, time.perf_counter() - started


def _close_all(client: CanaryClient, episode_ids: list[str]) -> None:
    for episode_id in episode_ids:
        client.close(episode_id)


def _validate_phase_deltas(
    *,
    coordinator: dict[str, float],
    engine: dict[str, float],
    requests: int,
) -> None:
    exact_coordinator = (
        "tokenization_calls_total",
        "requests_started_total",
        "requests_succeeded_total",
        "backend_calls_started_total",
        "backend_calls_succeeded_total",
    )
    for field in exact_coordinator:
        if coordinator[field] != requests:
            raise RuntimeError(f"encoder coordinator count diverged: {field}")
    if coordinator["requests_failed_total"] != 0:
        raise RuntimeError("encoder diagnostic observed request failures")
    if coordinator["backend_calls_failed_total"] != 0:
        raise RuntimeError("encoder diagnostic observed backend failures")
    if coordinator["session_lock_contentions_total"] != 0:
        raise RuntimeError("cross-episode diagnostic hit same-session serialization")
    if coordinator["backend_appended_tokens_total"] <= 0:
        raise RuntimeError("encoder diagnostic appended no backend tokens")
    for field in (
        "queue_time_observations",
        "inference_time_observations",
        "e2e_time_observations",
        "prompt_token_observations",
    ):
        if engine[field] != requests:
            raise RuntimeError(f"vLLM completed-request metric diverged: {field}")


def _stage_report(delta: dict[str, float], calls: int) -> dict[str, float]:
    return {
        "tokenization_mean_seconds": delta["tokenization_seconds_total"] / calls,
        "coordinator_mean_seconds": delta["request_seconds_total"] / calls,
        "backend_mean_seconds": delta["backend_seconds_total"] / calls,
        "session_lock_wait_mean_seconds": (
            delta["session_lock_wait_seconds_total"] / calls
        ),
        "backend_appended_tokens": delta["backend_appended_tokens_total"],
    }


def _engine_report(delta: dict[str, float]) -> dict[str, float]:
    def mean(seconds: str, observations: str) -> float:
        count = delta[observations]
        return delta[seconds] / count if count else 0.0

    return {
        **delta,
        "queue_time_mean_seconds": mean(
            "queue_time_seconds_total", "queue_time_observations"
        ),
        "inference_time_mean_seconds": mean(
            "inference_time_seconds_total", "inference_time_observations"
        ),
        "e2e_time_mean_seconds": mean(
            "e2e_time_seconds_total", "e2e_time_observations"
        ),
    }


def _run_level(
    *,
    client: CanaryClient,
    run_id: str,
    phase: str,
    concurrency: int,
) -> dict[str, Any]:
    all_latencies: list[float] = []
    wave_walls: list[float] = []
    coordinator_deltas: list[dict[str, float]] = []
    engine_deltas: list[dict[str, float]] = []
    sampler_reports: list[dict[str, Any]] = []
    serialized_tokens: list[int] = []

    for wave in range(WAVES_PER_LEVEL):
        episode_ids = [
            _episode_hash(f"{run_id}:{phase}:c{concurrency}:w{wave}:{index}")
            for index in range(concurrency)
        ]
        turns = FIRST_TURNS if phase == "create" else APPENDED_TURNS
        if phase == "append":
            setup, _setup_wall = _run_concurrent(client, episode_ids, FIRST_TURNS)
            if any(result["action"] != "created" for result in setup):
                raise RuntimeError("append setup did not create every session")

        before = _read_metrics(client)
        sampler = _MetricsSampler(client)
        sampler.start()
        try:
            results, wall = _run_concurrent(client, episode_ids, turns)
        finally:
            try:
                sampler.stop()
            finally:
                _close_all(client, episode_ids)
        after = _read_completed_metrics(client, before, concurrency)

        expected_action = "created" if phase == "create" else "appended"
        if any(result["action"] != expected_action for result in results):
            raise RuntimeError(f"encoder diagnostic {phase} action diverged")
        coordinator_delta = _metric_delta(
            before,
            after,
            "coordinator",
            COORDINATOR_CUMULATIVE_FIELDS,
        )
        engine_delta = _metric_delta(
            before,
            after,
            "engine",
            ENGINE_CUMULATIVE_FIELDS,
        )
        _validate_phase_deltas(
            coordinator=coordinator_delta,
            engine=engine_delta,
            requests=concurrency,
        )
        all_latencies.extend(result["latency_seconds"] for result in results)
        serialized_tokens.extend(result["serialized_tokens"] for result in results)
        wave_walls.append(wall)
        coordinator_deltas.append(coordinator_delta)
        engine_deltas.append(engine_delta)
        sampler_reports.append(sampler.report())

    requests = concurrency * WAVES_PER_LEVEL
    coordinator = _merge_deltas(coordinator_deltas)
    engine = _merge_deltas(engine_deltas)
    sampled_maximum = {
        field: max(report["maximum"][field] for report in sampler_reports)
        for field in SAMPLED_FIELDS
    }
    if phase == "create" and sampled_maximum["coordinator_backend_inflight"] < 1:
        raise RuntimeError("encoder sampler observed no backend work")
    return {
        "phase": phase,
        "concurrency": concurrency,
        "waves": WAVES_PER_LEVEL,
        "requests": requests,
        "client_latency": _latency_summary(all_latencies),
        "throughput_requests_per_sum_wave_second": requests / math.fsum(wave_walls),
        "sum_wave_wall_seconds": math.fsum(wave_walls),
        "serialized_tokens": {
            "minimum": min(serialized_tokens),
            "maximum": max(serialized_tokens),
            "total": sum(serialized_tokens),
        },
        "coordinator": {
            **coordinator,
            **_stage_report(coordinator, requests),
        },
        "engine": _engine_report(engine),
        "sampler": {
            "samples": sum(report["samples"] for report in sampler_reports),
            "errors": sum(report["errors"] for report in sampler_reports),
            "maximum": sampled_maximum,
        },
    }


def _run_same_episode_control(
    client: CanaryClient,
    run_id: str,
) -> dict[str, Any]:
    episode_id = _episode_hash(f"{run_id}:same-episode-control")
    before = _read_metrics(client)
    results, wall = _run_concurrent(
        client,
        [episode_id, episode_id],
        SAME_EPISODE_TURNS,
    )
    client.close(episode_id)
    after = _read_completed_metrics(client, before, 1)
    coordinator = _metric_delta(
        before,
        after,
        "coordinator",
        COORDINATOR_CUMULATIVE_FIELDS,
    )
    engine = _metric_delta(before, after, "engine", ENGINE_CUMULATIVE_FIELDS)
    if sorted(result["action"] for result in results) != ["created", "reused"]:
        raise RuntimeError("same-episode control action diverged")
    if coordinator["requests_started_total"] != SAME_EPISODE_CONTROL_REQUESTS:
        raise RuntimeError("same-episode request count diverged")
    if coordinator["session_lock_contentions_total"] != 1:
        raise RuntimeError("same-episode control did not observe exactly one waiter")
    if coordinator["backend_calls_succeeded_total"] != 1:
        raise RuntimeError("same-episode reuse unexpectedly called the backend")
    if engine["e2e_time_observations"] != 1:
        raise RuntimeError("same-episode engine completion count diverged")
    return {
        "requests": SAME_EPISODE_CONTROL_REQUESTS,
        "actions": sorted(result["action"] for result in results),
        "wall_seconds": wall,
        "client_latency": _latency_summary(
            [result["latency_seconds"] for result in results]
        ),
        "coordinator": coordinator,
        "engine": _engine_report(engine),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    modal_key = os.environ.get("RAYLINE_ARC_MODAL_KEY", "")
    modal_secret = os.environ.get("RAYLINE_ARC_MODAL_SECRET", "")
    if not modal_key or not modal_secret:
        raise SystemExit(
            "RAYLINE_ARC_MODAL_KEY and RAYLINE_ARC_MODAL_SECRET are required"
        )
    client = CanaryClient(
        base_url=args.base_url,
        modal_key=modal_key,
        modal_secret=modal_secret,
        timeout_seconds=args.timeout_seconds,
    )

    started = time.perf_counter()
    initial = _read_metrics(client)
    results = [
        _run_level(
            client=client,
            run_id=args.run_id,
            phase=phase,
            concurrency=concurrency,
        )
        for phase in PHASES
        for concurrency in CONCURRENCY_LEVELS
    ]
    same_episode = _run_same_episode_control(client, args.run_id)
    final = _read_metrics(client)
    health, _health_latency = client.request("GET", "/health")
    if health.get("resident_sessions") != 0 or health.get("resident_tokens") != 0:
        raise RuntimeError("encoder diagnostic leaked retained sessions")
    coordinator = final["coordinator"]
    if coordinator["requests_started_total"] != MAX_POOLING_REQUESTS:
        raise RuntimeError("encoder diagnostic total request count diverged")
    if coordinator["requests_failed_total"] != 0:
        raise RuntimeError("encoder diagnostic recorded failed requests")
    if coordinator["backend_inflight_max"] < max(CONCURRENCY_LEVELS):
        raise RuntimeError("cross-episode calls did not overlap at the vLLM boundary")
    if coordinator["session_lock_contentions_total"] != 1:
        raise RuntimeError("encoder diagnostic lock-contention total diverged")
    maximum_engine_running = max(
        result["sampler"]["maximum"]["engine_requests_running"] for result in results
    )
    if maximum_engine_running < MIN_ENGINE_CONCURRENT_REQUESTS:
        raise RuntimeError("vLLM scheduler never exposed concurrent running requests")

    report = {
        "schema_version": REPORT_SCHEMA,
        "run_id": args.run_id,
        "status": "passed",
        "workload": {
            "phases": list(PHASES),
            "concurrency_levels": list(CONCURRENCY_LEVELS),
            "waves_per_level": WAVES_PER_LEVEL,
            "measured_pooling_requests": MEASURED_REQUESTS,
            "append_setup_requests": APPEND_SETUP_REQUESTS,
            "same_episode_control_requests": SAME_EPISODE_CONTROL_REQUESTS,
            "maximum_pooling_requests": MAX_POOLING_REQUESTS,
        },
        "results": results,
        "same_episode_control": same_episode,
        "overall": {
            "elapsed_seconds": time.perf_counter() - started,
            "requests_started_total": coordinator["requests_started_total"],
            "requests_failed_total": coordinator["requests_failed_total"],
            "backend_inflight_max": coordinator["backend_inflight_max"],
            "session_lock_contentions_total": coordinator[
                "session_lock_contentions_total"
            ],
            "maximum_engine_requests_running": maximum_engine_running,
            "initial_engine_metrics_available": initial["engine"]["available"],
        },
        "interpretation_boundary": (
            "coordinator backend time surrounds the retained vLLM append and "
            "therefore includes vLLM scheduling, model execution, and pooling; "
            "vLLM completed-request histograms cover the whole retained session"
        ),
        "budget": {
            "maximum_resource_envelope_usd": MAX_RESOURCE_ENVELOPE_USD,
            "cumulative_before_usd": CUMULATIVE_BEFORE_USD,
            "cumulative_if_full_envelope_usd": (
                CUMULATIVE_BEFORE_USD + MAX_RESOURCE_ENVELOPE_USD
            ),
            "provider_spend_usd": 0.0,
        },
        "provider_calls": 0,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
