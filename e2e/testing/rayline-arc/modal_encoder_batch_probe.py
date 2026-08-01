#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Minimal protected-encoder scheduled-batch proof.

The packet sends one synchronized wave of fixed public synthetic turns. It
emits aggregate timings and counters only and never calls a generation worker
or provider.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from modal_encoder_diagnostic import (
    COORDINATOR_CUMULATIVE_FIELDS,
    ENGINE_CUMULATIVE_FIELDS,
    FIRST_TURNS,
    _close_all,
    _engine_report,
    _latency_summary,
    _metric_delta,
    _read_append_metrics,
    _read_metrics,
    _run_concurrent,
    _stage_report,
    _validate_phase_deltas,
)
from modal_encoder_diagnostic_report import scheduler_snapshot, validate_or_emit_failure
from modal_session_canary import CanaryClient, _episode_hash

PROBE_CONCURRENCY = 8
PROBE_WAVES = 1
MAX_POOLING_REQUESTS = PROBE_CONCURRENCY * PROBE_WAVES
MIN_SCHEDULED_REQUESTS = 2
METRICS_SCHEMA = "rayline.arc.session-metrics-response.v4"
REPORT_SCHEMA = "rayline.arc.modal-encoder-batch-probe.v1"
MAX_RESOURCE_ENVELOPE_USD = 2.4996168
CUMULATIVE_BEFORE_USD = 24.23093122


def _validate_fresh_start(metrics: dict[str, Any]) -> None:
    coordinator = metrics["coordinator"]
    engine = metrics["engine"]
    if coordinator["requests_started_total"] != 0:
        raise RuntimeError("batch probe requires a fresh coordinator")
    if engine["e2e_time_observations"] != 0:
        raise RuntimeError("batch probe requires fresh append metrics")
    if scheduler_snapshot(metrics)["requests_scheduled_max"] != 0:
        raise RuntimeError("batch probe requires a fresh scheduler peak")


def _build_report(
    *,
    run_id: str,
    results: list[dict[str, Any]],
    wall_seconds: float,
    coordinator_delta: dict[str, float],
    engine_delta: dict[str, float],
    final_metrics: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    scheduler = scheduler_snapshot(final_metrics)
    return {
        "schema_version": REPORT_SCHEMA,
        "run_id": run_id,
        "status": "passed",
        "workload": {
            "phase": "create",
            "concurrency": PROBE_CONCURRENCY,
            "waves": PROBE_WAVES,
            "maximum_pooling_requests": MAX_POOLING_REQUESTS,
        },
        "result": {
            "client_latency": _latency_summary(
                [result["latency_seconds"] for result in results]
            ),
            "throughput_requests_per_second": MAX_POOLING_REQUESTS / wall_seconds,
            "wall_seconds": wall_seconds,
            "serialized_tokens": {
                "minimum": min(result["serialized_tokens"] for result in results),
                "maximum": max(result["serialized_tokens"] for result in results),
                "total": sum(result["serialized_tokens"] for result in results),
            },
            "coordinator": {
                **coordinator_delta,
                **_stage_report(coordinator_delta, MAX_POOLING_REQUESTS),
            },
            "engine": _engine_report(engine_delta),
        },
        "overall": {
            "elapsed_seconds": elapsed_seconds,
            "requests_started_total": final_metrics["coordinator"][
                "requests_started_total"
            ],
            "requests_failed_total": final_metrics["coordinator"][
                "requests_failed_total"
            ],
            "backend_inflight_max": final_metrics["coordinator"][
                "backend_inflight_max"
            ],
            "engine_scheduler_process_lifetime": scheduler,
        },
        "acceptance": {
            "minimum_backend_inflight": PROBE_CONCURRENCY,
            "minimum_scheduled_requests": MIN_SCHEDULED_REQUESTS,
        },
        "interpretation_boundary": (
            "requests_scheduled_max is captured from SchedulerIterationDetails "
            "before vLLM execution; backend_inflight_max is retained by the "
            "protected coordinator; both peaks belong to this fresh process"
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
        "release_qualification_1000_executed": False,
    }


def run_probe(client: CanaryClient, run_id: str) -> dict[str, Any]:
    started = time.perf_counter()
    before = _read_metrics(client)
    _validate_fresh_start(before)
    episode_ids = [
        _episode_hash(f"{run_id}:batch-proof:{index}")
        for index in range(PROBE_CONCURRENCY)
    ]
    try:
        results, wall_seconds = _run_concurrent(client, episode_ids, FIRST_TURNS)
    finally:
        _close_all(client, episode_ids)
    after = _read_append_metrics(client, before, MAX_POOLING_REQUESTS)
    if any(result["action"] != "created" for result in results):
        raise RuntimeError("batch probe did not create every session")
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
        requests=MAX_POOLING_REQUESTS,
    )
    health, _health_latency = client.request("GET", "/health")
    if health.get("resident_sessions") != 0 or health.get("resident_tokens") != 0:
        raise RuntimeError("batch probe leaked retained sessions")
    report = _build_report(
        run_id=run_id,
        results=results,
        wall_seconds=wall_seconds,
        coordinator_delta=coordinator_delta,
        engine_delta=engine_delta,
        final_metrics=after,
        elapsed_seconds=time.perf_counter() - started,
    )
    validate_or_emit_failure(
        report,
        after["coordinator"],
        after["engine"],
        minimum_backend_inflight=PROBE_CONCURRENCY,
        minimum_scheduled_requests=MIN_SCHEDULED_REQUESTS,
    )
    return report


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
    report = run_probe(
        CanaryClient(
            base_url=args.base_url,
            modal_key=modal_key,
            modal_secret=modal_secret,
            timeout_seconds=args.timeout_seconds,
        ),
        args.run_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
