# SPDX-License-Identifier: Apache-2.0

"""Aggregate-only scheduler gates and failure receipt emission."""

from __future__ import annotations

import json
from typing import Any

ENGINE_SCHEDULER_FIELDS = (
    "requests_running",
    "requests_waiting",
    "requests_running_max",
    "requests_waiting_max",
    "requests_scheduled_max",
    "scheduler_updates_total",
)


def build_report(
    *,
    schema_version: str,
    run_id: str,
    phases: tuple[str, ...],
    concurrency_levels: tuple[int, ...],
    waves_per_level: int,
    measured_requests: int,
    append_setup_requests: int,
    same_episode_requests: int,
    maximum_requests: int,
    results: list[dict[str, Any]],
    same_episode: dict[str, Any],
    coordinator: dict[str, Any],
    scheduler: dict[str, int],
    initial_engine_available: bool,
    elapsed_seconds: float,
    maximum_resource_envelope_usd: float,
    cumulative_before_usd: float,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "run_id": run_id,
        "status": "passed",
        "workload": {
            "phases": list(phases),
            "concurrency_levels": list(concurrency_levels),
            "waves_per_level": waves_per_level,
            "measured_pooling_requests": measured_requests,
            "append_setup_requests": append_setup_requests,
            "same_episode_control_requests": same_episode_requests,
            "maximum_pooling_requests": maximum_requests,
        },
        "results": results,
        "same_episode_control": same_episode,
        "overall": {
            "elapsed_seconds": elapsed_seconds,
            "requests_started_total": coordinator["requests_started_total"],
            "requests_failed_total": coordinator["requests_failed_total"],
            "backend_inflight_max": coordinator["backend_inflight_max"],
            "session_lock_contentions_total": coordinator[
                "session_lock_contentions_total"
            ],
            "engine_scheduler_process_lifetime": scheduler,
            "initial_engine_metrics_available": initial_engine_available,
        },
        "interpretation_boundary": (
            "coordinator backend time surrounds the retained vLLM append and "
            "therefore includes vLLM scheduling, model execution, and pooling; "
            "engine timing counters cover each completed retained append, while "
            "the scheduled-request maximum is captured before engine execution and "
            "occupancy maxima are captured after output processing; all are retained "
            "inside the fresh dedicated vLLM encoder and read after completion without "
            "hot client-side metrics polling"
        ),
        "budget": {
            "maximum_resource_envelope_usd": maximum_resource_envelope_usd,
            "cumulative_before_usd": cumulative_before_usd,
            "cumulative_if_full_envelope_usd": (
                cumulative_before_usd + maximum_resource_envelope_usd
            ),
            "provider_spend_usd": 0.0,
        },
        "provider_calls": 0,
        "automatic_prefix_cache_enabled": False,
        "release_qualification_1000_executed": False,
    }


def scheduler_snapshot(metrics: dict[str, Any]) -> dict[str, int]:
    engine = metrics["engine"]
    snapshot: dict[str, int] = {}
    for field in ENGINE_SCHEDULER_FIELDS:
        value = engine.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise TypeError(f"invalid vLLM scheduler metric: engine.{field}")
        snapshot[field] = value
    return snapshot


def validate_capacity_peaks(
    coordinator: dict[str, Any],
    engine: dict[str, Any],
    *,
    minimum_backend_inflight: int,
    minimum_scheduled_requests: int,
) -> None:
    if coordinator["backend_inflight_max"] < minimum_backend_inflight:
        raise RuntimeError("cross-episode calls did not overlap at the vLLM boundary")
    scheduled_max = engine.get("requests_scheduled_max")
    if not isinstance(scheduled_max, int) or isinstance(scheduled_max, bool):
        raise TypeError("vLLM scheduled batch peak is unavailable")
    if scheduled_max < minimum_scheduled_requests:
        raise RuntimeError("vLLM scheduler never scheduled concurrent requests")
    updates = engine.get("scheduler_updates_total")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise RuntimeError("vLLM scheduler emitted no load updates")


def validate_or_emit_failure(
    report: dict[str, Any],
    coordinator: dict[str, Any],
    engine: dict[str, Any],
    *,
    minimum_backend_inflight: int,
    minimum_scheduled_requests: int,
) -> None:
    try:
        validate_capacity_peaks(
            coordinator,
            engine,
            minimum_backend_inflight=minimum_backend_inflight,
            minimum_scheduled_requests=minimum_scheduled_requests,
        )
    except (RuntimeError, TypeError) as error:
        report["status"] = "failed"
        report["failure"] = {
            "phase": "final_capacity_peak_validation",
            "type": type(error).__name__,
            "message": str(error),
        }
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        raise
