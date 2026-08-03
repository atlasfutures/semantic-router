#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Payload-free stage attribution for the agentic OpenRouter diagnostic."""

from __future__ import annotations

import math
import os
from typing import Any

from modal_encoder_diagnostic import (
    COORDINATOR_CUMULATIVE_FIELDS,
    ENGINE_CUMULATIVE_FIELDS,
    _engine_report,
    _metric_delta,
    _read_metrics,
    _stage_report,
    _validate_phase_deltas,
)
from modal_encoder_diagnostic_report import scheduler_snapshot
from modal_fullstack_canary import _summary
from modal_fullstack_diagnostic_metrics import (
    _histogram_report,
    _router_metric_deltas,
    _router_metric_snapshot,
)
from modal_session_canary import CanaryClient

ENCODER_BASE_URL_ENV = "RAYLINE_ARC_E2E_ENCODER_BASE_URL"
ENCODER_KEY_ENV = "RAYLINE_ARC_E2E_MODAL_KEY"
ENCODER_SECRET_ENV = "RAYLINE_ARC_E2E_MODAL_SECRET"
COUNT_TOLERANCE = 1e-9


def encoder_client_from_environment(timeout_seconds: float) -> CanaryClient:
    values = {
        "base_url": os.environ.get(ENCODER_BASE_URL_ENV, ""),
        "modal_key": os.environ.get(ENCODER_KEY_ENV, ""),
        "modal_secret": os.environ.get(ENCODER_SECRET_ENV, ""),
    }
    if not all(values.values()):
        raise ValueError("protected encoder metrics configuration is incomplete")
    return CanaryClient(timeout_seconds=timeout_seconds, **values)


def read_encoder_snapshot(client: CanaryClient) -> dict[str, Any]:
    return _read_metrics(client)


def encoder_stage_delta(
    *,
    before: dict[str, Any],
    after: dict[str, Any],
    requests: int,
) -> dict[str, Any]:
    coordinator = _metric_delta(
        before,
        after,
        "coordinator",
        COORDINATOR_CUMULATIVE_FIELDS,
    )
    engine = _metric_delta(
        before,
        after,
        "engine",
        ENGINE_CUMULATIVE_FIELDS,
    )
    _validate_phase_deltas(
        coordinator=coordinator,
        engine=engine,
        requests=requests,
    )
    coordinator_after = after["coordinator"]
    engine_after = after["engine"]
    idle_fields = {
        "requests_inflight": coordinator_after.get("requests_inflight"),
        "session_lock_waiters": coordinator_after.get("session_lock_waiters"),
        "backend_inflight": coordinator_after.get("backend_inflight"),
        "requests_running": engine_after.get("requests_running"),
        "requests_waiting": engine_after.get("requests_waiting"),
    }
    if any(value != 0 for value in idle_fields.values()):
        raise RuntimeError("protected encoder was not idle after the measured phase")
    before_scheduler = scheduler_snapshot(before)
    after_scheduler = scheduler_snapshot(after)
    return {
        "requests": requests,
        "coordinator": _stage_report(coordinator, requests),
        "engine": _engine_report(engine),
        "process_lifetime_peaks": {
            "coordinator_backend_inflight_before": before["coordinator"][
                "backend_inflight_max"
            ],
            "coordinator_backend_inflight_after": coordinator_after[
                "backend_inflight_max"
            ],
            "engine_requests_running_max_before": before_scheduler[
                "requests_running_max"
            ],
            "engine_requests_running_max_after": after_scheduler[
                "requests_running_max"
            ],
            "engine_requests_waiting_max_before": before_scheduler[
                "requests_waiting_max"
            ],
            "engine_requests_waiting_max_after": after_scheduler[
                "requests_waiting_max"
            ],
            "engine_requests_scheduled_max_before": before_scheduler[
                "requests_scheduled_max"
            ],
            "engine_requests_scheduled_max_after": after_scheduler[
                "requests_scheduled_max"
            ],
        },
        "idle_after": idle_fields,
    }


def router_snapshot(metrics: str) -> dict[str, dict[str, Any]]:
    return _router_metric_snapshot(metrics)


def router_stage_delta(
    *,
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    path: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    requests = len(results)
    deltas = _router_metric_deltas(before, after)
    expected_routing = 0 if path == "direct" else requests
    expected_encoder = requests if path == "arc" else 0
    actual_routing = float(deltas["routing"]["count"])
    actual_encoder = float(deltas["encoder"]["count"])
    if not math.isclose(actual_routing, expected_routing, abs_tol=COUNT_TOLERANCE):
        raise RuntimeError("routing histogram count diverged during stage diagnostic")
    if not math.isclose(actual_encoder, expected_encoder, abs_tol=COUNT_TOLERANCE):
        raise RuntimeError("encoder histogram count diverged during stage diagnostic")

    routing = _histogram_report(deltas["routing"])
    encoder = _histogram_report(deltas["encoder"])
    upstream_values = [
        float(result["envoy_upstream_service_seconds"])
        for result in results
        if result["envoy_upstream_service_seconds"] is not None
    ]
    residual_values = [
        max(
            0.0,
            float(result["total_seconds"])
            - float(result["envoy_upstream_service_seconds"]),
        )
        for result in results
        if result["envoy_upstream_service_seconds"] is not None
    ]
    mean_decomposition: dict[str, float | None] | None = None
    residual = None
    if upstream_values and residual_values:
        residual = {
            **_summary(residual_values),
            "mean_seconds": math.fsum(residual_values) / len(residual_values),
        }
        client_mean = (
            math.fsum(float(result["total_seconds"]) for result in results) / requests
        )
        upstream_mean = math.fsum(upstream_values) / len(upstream_values)
        routing_mean = routing["mean_seconds"] or 0.0
        encoder_mean = encoder["mean_seconds"] or 0.0
        mean_decomposition = {
            "client_e2e_seconds": client_mean,
            "upstream_service_seconds": upstream_mean,
            "gateway_residual_seconds": residual["mean_seconds"],
            "router_seconds": routing["mean_seconds"],
            "encoder_seconds": encoder["mean_seconds"],
            "router_non_encoder_seconds": max(0.0, routing_mean - encoder_mean),
            "residual_after_router_seconds": max(
                0.0,
                float(residual["mean_seconds"]) - routing_mean,
            ),
        }
    return {
        "routing_histogram": routing,
        "encoder_histogram": encoder,
        "client_minus_upstream": residual,
        "mean_decomposition": mean_decomposition,
    }
