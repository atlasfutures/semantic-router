#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Prometheus delta helpers for the Rayline full-stack diagnostic."""

from __future__ import annotations

import math
import re
from typing import Any

from modal_fullstack_benchmark import (
    COUNTER_METRICS,
    HISTOGRAM_METRICS,
    _metric_total,
)
from modal_fullstack_canary import WORKERS

ROUTER_HISTOGRAMS = {
    "routing": "llm_model_routing_latency_seconds",
    "encoder": "llm_rayline_arc_encoder_latency_seconds",
}
METRIC_RESET_TOLERANCE = 1e-9


def _histogram_buckets(metrics: str, name: str) -> dict[float, float]:
    number = r"([0-9.eE+-]+)"
    pattern = re.compile(rf"^{re.escape(name)}_bucket\{{([^}}]+)\}}\s+{number}$")
    buckets: dict[float, float] = {}
    for line in metrics.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        le_match = re.search(r'(?:^|,)le="([^"]+)"(?:,|$)', match.group(1))
        if le_match is None:
            continue
        boundary = math.inf if le_match.group(1) == "+Inf" else float(le_match.group(1))
        buckets[boundary] = buckets.get(boundary, 0.0) + float(match.group(2))
    return buckets


def _histogram_snapshot(metrics: str, name: str) -> dict[str, Any]:
    return {
        "count": _metric_total(metrics, f"{name}_count"),
        "sum_seconds": _metric_total(metrics, f"{name}_sum"),
        "buckets": _histogram_buckets(metrics, name),
    }


def _router_metric_snapshot(metrics: str) -> dict[str, dict[str, Any]]:
    return {
        label: _histogram_snapshot(metrics, name)
        for label, name in ROUTER_HISTOGRAMS.items()
    }


def _histogram_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    buckets = {
        boundary: after["buckets"].get(boundary, 0.0)
        - before["buckets"].get(boundary, 0.0)
        for boundary in set(before["buckets"]) | set(after["buckets"])
    }
    result = {
        "count": after["count"] - before["count"],
        "sum_seconds": after["sum_seconds"] - before["sum_seconds"],
        "buckets": buckets,
    }
    values = [result["count"], result["sum_seconds"], *buckets.values()]
    if any(float(value) < -METRIC_RESET_TOLERANCE for value in values):
        raise RuntimeError("router histogram reset during the diagnostic")
    return result


def _router_metric_deltas(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        label: _histogram_delta(before[label], after[label])
        for label in ROUTER_HISTOGRAMS
    }


def _merge_histograms(values: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[float, float] = {}
    for value in values:
        for boundary, count in value["buckets"].items():
            buckets[boundary] = buckets.get(boundary, 0.0) + float(count)
    return {
        "count": sum(float(value["count"]) for value in values),
        "sum_seconds": sum(float(value["sum_seconds"]) for value in values),
        "buckets": buckets,
    }


def _histogram_report(value: dict[str, Any]) -> dict[str, float | None]:
    count = float(value["count"])
    mean = float(value["sum_seconds"]) / count if count else None
    p95_upper: float | None = None
    if count:
        target = count * 0.95
        for boundary, cumulative in sorted(value["buckets"].items()):
            if float(cumulative) >= target:
                p95_upper = None if math.isinf(boundary) else boundary
                break
    return {
        "observations": count,
        "mean_seconds": mean,
        "p95_bucket_upper_bound_seconds": p95_upper,
    }


def _merge_worker_deltas(
    values: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    merged = {worker: {} for worker in WORKERS}
    for worker in WORKERS:
        names = {name for value in values for name in value[worker]}
        merged[worker] = {
            name: sum(float(value[worker].get(name, 0.0)) for value in values)
            for name in names
        }
    return merged


def _worker_metric_report(
    deltas: dict[str, dict[str, float]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        name: sum(float(deltas[worker][name]) for worker in WORKERS)
        for name in COUNTER_METRICS
    }
    for name in HISTOGRAM_METRICS:
        count = sum(float(deltas[worker][f"{name}_count"]) for worker in WORKERS)
        total = sum(float(deltas[worker][f"{name}_sum_seconds"]) for worker in WORKERS)
        report[name] = {
            "observations": count,
            "mean_seconds": total / count if count else None,
        }
    return report


def _validate_phase_metrics(
    *,
    path: str,
    requests: int,
    worker_deltas: dict[str, dict[str, float]],
    router_deltas: dict[str, dict[str, Any]],
) -> None:
    worker_report = _worker_metric_report(worker_deltas)
    if not math.isclose(worker_report["request_success"], requests, abs_tol=1e-9):
        raise RuntimeError("vLLM success counter mismatch during diagnostic")
    if worker_report["preemptions"] != 0:
        raise RuntimeError("vLLM preempted a diagnostic request")
    for name in HISTOGRAM_METRICS:
        if not math.isclose(
            float(worker_report[name]["observations"]), requests, abs_tol=1e-9
        ):
            raise RuntimeError(f"vLLM {name} histogram mismatch during diagnostic")

    expected_routing = 0 if path == "direct" else requests
    expected_encoder = requests if path == "arc" else 0
    if not math.isclose(
        float(router_deltas["routing"]["count"]),
        expected_routing,
        abs_tol=1e-9,
    ):
        raise RuntimeError("router routing histogram mismatch during diagnostic")
    if not math.isclose(
        float(router_deltas["encoder"]["count"]),
        expected_encoder,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Rayline encoder histogram mismatch during diagnostic")
