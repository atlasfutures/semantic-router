# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.machinery
import math
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

stage = importlib.import_module("openrouter_agentic_stage_benchmark")
stage_metrics = importlib.import_module("openrouter_agentic_stage_metrics")
if "modal" not in sys.modules and importlib.util.find_spec("modal") is None:
    modal_stub = types.ModuleType("modal")
    modal_stub.__spec__ = importlib.machinery.ModuleSpec("modal", loader=None)
    sys.modules["modal"] = modal_stub
launcher = importlib.import_module("run_openrouter_fullstack")
EXPECTED_CONCURRENCY = 4
EXPECTED_REQUESTS_PER_CELL = 12
EXPECTED_MEASURED_REQUESTS = 48
EXPECTED_BENCHMARK_PROVIDER_REQUESTS = 76
EXPECTED_PROVIDER_REQUESTS = 80
EXPECTED_BENCHMARK_EXTERNAL_ATTEMPTS = 155
EXPECTED_EXTERNAL_ATTEMPTS = 166
EXPECTED_COST_GATE_USD = 0.50
EXPECTED_STRATIFIED_CASES = 6
EXPECTED_CASES_PER_MODEL = 2
EXPECTED_CLIENT_MEAN_SECONDS = 2.2
EXPECTED_UPSTREAM_MEAN_SECONDS = 1.1
EXPECTED_COORDINATOR_MEAN_SECONDS = 0.5
EXPECTED_INFERENCE_MEAN_SECONDS = 0.35
EXPECTED_EPHEMERAL_KEY_LIMIT_USD = 0.75


def _histogram(
    *, count: float, total: float, upper_bound: float = 1.0
) -> dict[str, Any]:
    return {
        "count": count,
        "sum_seconds": total,
        "buckets": {upper_bound: count, math.inf: count},
    }


def _encoder_snapshot(*, requests: int, elapsed: float) -> dict[str, Any]:
    coordinator = dict.fromkeys(stage_metrics.COORDINATOR_CUMULATIVE_FIELDS, 0.0)
    engine = dict.fromkeys(stage_metrics.ENGINE_CUMULATIVE_FIELDS, 0.0)
    for field in (
        "tokenization_calls_total",
        "requests_started_total",
        "requests_succeeded_total",
        "backend_calls_started_total",
        "backend_calls_succeeded_total",
    ):
        coordinator[field] = requests
    coordinator.update(
        {
            "tokenization_seconds_total": elapsed * 0.01,
            "request_seconds_total": elapsed,
            "session_lock_wait_seconds_total": 0.0,
            "backend_seconds_total": elapsed * 0.9,
            "backend_appended_tokens_total": requests * 10,
            "backend_inflight_max": min(requests, stage.CONCURRENCY),
            "requests_inflight": 0,
            "session_lock_waiters": 0,
            "backend_inflight": 0,
        }
    )
    for field in (
        "queue_time_observations",
        "inference_time_observations",
        "e2e_time_observations",
        "prompt_token_observations",
    ):
        engine[field] = requests
    engine.update(
        {
            "queue_time_seconds_total": elapsed * 0.1,
            "inference_time_seconds_total": elapsed * 0.7,
            "e2e_time_seconds_total": elapsed * 0.85,
            "prompt_tokens_total": requests * 10,
            "requests_running": 0,
            "requests_waiting": 0,
            "requests_running_max": min(requests, stage.CONCURRENCY),
            "requests_waiting_max": 0,
            "requests_scheduled_max": min(requests, stage.CONCURRENCY),
            "scheduler_updates_total": requests,
        }
    )
    return {"coordinator": coordinator, "engine": engine}


def test_stage_packet_has_exact_bounded_request_cells() -> None:
    assert stage.CONCURRENCY == EXPECTED_CONCURRENCY
    assert stage.NATURAL_REQUESTS_PER_PATH == EXPECTED_REQUESTS_PER_CELL
    assert stage.STRATIFIED_REQUESTS_PER_PATH == EXPECTED_REQUESTS_PER_CELL
    assert stage.MEASURED_REQUESTS == EXPECTED_MEASURED_REQUESTS
    assert stage.MAX_BENCHMARK_PROVIDER_REQUESTS == EXPECTED_BENCHMARK_PROVIDER_REQUESTS
    assert stage.MAX_PROVIDER_REQUESTS == EXPECTED_PROVIDER_REQUESTS
    assert stage.MAX_BENCHMARK_EXTERNAL_ATTEMPTS == EXPECTED_BENCHMARK_EXTERNAL_ATTEMPTS
    assert stage.MAX_EXTERNAL_ATTEMPTS == EXPECTED_EXTERNAL_ATTEMPTS
    assert stage.MAX_REPORTED_PROVIDER_COST_USD == EXPECTED_COST_GATE_USD


def test_stratified_control_uses_the_same_cases_for_every_model() -> None:
    cases = stage._stratified_cases()

    assert len(cases) == EXPECTED_STRATIFIED_CASES
    assert {
        worker: sum(case["expected_worker"] == worker for case in cases)
        for worker in stage.WORKERS
    } == dict.fromkeys(stage.WORKERS, EXPECTED_CASES_PER_MODEL)
    shapes = {
        worker: [
            (case["scenario"], case["messages"], case["tools"])
            for case in cases
            if case["expected_worker"] == worker
        ]
        for worker in stage.WORKERS
    }
    assert shapes["worker-a"] == shapes["worker-b"] == shapes["worker-c"]


def test_router_stage_delta_attributes_arc_without_payloads() -> None:
    before = {
        "routing": _histogram(count=10, total=1.0),
        "encoder": _histogram(count=5, total=0.5),
    }
    after = {
        "routing": _histogram(count=12, total=1.8),
        "encoder": _histogram(count=7, total=1.1),
    }
    results = [
        {
            "total_seconds": 2.0,
            "envoy_upstream_service_seconds": 1.0,
        },
        {
            "total_seconds": 2.4,
            "envoy_upstream_service_seconds": 1.2,
        },
    ]

    report = stage_metrics.router_stage_delta(
        before=before,
        after=after,
        path="arc",
        results=results,
    )

    assert report["routing_histogram"]["observations"] == EXPECTED_CASES_PER_MODEL
    assert report["encoder_histogram"]["observations"] == EXPECTED_CASES_PER_MODEL
    decomposition = report["mean_decomposition"]
    assert decomposition["client_e2e_seconds"] == EXPECTED_CLIENT_MEAN_SECONDS
    assert decomposition["upstream_service_seconds"] == EXPECTED_UPSTREAM_MEAN_SECONDS
    assert math.isclose(decomposition["router_seconds"], 0.4)
    assert math.isclose(decomposition["encoder_seconds"], 0.3)
    assert math.isclose(decomposition["router_non_encoder_seconds"], 0.1)
    assert math.isclose(decomposition["residual_after_router_seconds"], 0.7)


def test_protected_encoder_stage_accepts_last_reported_scheduler_occupancy() -> None:
    after = _encoder_snapshot(requests=12, elapsed=6.0)
    after["engine"]["requests_running"] = EXPECTED_CONCURRENCY
    report = stage_metrics.encoder_stage_delta(
        before=_encoder_snapshot(requests=0, elapsed=0.0),
        after=after,
        requests=12,
    )

    assert report["requests"] == EXPECTED_REQUESTS_PER_CELL
    assert (
        report["coordinator"]["coordinator_mean_seconds"]
        == EXPECTED_COORDINATOR_MEAN_SECONDS
    )
    assert math.isclose(
        report["engine"]["inference_time_mean_seconds"],
        EXPECTED_INFERENCE_MEAN_SECONDS,
    )
    assert report["coordinator_idle_after"] == {
        "requests_inflight": 0,
        "session_lock_waiters": 0,
        "backend_inflight": 0,
    }
    assert report["scheduler_last_reported_after"] == {
        "requests_running": EXPECTED_CONCURRENCY,
        "requests_waiting": 0,
    }
    assert "not asserted as synchronous engine idle" in report["scheduler_semantics"]


def test_launcher_stage_packet_is_closed_and_reuses_agentic_sources() -> None:
    packet = launcher.PACKETS["agentic-stage"]

    assert packet.driver == SCRIPT_DIR / "openrouter_agentic_stage_benchmark.py"
    assert packet.preflight_driver == SCRIPT_DIR / "openrouter_agentic_preflight.py"
    assert packet.config == (
        REPO_ROOT / "deploy/compose/rayline-arc/config-openrouter-agentic.yaml"
    )
    assert packet.key_limit_usd == EXPECTED_EPHEMERAL_KEY_LIMIT_USD
    assert packet.maximum_seconds == 30 * 60
    assert packet.protected_encoder is True
    assert launcher.AGT013_PREREGISTRATION_COMMIT == (
        "a4c7b1b8b8d4bd7b41ed35499d9bc22469e79c3f"
    )
    assert launcher.AGT013_AUTHORIZATION_COMMIT == ""
