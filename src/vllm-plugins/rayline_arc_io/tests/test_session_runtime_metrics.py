# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from rayline_arc_io.session_metrics import SessionEngineMetricsSnapshot
from rayline_arc_io.session_runtime import VLLMSessionEngineMetricsProvider


def gauge(name: str, value: float):
    return SimpleNamespace(name=name, value=value)


def histogram(name: str, count: int, total: float):
    return SimpleNamespace(name=name, count=count, sum=total)


def test_vllm_metrics_provider_aggregates_curated_scheduler_metrics() -> None:
    metrics = [
        gauge("vllm:num_requests_running", 2),
        gauge("vllm:num_requests_waiting", 1),
        histogram("vllm:request_queue_time_seconds", 4, 0.25),
        histogram("vllm:request_inference_time_seconds", 4, 1.5),
        histogram("vllm:e2e_request_latency_seconds", 4, 1.75),
        histogram("vllm:request_prompt_tokens", 4, 128),
        gauge("unrelated", 99),
    ]

    snapshot = VLLMSessionEngineMetricsProvider(lambda: metrics)()

    assert snapshot == SessionEngineMetricsSnapshot(
        available=True,
        requests_running=2,
        requests_waiting=1,
        queue_time_observations=4,
        queue_time_seconds_total=0.25,
        inference_time_observations=4,
        inference_time_seconds_total=1.5,
        e2e_time_observations=4,
        e2e_time_seconds_total=1.75,
        prompt_token_observations=4,
        prompt_tokens_total=128,
    )


def test_vllm_metrics_provider_marks_incomplete_registry_unavailable() -> None:
    snapshot = VLLMSessionEngineMetricsProvider(
        lambda: [gauge("vllm:num_requests_running", 0)]
    )()

    assert snapshot.available is False
    assert snapshot.requests_running is None
