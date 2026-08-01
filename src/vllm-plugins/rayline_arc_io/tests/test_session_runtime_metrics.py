# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from rayline_arc_io.session_metrics import (
    SessionAppendMetricsSnapshot,
    SessionEngineMetricsSnapshot,
)
from rayline_arc_io.session_runtime import (
    VLLMSessionEngineMetricsProvider,
    _retained_append_metrics,
)

QUEUE_TIME = 0.1
INFERENCE_TIME = 0.2
E2E_TIME = 0.3


def test_retained_append_metrics_fail_closed_and_preserve_scope() -> None:
    metrics = _retained_append_metrics(
        SimpleNamespace(
            queue_time=QUEUE_TIME,
            inference_time=INFERENCE_TIME,
            e2e_time=E2E_TIME,
        )
    )

    assert metrics.queue_time == QUEUE_TIME
    assert metrics.inference_time == INFERENCE_TIME
    assert metrics.e2e_time == E2E_TIME
    with pytest.raises(RuntimeError, match="unavailable"):
        _retained_append_metrics(None)
    with pytest.raises(RuntimeError, match="invalid"):
        _retained_append_metrics(
            SimpleNamespace(
                queue_time=QUEUE_TIME,
                inference_time=float("nan"),
                e2e_time=E2E_TIME,
            )
        )


def test_vllm_metrics_provider_combines_cached_load_and_append_metrics() -> None:
    scheduler = SimpleNamespace(num_requests_running=2, num_requests_waiting=1)
    append = SessionAppendMetricsSnapshot(
        observations=4,
        queue_time_seconds_total=0.25,
        inference_time_seconds_total=1.5,
        e2e_time_seconds_total=1.75,
        appended_tokens_total=128,
    )

    snapshot = VLLMSessionEngineMetricsProvider(
        lambda: scheduler,
        lambda: append,
    )()

    assert snapshot == SessionEngineMetricsSnapshot(
        available=True,
        measurement_scope="retained_append",
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


def test_vllm_metrics_provider_reports_zero_before_first_append() -> None:
    scheduler = SimpleNamespace(num_requests_running=0, num_requests_waiting=0)
    append = SessionAppendMetricsSnapshot(
        observations=0,
        queue_time_seconds_total=0.0,
        inference_time_seconds_total=0.0,
        e2e_time_seconds_total=0.0,
        appended_tokens_total=0,
    )

    snapshot = VLLMSessionEngineMetricsProvider(
        lambda: scheduler,
        lambda: append,
    )()

    assert snapshot.available is True
    assert snapshot.measurement_scope == "retained_append"
    assert snapshot.e2e_time_observations == 0
