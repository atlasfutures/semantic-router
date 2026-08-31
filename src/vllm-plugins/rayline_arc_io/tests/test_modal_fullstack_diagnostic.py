# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "e2e/testing/rayline-arc"
sys.path.insert(0, str(SCRIPT_DIR))

diagnostic = importlib.import_module("modal_fullstack_diagnostic")
diagnostic_metrics = importlib.import_module("modal_fullstack_diagnostic_metrics")

EXPECTED_REQUESTS_PER_PATH = 10
EXPECTED_MEASURED_REQUESTS = 30
EXPECTED_MAX_REQUESTS = 62
EXPECTED_REQUESTS_PER_WORKER = 15
EXPECTED_WAVES_PER_LEVEL = 2
EXPECTED_HISTOGRAM_OBSERVATIONS = 2
EXPECTED_P95_BUCKET_SECONDS = 0.2
EXPECTED_WORKER_B_TEMPERATURE = 0.3


def test_diagnostic_workload_is_fixed_small_and_balanced() -> None:
    assert diagnostic.PATHS == ("direct", "gateway_static", "arc")
    assert diagnostic.CONCURRENCY_LEVELS == (1, 4)
    assert diagnostic.WAVES_PER_LEVEL == EXPECTED_WAVES_PER_LEVEL
    assert diagnostic.REQUESTS_PER_PATH == EXPECTED_REQUESTS_PER_PATH
    assert diagnostic.MEASURED_GENERATION_REQUESTS == EXPECTED_MEASURED_REQUESTS
    assert diagnostic.MAX_GENERATION_REQUESTS == EXPECTED_MAX_REQUESTS
    assert diagnostic.EXPECTED_REQUESTS_PER_WORKER == EXPECTED_REQUESTS_PER_WORKER
    for concurrency in diagnostic.CONCURRENCY_LEVELS:
        targets = [
            target
            for wave in range(diagnostic.WAVES_PER_LEVEL)
            for target in diagnostic._balanced_targets(concurrency, wave)
        ]
        assert targets.count("worker-a") == targets.count("worker-b")


def test_router_histogram_delta_reports_mean_and_p95_bucket() -> None:
    before_text = """
llm_rayline_arc_encoder_latency_seconds_bucket{le="0.1"} 1
llm_rayline_arc_encoder_latency_seconds_bucket{le="0.2"} 2
llm_rayline_arc_encoder_latency_seconds_bucket{le="+Inf"} 2
llm_rayline_arc_encoder_latency_seconds_sum 0.25
llm_rayline_arc_encoder_latency_seconds_count 2
""".strip()
    after_text = """
llm_rayline_arc_encoder_latency_seconds_bucket{le="0.1"} 2
llm_rayline_arc_encoder_latency_seconds_bucket{le="0.2"} 4
llm_rayline_arc_encoder_latency_seconds_bucket{le="+Inf"} 4
llm_rayline_arc_encoder_latency_seconds_sum 0.55
llm_rayline_arc_encoder_latency_seconds_count 4
""".strip()

    before = diagnostic_metrics._histogram_snapshot(
        before_text, "llm_rayline_arc_encoder_latency_seconds"
    )
    after = diagnostic_metrics._histogram_snapshot(
        after_text, "llm_rayline_arc_encoder_latency_seconds"
    )
    report = diagnostic_metrics._histogram_report(
        diagnostic_metrics._histogram_delta(before, after)
    )

    assert report["observations"] == EXPECTED_HISTOGRAM_OBSERVATIONS
    assert math.isclose(float(report["mean_seconds"]), 0.15)
    assert report["p95_bucket_upper_bound_seconds"] == EXPECTED_P95_BUCKET_SECONDS


@pytest.mark.parametrize("path", diagnostic.PATHS)
def test_all_paths_use_the_same_frozen_execution_fields(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_chat(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        target = "worker-a"
        if kwargs["prompt"] == "public-b":
            target = "worker-b"
        return {
            "latency_seconds": 0.5,
            "response_model": diagnostic.WORKERS[target],
            "completion_tokens": 4,
            "selected_worker": target if path == "arc" else "",
            "envoy_upstream_service_seconds": (None if path == "direct" else 0.4),
            "envoy_attempt_count": None if path == "direct" else 1,
        }

    monkeypatch.setattr(diagnostic, "_nonstream_chat", fake_chat)
    result = diagnostic._one_request(
        path=path,
        target_worker="worker-b",
        request_index=0,
        gateway_url="http://gateway",
        worker_urls={"worker-a": "http://a", "worker-b": "http://b"},
        selected_prompts={"worker-a": "public-a", "worker-b": "public-b"},
        worker_authorization="Bearer public-test-key",
        run_id="public-test-run",
        phase_label="unit",
        timeout_seconds=1,
    )

    assert result["selected_worker"] == "worker-b"
    assert len(calls) == 1
    call = calls[0]
    assert call["prompt"] == "public-b"
    assert call["max_tokens"] == diagnostic.FIXED_MAX_TOKENS
    assert call["temperature"] == EXPECTED_WORKER_B_TEMPERATURE
    assert call["extra_fields"] == {
        "seed": diagnostic.FIXED_SEED,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    if path == "direct":
        assert call["base_url"] == "http://b"
        assert call["model"] == diagnostic.WORKERS["worker-b"]
        assert "episode_id" not in call
    elif path == "gateway_static":
        assert call["base_url"] == "http://gateway"
        assert call["model"] == "worker-b"
        assert "episode_id" not in call
    else:
        assert call["base_url"] == "http://gateway"
        assert call["model"] == "auto"
        assert call["episode_id"].startswith("real-workers-")


def test_static_phase_requires_router_but_not_encoder_observations() -> None:
    worker_deltas = {
        worker: {
            "request_success": 2,
            "prompt_tokens": 20,
            "generation_tokens": 8,
            "preemptions": 0,
            "time_to_first_token_count": 2,
            "time_to_first_token_sum_seconds": 0.2,
            "e2e_request_latency_count": 2,
            "e2e_request_latency_sum_seconds": 0.8,
            "request_queue_time_count": 2,
            "request_queue_time_sum_seconds": 0.01,
        }
        for worker in diagnostic.WORKERS
    }
    router_deltas = {
        "routing": {"count": 4, "sum_seconds": 0.1, "buckets": {}},
        "encoder": {"count": 0, "sum_seconds": 0, "buckets": {}},
    }

    diagnostic._validate_phase_metrics(
        path="gateway_static",
        requests=4,
        worker_deltas=worker_deltas,
        router_deltas=router_deltas,
    )

    router_deltas["encoder"]["count"] = 1
    with pytest.raises(RuntimeError, match="encoder histogram"):
        diagnostic._validate_phase_metrics(
            path="gateway_static",
            requests=4,
            worker_deltas=worker_deltas,
            router_deltas=router_deltas,
        )
